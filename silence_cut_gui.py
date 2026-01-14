# -*- coding: UTF-8 -*-
"""
SilenceCut GUI - 自动静音剪辑工具
基于 PyQt6 的本地桌面客户端
"""

import sys
import os
import json
import math
import subprocess
import time
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.io import wavfile

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QRadioButton, QCheckBox, QListWidget, QListWidgetItem,
    QTextEdit, QProgressBar, QFileDialog, QMessageBox, QButtonGroup,
    QSplitter, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QMimeData
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont, QIcon


# ============ 常量定义 ============

class AudioConfig:
    """音频处理常量"""
    DEFAULT_SAMPLE_RATE: int = 44100
    DEFAULT_CHANNELS: int = 2
    OUTPUT_CODEC: str = "aac"
    OUTPUT_BITRATE: str = "192k"


class VideoConfig:
    """视频处理常量"""
    DEFAULT_FRAME_RATE: float = 25.0
    DEFAULT_WIDTH: int = 1920
    DEFAULT_HEIGHT: int = 1080
    SEGMENT_CRF: int = 18
    SEGMENT_PRESET: str = "ultrafast"
    OUTPUT_CRF_HIGH: int = 20
    OUTPUT_CRF_LOW: int = 22


class ProcessConfig:
    """处理流程常量"""
    MIN_SEGMENT_DURATION: float = 0.05
    ATEMPO_MIN: float = 0.5
    ATEMPO_MAX: float = 2.0
    PROGRESS_LOG_INTERVAL: int = 50


class BitrateConfig:
    """码率配置"""
    BITRATE_TABLE: Dict[int, tuple] = {
        3840 * 2160: (25000, 15000, 40000),
        2560 * 1440: (15000, 10000, 25000),
        1920 * 1080: (8000, 5000, 15000),
        1280 * 720: (5000, 3000, 10000),
        854 * 480: (2500, 1500, 5000),
        0: (1500, 800, 3000),
    }

    @classmethod
    def calculate(cls, width: int, height: int, original_bitrate: int = 0, frame_rate: float = 30.0) -> int:
        pixels = width * height
        recommended, min_br, max_br = cls.BITRATE_TABLE[0]
        for threshold, values in sorted(cls.BITRATE_TABLE.items(), reverse=True):
            if pixels >= threshold:
                recommended, min_br, max_br = values
                break
        if frame_rate > 50:
            recommended = int(recommended * 1.3)
            max_br = int(max_br * 1.3)
        if original_bitrate > 0:
            return max(min_br, min(original_bitrate, max_br))
        return recommended


# ============ 数据类 ============

@dataclass
class VideoInfo:
    """视频元信息"""
    frame_rate: float = VideoConfig.DEFAULT_FRAME_RATE
    sample_rate: int = AudioConfig.DEFAULT_SAMPLE_RATE
    width: int = VideoConfig.DEFAULT_WIDTH
    height: int = VideoConfig.DEFAULT_HEIGHT
    video_bitrate: int = 0
    audio_bitrate: int = 0
    video_codec: str = ""
    audio_codec: str = ""
    duration: float = 0.0

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    @property
    def resolution_str(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass
class AudioChunk:
    """音频片段信息"""
    start_frame: int
    end_frame: int
    is_loud: bool

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    def get_time_range(self, frame_rate: float) -> tuple:
        return self.start_frame / frame_rate, self.end_frame / frame_rate


@dataclass
class ProcessResult:
    """处理结果"""
    input_path: str
    output_path: str = ""
    success: bool = False
    elapsed: float = 0.0
    error: str = ""


# ============ 工具函数 ============

def run_command(cmd: str, log_callback=None) -> int:
    """执行命令"""
    try:
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace"
        )
        for line in process.stdout:
            if log_callback:
                log_callback(line.rstrip())
        process.wait()
        return process.returncode
    except OSError as e:
        if log_callback:
            log_callback(f"⚠️ 命令执行异常: {e}")
        return -1


def get_media_duration(file_path: str) -> float:
    """获取媒体文件时长"""
    cmd = f'ffprobe -v quiet -print_format json -show_format "{file_path}"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8", errors="replace")
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0.0


def build_atempo_filter(speed: float) -> str:
    """构建 atempo 音频滤镜"""
    if abs(speed - 1.0) < 0.001:
        return "anull"
    filters = []
    s = speed
    while s > ProcessConfig.ATEMPO_MAX:
        filters.append(f"atempo={ProcessConfig.ATEMPO_MAX}")
        s /= ProcessConfig.ATEMPO_MAX
    while s < ProcessConfig.ATEMPO_MIN:
        filters.append(f"atempo={ProcessConfig.ATEMPO_MIN}")
        s /= ProcessConfig.ATEMPO_MIN
    if abs(s - 1.0) > 0.001:
        filters.append(f"atempo={s}")
    return ",".join(filters) if filters else "anull"


# ============ 编码器配置 ============

class EncoderConfig:
    """编码器配置管理"""

    @staticmethod
    def is_nvenc_available() -> bool:
        """检测 NVENC 硬件编码是否可用"""
        cmd = "ffmpeg -hide_banner -f lavfi -i nullsrc=s=256x256:d=1 -c:v h264_nvenc -f null - 2>&1"
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True,
                                    encoding="utf-8", errors="replace", timeout=10)
            if any(x in result.stderr for x in ("Cannot load", "not found", "No NVENC")):
                return False
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    @staticmethod
    def get_encoder_params(video_info: VideoInfo, use_gpu: bool = True) -> str:
        """获取编码器参数"""
        video_bitrate = BitrateConfig.calculate(
            video_info.width, video_info.height,
            video_info.video_bitrate, video_info.frame_rate
        )
        max_bitrate = int(video_bitrate * 1.2)
        buf_size = max_bitrate * 2
        audio_params = f"-c:a {AudioConfig.OUTPUT_CODEC} -b:a {AudioConfig.OUTPUT_BITRATE}"

        if use_gpu:
            return (f"-c:v h264_nvenc -preset p5 -rc vbr -b:v {video_bitrate}k "
                    f"-maxrate {max_bitrate}k -bufsize {buf_size}k {audio_params}")
        else:
            crf = VideoConfig.OUTPUT_CRF_HIGH if video_info.pixel_count >= 1920 * 1080 else VideoConfig.OUTPUT_CRF_LOW
            return f"-c:v libx264 -preset medium -crf {crf} {audio_params}"


# ============ 静音检测器 ============

class NumpySilenceDetector:
    """基于 NumPy 的静音检测器"""

    def detect(self, audio_path: str, frame_rate: float, sample_rate: int,
               threshold: float, frame_margin: int, log_callback=None) -> List[AudioChunk]:
        if log_callback:
            log_callback("  📊 检测方式: NumPy")

        _, audio_data = wavfile.read(audio_path)
        audio_sample_count = audio_data.shape[0]
        max_volume = max(float(np.max(audio_data)), -float(np.min(audio_data)))
        max_volume = max_volume if max_volume > 0 else 1.0

        samples_per_frame = sample_rate / frame_rate
        audio_frame_count = int(math.ceil(audio_sample_count / samples_per_frame))
        is_loud_per_frame = np.zeros(audio_frame_count, dtype=np.int8)

        for i in range(audio_frame_count):
            start = int(i * samples_per_frame)
            end = min(int((i + 1) * samples_per_frame), audio_sample_count)
            chunk = audio_data[start:end]
            if len(chunk) == 0:
                continue
            max_chunk_volume = max(float(np.max(chunk)), -float(np.min(chunk)))
            if max_chunk_volume / max_volume >= threshold:
                is_loud_per_frame[i] = 1

        should_include = np.zeros(audio_frame_count, dtype=np.int8)
        for i in range(audio_frame_count):
            start = max(0, i - frame_margin)
            end = min(audio_frame_count, i + 1 + frame_margin)
            should_include[i] = np.max(is_loud_per_frame[start:end])

        return self._generate_chunks(should_include, audio_frame_count)

    @staticmethod
    def _generate_chunks(should_include: np.ndarray, frame_count: int) -> List[AudioChunk]:
        if frame_count == 0:
            return []
        chunks = []
        chunk_start = 0
        current_loud = bool(should_include[0])

        for i in range(1, frame_count):
            if bool(should_include[i]) != current_loud:
                chunks.append(AudioChunk(chunk_start, i, current_loud))
                chunk_start = i
                current_loud = bool(should_include[i])

        chunks.append(AudioChunk(chunk_start, frame_count, current_loud))
        return chunks


class PydubSilenceDetector:
    """基于 Pydub 的静音检测器"""

    def __init__(self, min_silence_len: int = 300):
        self.min_silence_len = min_silence_len

    def detect(self, audio_path: str, frame_rate: float, sample_rate: int,
               threshold: float, frame_margin: int, log_callback=None) -> List[AudioChunk]:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent

        if log_callback:
            log_callback("  📊 检测方式: Pydub")

        audio = AudioSegment.from_wav(audio_path)
        duration_ms = len(audio)
        silence_thresh_db = audio.dBFS + (20 * math.log10(threshold + 0.0001))
        if log_callback:
            log_callback(f"  📊 静音阈值: {silence_thresh_db:.2f} dB")

        nonsilent_ranges = detect_nonsilent(
            audio, min_silence_len=self.min_silence_len,
            silence_thresh=silence_thresh_db, seek_step=int(1000 / frame_rate)
        )

        ms_per_frame = 1000 / frame_rate
        total_frames = int(math.ceil(duration_ms / ms_per_frame))
        return self._build_chunks(nonsilent_ranges, total_frames, ms_per_frame, frame_margin)

    @staticmethod
    def _build_chunks(nonsilent_ranges: List[tuple], total_frames: int,
                      ms_per_frame: float, frame_margin: int) -> List[AudioChunk]:
        chunks = []
        last_end = 0

        for start_ms, end_ms in nonsilent_ranges:
            start_frame = max(0, int(start_ms / ms_per_frame) - frame_margin)
            end_frame = min(total_frames, int(math.ceil(end_ms / ms_per_frame)) + frame_margin)

            if start_frame > last_end:
                chunks.append(AudioChunk(last_end, start_frame, False))

            if chunks and chunks[-1].is_loud and start_frame <= chunks[-1].end_frame:
                chunks[-1] = AudioChunk(chunks[-1].start_frame, end_frame, True)
            else:
                chunks.append(AudioChunk(start_frame, end_frame, True))
            last_end = end_frame

        if last_end < total_frames:
            chunks.append(AudioChunk(last_end, total_frames, False))
        if not chunks:
            chunks.append(AudioChunk(0, total_frames, True))
        return chunks


# ============ FFmpeg 处理器 ============

class FFmpegProcessor:
    """FFmpeg 视频处理器"""

    def __init__(self, log_callback=None):
        self.platform = platform.system()
        self.log_callback = log_callback

    def log(self, msg: str):
        if self.log_callback:
            self.log_callback(msg)

    def process(self, input_file: str, output_file: str, chunks: List[AudioChunk],
                silent_speed: float, sounded_speed: float, frame_rate: float,
                temp_folder: str, encoder_params: str = "", global_speed: float = 1.0) -> bool:
        self.log(f"\n{'=' * 50}\n步骤1: 处理静音片段\n{'=' * 50}")

        segments = self._process_chunks(input_file, chunks, silent_speed, sounded_speed, frame_rate, temp_folder)
        if not segments:
            self.log("  ❌ 没有有效片段")
            return False

        intermediate_file = self._merge_segments(segments, temp_folder)
        if not intermediate_file:
            return False

        return self._apply_global_speed_and_encode(intermediate_file, output_file, global_speed, encoder_params)

    def _process_chunks(self, input_file: str, chunks: List[AudioChunk], silent_speed: float,
                        sounded_speed: float, frame_rate: float, temp_folder: str) -> List[str]:
        segments = []
        total, skipped = len(chunks), 0

        for i, chunk in enumerate(chunks):
            start_sec, end_sec = chunk.get_time_range(frame_rate)
            duration = end_sec - start_sec
            if duration < ProcessConfig.MIN_SEGMENT_DURATION:
                continue

            speed = sounded_speed if chunk.is_loud else silent_speed
            if speed >= 100:
                skipped += 1
                continue

            seg_file = os.path.join(temp_folder, f"seg_{len(segments):06d}.mp4")
            if self._extract_segment(input_file, seg_file, start_sec, duration, speed):
                segments.append(seg_file)

            if (i + 1) % ProcessConfig.PROGRESS_LOG_INTERVAL == 0:
                self.log(f"  进度: {i + 1}/{total}, 已生成: {len(segments)}, 跳过静音: {skipped}")

        self.log(f"  ✅ 生成 {len(segments)} 个片段, 跳过 {skipped} 个静音片段")
        return segments

    def _extract_segment(self, input_file: str, output_file: str,
                         start_sec: float, duration: float, speed: float) -> bool:
        crf, preset = VideoConfig.SEGMENT_CRF, VideoConfig.SEGMENT_PRESET

        if abs(speed - 1.0) > 0.01:
            atempo, pts = build_atempo_filter(speed), 1.0 / speed
            cmd = (f'ffmpeg -hide_banner -y -i "{input_file}" -ss {start_sec} -t {duration} '
                   f'-vf "setpts={pts}*PTS" -af "{atempo}" -c:v libx264 -preset {preset} '
                   f'-crf {crf} -c:a {AudioConfig.OUTPUT_CODEC} "{output_file}"')
        else:
            cmd = (f'ffmpeg -hide_banner -y -i "{input_file}" -ss {start_sec} -t {duration} '
                   f'-c:v libx264 -preset {preset} -crf {crf} -c:a {AudioConfig.OUTPUT_CODEC} "{output_file}"')

        return run_command(cmd) == 0 and os.path.exists(output_file)

    def _merge_segments(self, segments: List[str], temp_folder: str) -> Optional[str]:
        concat_file = os.path.join(temp_folder, "concat.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"file '{os.path.basename(seg)}'\n")

        intermediate_file = os.path.join(temp_folder, "merged.mp4")
        self.log("\n  合并片段...")
        cmd = f'ffmpeg -hide_banner -y -f concat -safe 0 -i "{concat_file}" -c copy "{intermediate_file}"'

        if run_command(cmd) != 0:
            self.log("  ❌ 合并失败")
            return None

        self.log(f"  ✅ 合并完成, 时长: {get_media_duration(intermediate_file):.2f}s")
        return intermediate_file

    def _apply_global_speed_and_encode(self, input_file: str, output_file: str,
                                       global_speed: float, encoder_params: str) -> bool:
        input_duration = get_media_duration(input_file)
        self.log(f"\n{'=' * 50}\n步骤2: 全局加速 {global_speed}x\n{'=' * 50}")

        if abs(global_speed - 1.0) > 0.01:
            atempo, pts = build_atempo_filter(global_speed), 1.0 / global_speed
            self.log(f"  视频 PTS: {pts}\n  音频滤镜: {atempo}")
            cmd = (f'ffmpeg -hide_banner -y -i "{input_file}" -vf "setpts={pts}*PTS" '
                   f'-af "{atempo}" {encoder_params} "{output_file}"')
            expected_duration = input_duration / global_speed
            self.log(f"  预期输出时长: {expected_duration:.2f}s")
        else:
            self.log("  全局加速为 1.0x，跳过加速步骤")
            cmd = f'ffmpeg -hide_banner -y -i "{input_file}" {encoder_params} "{output_file}"'
            expected_duration = input_duration

        self.log("\n  执行最终编码...")
        if run_command(cmd, self.log_callback) != 0:
            self.log("  ❌ 编码失败")
            return False

        actual_duration = get_media_duration(output_file)
        self.log(f"\n  ✅ 完成!\n     预期时长: {expected_duration:.2f}s\n     实际时长: {actual_duration:.2f}s")
        return True


# ============ 主处理类 ============

class SilenceCut:
    """自动剪辑主类"""

    def __init__(self, input_file: str, output_file: Optional[str] = None,
                 silent_speed: float = 999, sounded_speed: float = 1.0,
                 global_speed: float = 1.0, frame_margin: int = 3,
                 silent_threshold: float = 0.025, use_pydub: bool = True,
                 use_gpu: bool = True, min_silence_len: int = 300,
                 log_callback=None):
        self.input_file = input_file
        self.output_file = output_file or self._generate_output_name(input_file)
        self.silent_speed = silent_speed
        self.sounded_speed = sounded_speed
        self.global_speed = global_speed
        self.frame_margin = frame_margin
        self.silent_threshold = silent_threshold
        self.use_pydub = use_pydub
        self.use_gpu = use_gpu
        self.min_silence_len = min_silence_len
        self.log_callback = log_callback

        self._temp_folder: Optional[str] = None
        self._video_info: Optional[VideoInfo] = None
        self._gpu_available: bool = False

    def log(self, msg: str):
        if self.log_callback:
            self.log_callback(msg)

    @staticmethod
    def _generate_output_name(input_path: str) -> str:
        base, ext = os.path.splitext(input_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base}_{timestamp}{ext}"

    def _probe_video_info(self) -> VideoInfo:
        cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams "{self.input_file}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8", errors="replace")
        info = VideoInfo()

        try:
            data = json.loads(result.stdout)
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    fps_str = stream.get("r_frame_rate", "25/1")
                    if "/" in fps_str:
                        n, d = fps_str.split("/")
                        info.frame_rate = float(n) / float(d) if float(d) else 25
                    info.width = stream.get("width", VideoConfig.DEFAULT_WIDTH)
                    info.height = stream.get("height", VideoConfig.DEFAULT_HEIGHT)
                    if "bit_rate" in stream:
                        info.video_bitrate = int(stream["bit_rate"]) // 1000
                elif stream.get("codec_type") == "audio":
                    info.sample_rate = int(stream.get("sample_rate", AudioConfig.DEFAULT_SAMPLE_RATE))

            fmt = data.get("format", {})
            if info.video_bitrate == 0 and "bit_rate" in fmt:
                info.video_bitrate = int(fmt["bit_rate"]) // 1000
            info.duration = float(fmt.get("duration", 0))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.log(f"  ⚠️ 视频信息解析异常: {e}")

        return info

    def _setup_temp_folder(self) -> None:
        self._temp_folder = os.path.splitext(self.input_file)[0] + "_TEMP"
        if os.path.exists(self._temp_folder):
            rmtree(self._temp_folder)
        os.makedirs(self._temp_folder)

    def _cleanup(self, ignore_errors: bool = False) -> None:
        if self._temp_folder and os.path.exists(self._temp_folder):
            rmtree(self._temp_folder, ignore_errors=ignore_errors)

    def _extract_audio(self, output_path: str) -> None:
        cmd = (f'ffmpeg -hide_banner -y -i "{self.input_file}" -vn -acodec pcm_s16le '
               f'-ar {self._video_info.sample_rate} -ac {AudioConfig.DEFAULT_CHANNELS} "{output_path}"')
        run_command(cmd)

    def _detect_silence(self) -> List[AudioChunk]:
        audio_path = os.path.join(self._temp_folder, "audio.wav")
        self._extract_audio(audio_path)

        if self.use_pydub:
            detector = PydubSilenceDetector(self.min_silence_len)
        else:
            detector = NumpySilenceDetector()

        return detector.detect(audio_path, self._video_info.frame_rate,
                               self._video_info.sample_rate, self.silent_threshold,
                               self.frame_margin, self.log_callback)

    def _process_video(self, chunks: List[AudioChunk]) -> bool:
        encoder_params = EncoderConfig.get_encoder_params(self._video_info, self.use_gpu and self._gpu_available)
        processor = FFmpegProcessor(self.log_callback)
        return processor.process(self.input_file, self.output_file, chunks,
                                 self.silent_speed, self.sounded_speed, self._video_info.frame_rate,
                                 temp_folder=self._temp_folder, encoder_params=encoder_params,
                                 global_speed=self.global_speed)

    def run(self) -> str:
        """执行剪辑处理"""
        self._print_header()
        self._setup_temp_folder()

        try:
            self.log("\n[1/4] 分析视频...")
            self._video_info = self._probe_video_info()
            self.log(f"  分辨率: {self._video_info.resolution_str}")
            self.log(f"  帧率: {self._video_info.frame_rate:.2f} fps")
            self.log(f"  时长: {self._video_info.duration:.2f}s")
            self.log(f"  码率: {self._video_info.video_bitrate} kbps")

            self.log("\n[2/4] 检测编码器...")
            if self.use_gpu:
                self._gpu_available = EncoderConfig.is_nvenc_available()
                self.log(f"  GPU: {'可用 ✅' if self._gpu_available else '不可用 ❌ (将使用CPU)'}")

            self.log("\n[3/4] 检测静音...")
            chunks = self._detect_silence()
            loud_count = sum(1 for c in chunks if c.is_loud)
            self.log(f"  片段: {len(chunks)} (有声: {loud_count}, 静音: {len(chunks) - loud_count})")

            self.log("\n[4/4] 处理视频...")
            success = self._process_video(chunks)
            self._cleanup()

            if success:
                self.log(f"\n{'=' * 60}\n✅ 完成!\n📁 {self.output_file}\n{'=' * 60}")
                return self.output_file
            return ""

        except Exception as e:
            self._cleanup(ignore_errors=True)
            self.log(f"❌ 处理异常: {e}")
            raise

    def _print_header(self) -> None:
        self.log("=" * 60)
        self.log("🎬 自动剪辑工具 - SilenceCut")
        self.log("=" * 60)
        self.log(f"📁 输入: {self.input_file}")
        self.log(f"📁 输出: {self.output_file}")
        self.log(f"⏩ 静音倍速: {self.silent_speed} | 有声倍速: {self.sounded_speed}")
        self.log(f"🚀 全局加速: {self.global_speed}x")
        self.log("=" * 60)


# ============ 处理线程 ============

class ProcessThread(QThread):
    """后台处理线程"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, str)  # current, total, filename
    finished_signal = pyqtSignal(list)  # results

    def __init__(self, files: List[str], config: Dict[str, Any], output_dir: str,
                 parallel: bool = False, workers: int = 2):
        super().__init__()
        self.files = files
        self.config = config
        self.output_dir = output_dir
        self.parallel = parallel
        self.workers = workers
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def log(self, msg: str):
        self.log_signal.emit(msg)

    def _generate_output_path(self, input_path: str) -> str:
        """生成输出文件路径"""
        base_name = os.path.basename(input_path)
        name, ext = os.path.splitext(base_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"{name}_{timestamp}{ext}")

    def _process_single(self, input_file: str) -> ProcessResult:
        """处理单个文件"""
        result = ProcessResult(input_path=input_file)
        if not os.path.exists(input_file):
            result.error = "文件不存在"
            return result

        start_time = time.time()
        try:
            output_file = self._generate_output_path(input_file)
            cutter = SilenceCut(
                input_file=input_file,
                output_file=output_file,
                silent_speed=self.config.get("silent_speed", 999),
                sounded_speed=self.config.get("sounded_speed", 1.0),
                global_speed=self.config.get("global_speed", 1.0),
                frame_margin=self.config.get("frame_margin", 3),
                silent_threshold=self.config.get("silent_threshold", 0.025),
                use_pydub=self.config.get("use_pydub", True),
                use_gpu=self.config.get("use_gpu", True),
                min_silence_len=self.config.get("min_silence_len", 300),
                log_callback=self.log
            )
            output = cutter.run()
            result.output_path = output
            result.success = bool(output)
        except Exception as e:
            result.error = str(e)

        result.elapsed = time.time() - start_time
        return result

    def run(self):
        results = []
        total = len(self.files)

        if self.parallel and total > 1:
            self.log(f"\n{'=' * 60}\n📦 并行批量处理 - 共 {total} 个文件, 并行数: {self.workers}\n{'=' * 60}")
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(self._process_single, f): f for f in self.files}
                completed = 0
                for future in as_completed(futures):
                    if self._stop_flag:
                        break
                    completed += 1
                    result = future.result()
                    results.append(result)
                    self.progress_signal.emit(completed, total, os.path.basename(result.input_path))
        else:
            for i, input_file in enumerate(self.files):
                if self._stop_flag:
                    break
                self.progress_signal.emit(i + 1, total, os.path.basename(input_file))
                result = self._process_single(input_file)
                results.append(result)

        self.finished_signal.emit(results)


# ============ 文件列表控件 ============

class FileListWidget(QListWidget):
    """支持拖拽的文件列表"""

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setMinimumHeight(120)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')):
                    self.add_file(file_path)
            event.acceptProposedAction()

    def add_file(self, file_path: str):
        """添加文件，避免重复"""
        for i in range(self.count()):
            if self.item(i).data(Qt.ItemDataRole.UserRole) == file_path:
                return
        item = QListWidgetItem(os.path.basename(file_path))
        item.setData(Qt.ItemDataRole.UserRole, file_path)
        item.setToolTip(file_path)
        self.addItem(item)

    def get_files(self) -> List[str]:
        """获取所有文件路径"""
        return [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]

    def remove_selected(self):
        """删除选中项"""
        for item in self.selectedItems():
            self.takeItem(self.row(item))


# ============ 主窗口 ============

class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.process_thread: Optional[ProcessThread] = None
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        self.setWindowTitle("🎬 SilenceCut - 自动静音剪辑工具")
        self.setMinimumSize(800, 700)

        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)

        # ===== 文件列表区域 =====
        file_group = QGroupBox("📁 视频文件列表 (支持拖拽添加)")
        file_layout = QVBoxLayout(file_group)

        self.file_list = FileListWidget()
        file_layout.addWidget(self.file_list)

        file_btn_layout = QHBoxLayout()
        self.btn_add_files = QPushButton("➕ 添加文件")
        self.btn_add_files.clicked.connect(self.add_files)
        self.btn_remove_selected = QPushButton("➖ 移除选中")
        self.btn_remove_selected.clicked.connect(self.file_list.remove_selected)
        self.btn_clear_list = QPushButton("🗑️ 清空列表")
        self.btn_clear_list.clicked.connect(self.file_list.clear)
        file_btn_layout.addWidget(self.btn_add_files)
        file_btn_layout.addWidget(self.btn_remove_selected)
        file_btn_layout.addWidget(self.btn_clear_list)
        file_btn_layout.addStretch()
        file_layout.addLayout(file_btn_layout)

        main_layout.addWidget(file_group)

        # ===== 参数设置区域 =====
        params_layout = QHBoxLayout()

        # 左侧：静音处理参数
        silence_group = QGroupBox("⚙️ 静音处理参数")
        silence_layout = QVBoxLayout(silence_group)

        # 静音速度
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("静音速度:"))
        self.spin_silent_speed = QDoubleSpinBox()
        self.spin_silent_speed.setRange(0.1, 999)
        self.spin_silent_speed.setValue(999)
        self.spin_silent_speed.setToolTip("≥100 表示删除静音片段")
        row1.addWidget(self.spin_silent_speed)
        row1.addWidget(QLabel("有声速度:"))
        self.spin_sounded_speed = QDoubleSpinBox()
        self.spin_sounded_speed.setRange(0.1, 10)
        self.spin_sounded_speed.setValue(1.0)
        self.spin_sounded_speed.setSingleStep(0.1)
        row1.addWidget(self.spin_sounded_speed)
        silence_layout.addLayout(row1)

        # 全局加速和边界帧
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("全局加速:"))
        self.spin_global_speed = QDoubleSpinBox()
        self.spin_global_speed.setRange(0.5, 5.0)
        self.spin_global_speed.setValue(1.0)
        self.spin_global_speed.setSingleStep(0.1)
        row2.addWidget(self.spin_global_speed)
        row2.addWidget(QLabel("边界缓冲帧:"))
        self.spin_frame_margin = QSpinBox()
        self.spin_frame_margin.setRange(0, 20)
        self.spin_frame_margin.setValue(3)
        row2.addWidget(self.spin_frame_margin)
        silence_layout.addLayout(row2)

        # 静音阈值和最小静音时长
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("静音阈值:"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.001, 1.0)
        self.spin_threshold.setValue(0.025)
        self.spin_threshold.setSingleStep(0.005)
        self.spin_threshold.setDecimals(3)
        row3.addWidget(self.spin_threshold)
        row3.addWidget(QLabel("最小静音(ms):"))
        self.spin_min_silence = QSpinBox()
        self.spin_min_silence.setRange(50, 2000)
        self.spin_min_silence.setValue(300)
        self.spin_min_silence.setSingleStep(50)
        row3.addWidget(self.spin_min_silence)
        silence_layout.addLayout(row3)

        params_layout.addWidget(silence_group)

        # 右侧：编码设置
        encode_group = QGroupBox("🎛️ 编码设置")
        encode_layout = QVBoxLayout(encode_group)

        # 编码器选择
        encoder_row = QHBoxLayout()
        encoder_row.addWidget(QLabel("编码器:"))
        self.radio_gpu = QRadioButton("GPU (NVENC)")
        self.radio_cpu = QRadioButton("CPU (libx264)")
        self.radio_gpu.setChecked(True)
        self.encoder_group = QButtonGroup()
        self.encoder_group.addButton(self.radio_gpu, 1)
        self.encoder_group.addButton(self.radio_cpu, 2)
        encoder_row.addWidget(self.radio_gpu)
        encoder_row.addWidget(self.radio_cpu)
        encoder_row.addStretch()
        encode_layout.addLayout(encoder_row)

        # 检测方式
        detect_row = QHBoxLayout()
        detect_row.addWidget(QLabel("检测方式:"))
        self.radio_pydub = QRadioButton("Pydub")
        self.radio_numpy = QRadioButton("NumPy")
        self.radio_pydub.setChecked(True)
        self.detect_group = QButtonGroup()
        self.detect_group.addButton(self.radio_pydub, 1)
        self.detect_group.addButton(self.radio_numpy, 2)
        detect_row.addWidget(self.radio_pydub)
        detect_row.addWidget(self.radio_numpy)
        detect_row.addStretch()
        encode_layout.addLayout(detect_row)

        # 处理模式
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("处理模式:"))
        self.radio_serial = QRadioButton("串行")
        self.radio_parallel = QRadioButton("并行")
        self.radio_serial.setChecked(True)
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_serial, 1)
        self.mode_group.addButton(self.radio_parallel, 2)
        mode_row.addWidget(self.radio_serial)
        mode_row.addWidget(self.radio_parallel)
        mode_row.addWidget(QLabel("并行数:"))
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, 8)
        self.spin_workers.setValue(2)
        mode_row.addWidget(self.spin_workers)
        mode_row.addStretch()
        encode_layout.addLayout(mode_row)

        params_layout.addWidget(encode_group)
        main_layout.addLayout(params_layout)

        # ===== 输出设置 =====
        output_group = QGroupBox("📂 输出设置")
        output_layout = QHBoxLayout(output_group)
        output_layout.addWidget(QLabel("输出目录:"))
        self.edit_output_dir = QLineEdit()
        self.edit_output_dir.setText(str(Path.home() / "Videos" / "SilenceCut_Output"))
        output_layout.addWidget(self.edit_output_dir)
        self.btn_browse_output = QPushButton("浏览...")
        self.btn_browse_output.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(self.btn_browse_output)
        main_layout.addWidget(output_group)

        # ===== 进度区域 =====
        progress_group = QGroupBox("📊 处理进度")
        progress_layout = QVBoxLayout(progress_group)

        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m - %p%")
        progress_row.addWidget(self.progress_bar)
        self.label_current_file = QLabel("")
        progress_row.addWidget(self.label_current_file)
        progress_layout.addLayout(progress_row)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        self.log_text.setStyleSheet("QTextEdit { font-family: 'Consolas', 'Courier New', monospace; font-size: 11px; }")
        progress_layout.addWidget(self.log_text)

        main_layout.addWidget(progress_group)

        # ===== 操作按钮 =====
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("🚀 开始处理")
        self.btn_start.setStyleSheet("QPushButton { font-size: 14px; font-weight: bold; padding: 10px 30px; }")
        self.btn_start.clicked.connect(self.start_process)
        self.btn_stop = QPushButton("⏹️ 停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_process)
        self.btn_open_output = QPushButton("📂 打开输出目录")
        self.btn_open_output.clicked.connect(self.open_output_dir)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_open_output)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

    def add_files(self):
        """添加文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;所有文件 (*.*)"
        )
        for f in files:
            self.file_list.add_file(f)

    def browse_output_dir(self):
        """选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.edit_output_dir.setText(dir_path)

    def open_output_dir(self):
        """打开输出目录"""
        output_dir = self.edit_output_dir.text()
        if os.path.exists(output_dir):
            if platform.system() == "Windows":
                os.startfile(output_dir)
            elif platform.system() == "Darwin":
                subprocess.run(["open", output_dir])
            else:
                subprocess.run(["xdg-open", output_dir])
        else:
            QMessageBox.warning(self, "提示", "输出目录不存在")

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return {
            "silent_speed": self.spin_silent_speed.value(),
            "sounded_speed": self.spin_sounded_speed.value(),
            "global_speed": self.spin_global_speed.value(),
            "frame_margin": self.spin_frame_margin.value(),
            "silent_threshold": self.spin_threshold.value(),
            "min_silence_len": self.spin_min_silence.value(),
            "use_gpu": self.radio_gpu.isChecked(),
            "use_pydub": self.radio_pydub.isChecked(),
        }

    def start_process(self):
        """开始处理"""
        files = self.file_list.get_files()
        if not files:
            QMessageBox.warning(self, "提示", "请先添加要处理的视频文件")
            return

        output_dir = self.edit_output_dir.text()
        if not output_dir:
            QMessageBox.warning(self, "提示", "请设置输出目录")
            return

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 清空日志
        self.log_text.clear()

        # 设置进度条
        self.progress_bar.setMaximum(len(files))
        self.progress_bar.setValue(0)

        # 禁用开始按钮
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        # 创建处理线程
        self.process_thread = ProcessThread(
            files=files,
            config=self.get_config(),
            output_dir=output_dir,
            parallel=self.radio_parallel.isChecked(),
            workers=self.spin_workers.value()
        )
        self.process_thread.log_signal.connect(self.append_log)
        self.process_thread.progress_signal.connect(self.update_progress)
        self.process_thread.finished_signal.connect(self.on_process_finished)
        self.process_thread.start()

    def stop_process(self):
        """停止处理"""
        if self.process_thread:
            self.process_thread.stop()
            self.append_log("\n⚠️ 用户请求停止处理...")

    def append_log(self, msg: str):
        """追加日志"""
        self.log_text.append(msg)
        # 滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, current: int, total: int, filename: str):
        """更新进度"""
        self.progress_bar.setValue(current)
        self.label_current_file.setText(f"处理中: {filename}")

    def on_process_finished(self, results: List[ProcessResult]):
        """处理完成"""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.label_current_file.setText("处理完成")

        # 统计结果
        success_count = sum(1 for r in results if r.success)
        total = len(results)
        total_time = sum(r.elapsed for r in results)

        self.append_log(f"\n{'=' * 60}")
        self.append_log(f"📊 处理完成汇总")
        self.append_log(f"{'=' * 60}")
        self.append_log(f"  成功: {success_count}/{total}")
        self.append_log(f"  失败: {total - success_count}/{total}")
        self.append_log(f"  总耗时: {total_time:.1f}s ({total_time / 60:.1f}min)")

        if total - success_count > 0:
            self.append_log("\n  ❌ 失败文件:")
            for r in results:
                if not r.success:
                    self.append_log(f"     - {os.path.basename(r.input_path)}: {r.error}")

        QMessageBox.information(
            self, "处理完成",
            f"处理完成！\n成功: {success_count}/{total}\n总耗时: {total_time:.1f}秒"
        )

    def save_settings(self):
        """保存设置"""
        settings = {
            "output_dir": self.edit_output_dir.text(),
            **self.get_config(),
            "parallel": self.radio_parallel.isChecked(),
            "workers": self.spin_workers.value()
        }
        settings_path = Path.home() / ".silencecut_settings.json"
        try:
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass

    def load_settings(self):
        """加载设置"""
        settings_path = Path.home() / ".silencecut_settings.json"
        if settings_path.exists():
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                self.edit_output_dir.setText(settings.get("output_dir", self.edit_output_dir.text()))
                self.spin_silent_speed.setValue(settings.get("silent_speed", 999))
                self.spin_sounded_speed.setValue(settings.get("sounded_speed", 1.0))
                self.spin_global_speed.setValue(settings.get("global_speed", 1.0))
                self.spin_frame_margin.setValue(settings.get("frame_margin", 3))
                self.spin_threshold.setValue(settings.get("silent_threshold", 0.025))
                self.spin_min_silence.setValue(settings.get("min_silence_len", 300))
                self.radio_gpu.setChecked(settings.get("use_gpu", True))
                self.radio_cpu.setChecked(not settings.get("use_gpu", True))
                self.radio_pydub.setChecked(settings.get("use_pydub", True))
                self.radio_numpy.setChecked(not settings.get("use_pydub", True))
                self.radio_parallel.setChecked(settings.get("parallel", False))
                self.radio_serial.setChecked(not settings.get("parallel", False))
                self.spin_workers.setValue(settings.get("workers", 2))
            except Exception:
                pass

    def closeEvent(self, event):
        """关闭窗口时保存设置"""
        self.save_settings()
        if self.process_thread and self.process_thread.isRunning():
            reply = QMessageBox.question(
                self, "确认退出",
                "处理正在进行中，确定要退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self.process_thread.stop()
            self.process_thread.wait(3000)
        event.accept()


# ============ 主入口 ============

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # 设置应用字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
