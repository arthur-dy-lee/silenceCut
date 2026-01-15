# -*- coding: UTF-8 -*-
"""
SilenceCut GUI v2.1 - 自动静音剪辑工具
功能: 自动检测并删除/加速视频中的静音片段
特性: 中英文切换、GPU/CPU编码、批量处理
"""

import sys, os, json, math, subprocess, time, platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.io import wavfile
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QAction, QPalette, QColor


# ==================== 国际化 ====================
class I18n:
    TEXTS = {
        "window_title": {"zh": "🎬 SilenceCut - 自动静音剪辑工具", "en": "🎬 SilenceCut - Auto Silence Cutter"},
        "menu_settings": {"zh": "设置", "en": "Settings"},
        "menu_language": {"zh": "语言", "en": "Language"},
        "file_list_title": {"zh": "📁 视频文件 (支持拖拽添加)", "en": "📁 Videos (Drag & Drop)"},
        "btn_add": {"zh": "➕ 添加文件", "en": "➕ Add Files"},
        "btn_remove": {"zh": "➖ 移除选中", "en": "➖ Remove"},
        "btn_clear": {"zh": "🗑️ 清空列表", "en": "🗑️ Clear All"},
        "params_title": {"zh": "⚙️ 静音处理参数", "en": "⚙️ Silence Parameters"},
        "silent_speed": {"zh": "静音速度:", "en": "Silent Speed:"},
        "silent_speed_tip": {"zh": "静音片段的处理速度\n≥100 = 完全删除静音\n2-4 = 加速播放静音部分",
                             "en": "Speed for silent segments\n≥100 = delete\n2-4 = speed up"},
        "global_speed": {"zh": "视频全局加速:", "en": "Global Speed:"},
        "global_speed_tip": {"zh": "最终视频的整体加速倍率\n1.0 = 原速不变\n1.2 = 整体加速20%",
                             "en": "Overall speed\n1.0 = original\n1.2 = 20% faster"},
        "frame_margin": {"zh": "边界缓冲帧:", "en": "Frame Margin:"},
        "threshold": {"zh": "静音阈值:", "en": "Threshold:"},
        "min_silence": {"zh": "最小静音(ms):", "en": "Min Silence(ms):"},
        "encode_title": {"zh": "🎛️ 编码设置", "en": "🎛️ Encoding Settings"},
        "encoder": {"zh": "编码器:", "en": "Encoder:"},
        "detect": {"zh": "检测方式:", "en": "Detection:"},
        "mode": {"zh": "处理模式:", "en": "Process Mode:"},
        "serial": {"zh": "串行", "en": "Serial"},
        "parallel": {"zh": "并行", "en": "Parallel"},
        "workers": {"zh": "并行数:", "en": "Workers:"},
        "output_title": {"zh": "📂 输出设置", "en": "📂 Output Settings"},
        "use_source": {"zh": "输出到源文件所在目录 (推荐)", "en": "Output to source directory (Recommended)"},
        "output_dir": {"zh": "输出目录:", "en": "Output Dir:"},
        "browse": {"zh": "浏览...", "en": "Browse..."},
        "hint_source": {"zh": "💡 提示: 输出文件将保存在每个源视频的同目录下，文件名自动添加时间戳后缀",
                        "en": "💡 Output saved alongside source video with timestamp suffix"},
        "hint_custom": {"zh": "💡 提示: 所有输出文件将统一保存到上方指定的目录",
                        "en": "💡 All output files saved to the directory above"},
        "progress_title": {"zh": "📊 处理进度", "en": "📊 Progress"},
        "processing": {"zh": "处理中:", "en": "Processing:"},
        "complete": {"zh": "处理完成", "en": "Complete"},
        "idle": {"zh": "等待开始...", "en": "Waiting..."},
        "btn_start": {"zh": "🚀 开始处理", "en": "🚀 Start"},
        "btn_stop": {"zh": "⏹️ 停止", "en": "⏹️ Stop"},
        "btn_open": {"zh": "📂 打开输出目录", "en": "📂 Open Output"},
        "no_files": {"zh": "请先添加要处理的视频文件", "en": "Please add video files first"},
        "no_dir": {"zh": "请设置输出目录", "en": "Please set output directory"},
        "dir_error": {"zh": "目录不存在", "en": "Directory does not exist"},
        "confirm_exit": {"zh": "视频正在处理中，确定要退出吗？", "en": "Processing in progress. Exit?"},
        "exit_title": {"zh": "确认退出", "en": "Confirm Exit"},
        "done_title": {"zh": "处理完成", "en": "Complete"},
        "done_text": {"zh": "处理完成！\n成功: {s}/{t}\n总耗时: {e:.1f} 秒",
                      "en": "Done! Success: {s}/{t}, Time: {e:.1f}s"},
        "tip": {"zh": "提示", "en": "Info"},
        "stopped": {"zh": "⚠️ 用户已停止处理", "en": "⚠️ Stopped by user"},
        "enc_title": {
            "zh": "\n╔═══════════════════════════════════════════════════════════╗\n║                      编 码 器 参 数                        ║\n╚═══════════════════════════════════════════════════════════╝",
            "en": "\n╔═══════════════════════════════════════════════════════════╗\n║                    ENCODER PARAMETERS                      ║\n╚═══════════════════════════════════════════════════════════╝"},
        "enc_type": {"zh": "  🎬 编码器类型:", "en": "  🎬 Encoder:"},
        "enc_bitrate": {"zh": "  📊 视频码率:", "en": "  📊 Bitrate:"},
        "enc_audio": {"zh": "  🔊 音频参数:", "en": "  🔊 Audio:"},
        "enc_params": {"zh": "  📝 完整参数:", "en": "  📝 Full Params:"},
        "batch_serial": {"zh": "📦 串行处理 - {n} 个文件", "en": "📦 Serial - {n} files"},
        "batch_parallel": {"zh": "📦 并行处理 - {n} 文件, {w} 线程", "en": "📦 Parallel - {n} files, {w} workers"},
        "summary": {"zh": "处理完成汇总", "en": "Summary"},
        "success": {"zh": "  ✅ 成功:", "en": "  ✅ Success:"},
        "failed": {"zh": "  ❌ 失败:", "en": "  ❌ Failed:"},
        "total_time": {"zh": "  ⏱️ 总耗时:", "en": "  ⏱️ Time:"},
        "failed_list": {"zh": "❌ 失败文件:", "en": "❌ Failed:"},
    }
    current_lang = "zh"

    @classmethod
    def get(cls, key): return cls.TEXTS.get(key, {}).get(cls.current_lang, key)

    @classmethod
    def set_language(cls, lang): cls.current_lang = lang if lang in ("zh", "en") else "zh"


# ==================== 浅色主题样式 (蓝色单选/进度条) ====================
STYLE_SHEET = """
    QMainWindow { background-color: #f5f5f5; }
    QWidget { color: #222222; }
    QGroupBox { font-weight: bold; color: #222222; border: 1px solid #cccccc; border-radius: 6px; margin-top: 12px; padding-top: 12px; background-color: #ffffff; }
    QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #222222; }
    QLabel { color: #222222; }
    QListWidget { border: 1px solid #cccccc; border-radius: 4px; background-color: #ffffff; color: #222222; }
    QListWidget::item { padding: 4px; color: #222222; }
    QListWidget::item:selected { background-color: #0078d4; color: #ffffff; }
    QTextEdit { border: 1px solid #cccccc; border-radius: 4px; background-color: #ffffff; color: #222222; font-family: Consolas, monospace; font-size: 11px; }
    QLineEdit { border: 1px solid #cccccc; border-radius: 4px; background-color: #ffffff; color: #222222; padding: 4px 8px; }
    QSpinBox, QDoubleSpinBox { border: 1px solid #cccccc; border-radius: 4px; background-color: #ffffff; color: #222222; padding: 4px; }
    QPushButton { padding: 6px 16px; border: 1px solid #cccccc; border-radius: 4px; background-color: #ffffff; color: #222222; }
    QPushButton:hover { background-color: #e6e6e6; }
    QPushButton:disabled { color: #999999; background-color: #f0f0f0; }
    QRadioButton { color: #222222; }
    QCheckBox { color: #222222; spacing: 6px; }
    QCheckBox { color: #222222; }
    QProgressBar { border: 1px solid #cccccc; border-radius: 4px; text-align: center; background-color: #e0e0e0; color: #222222; }
    QProgressBar::chunk { background-color: #0078d4; border-radius: 3px; }
    QMenuBar { background-color: #ffffff; color: #222222; }
    QMenuBar::item { padding: 6px 12px; color: #222222; }
    QMenuBar::item:selected { background-color: #e6e6e6; }
    QMenu { background-color: #ffffff; color: #222222; border: 1px solid #cccccc; }
    QMenu::item { padding: 6px 24px; color: #222222; }
    QMenu::item:selected { background-color: #0078d4; color: #ffffff; }
    QLabel#hint { color: #0066cc; font-weight: bold; padding: 10px; background-color: #e6f3ff; border-radius: 6px; border: 1px solid #99ccff; }
    QMessageBox { background-color: #ffffff; }
    QMessageBox QLabel { color: #222222; background-color: #ffffff; }
    QMessageBox QPushButton { padding: 6px 20px; border: 1px solid #cccccc; border-radius: 4px; background-color: #ffffff; color: #222222; min-width: 70px; }
    QMessageBox QPushButton:hover { background-color: #e6e6e6; }
    QDialog { background-color: #ffffff; }
"""


# ==================== 配置常量 ====================
class AudioCfg:
    RATE = 44100;
    CHANNELS = 2;
    CODEC = "aac";
    BITRATE = "192k"


class VideoCfg:
    FPS = 25.0;
    W = 1920;
    H = 1080;
    CRF_SEG = 18;
    PRESET_SEG = "ultrafast";
    CRF_HI = 20;
    CRF_LO = 22


class ProcCfg:
    MIN_DUR = 0.05;
    ATEMPO_MIN = 0.5;
    ATEMPO_MAX = 2.0;
    LOG_INTERVAL = 50


class BitrateCfg:
    TABLE = {3840 * 2160: (25000, 15000, 40000), 2560 * 1440: (15000, 10000, 25000), 1920 * 1080: (8000, 5000, 15000),
             1280 * 720: (5000, 3000, 10000), 854 * 480: (2500, 1500, 5000), 0: (1500, 800, 3000)}

    @classmethod
    def calc(cls, w, h, orig=0, fps=30.0):
        px = w * h
        rec, mn, mx = cls.TABLE[0]
        for th, v in sorted(cls.TABLE.items(), reverse=True):
            if px >= th: rec, mn, mx = v; break
        if fps > 50: rec = int(rec * 1.3); mx = int(mx * 1.3)
        return max(mn, min(orig, mx)) if orig > 0 else rec


# ==================== 数据类 ====================
@dataclass
class VideoInfo:
    fps: float = VideoCfg.FPS;
    sr: int = AudioCfg.RATE;
    w: int = VideoCfg.W;
    h: int = VideoCfg.H;
    vbr: int = 0;
    dur: float = 0.0

    @property
    def pixels(self): return self.w * self.h

    @property
    def res(self): return f"{self.w}x{self.h}"


@dataclass
class Chunk:
    start: int;
    end: int;
    loud: bool

    def time(self, fps): return self.start / fps, self.end / fps


@dataclass
class Result:
    inp: str;
    out: str = "";
    ok: bool = False;
    time: float = 0.0;
    err: str = ""


# ==================== 工具函数 ====================
def run_cmd(cmd, log_cb=None):
    try:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8",
                             errors="replace")
        for ln in p.stdout:
            if log_cb: log_cb(ln.rstrip())
        p.wait()
        return p.returncode
    except:
        return -1


def get_dur(path):
    try:
        r = subprocess.run(f'ffprobe -v quiet -print_format json -show_format "{path}"', shell=True,
                           capture_output=True, encoding="utf-8")
        return float(json.loads(r.stdout).get("format", {}).get("duration", 0))
    except:
        return 0.0


def atempo_filter(spd):
    if abs(spd - 1.0) < 0.001: return "anull"
    f = [];
    s = spd
    while s > ProcCfg.ATEMPO_MAX: f.append(f"atempo={ProcCfg.ATEMPO_MAX}"); s /= ProcCfg.ATEMPO_MAX
    while s < ProcCfg.ATEMPO_MIN: f.append(f"atempo={ProcCfg.ATEMPO_MIN}"); s /= ProcCfg.ATEMPO_MIN
    if abs(s - 1.0) > 0.001: f.append(f"atempo={s}")
    return ",".join(f) if f else "anull"


def get_downloads_folder():
    return str(Path.home() / "Downloads")


# ==================== 编码器 ====================
class Encoder:
    @staticmethod
    def nvenc_ok():
        try:
            r = subprocess.run("ffmpeg -hide_banner -f lavfi -i nullsrc=s=256x256:d=1 -c:v h264_nvenc -f null - 2>&1",
                               shell=True, capture_output=True, encoding="utf-8", timeout=10)
            return r.returncode == 0 and not any(x in r.stderr for x in ("Cannot load", "not found", "No NVENC"))
        except:
            return False

    @staticmethod
    def get_params(vi: VideoInfo, gpu=True):
        vbr = BitrateCfg.calc(vi.w, vi.h, vi.vbr, vi.fps)
        mx = int(vbr * 1.2);
        buf = mx * 2
        audio = f"-c:a {AudioCfg.CODEC} -b:a {AudioCfg.BITRATE}"
        if gpu:
            return f"-c:v h264_nvenc -preset p5 -rc vbr -b:v {vbr}k -maxrate {mx}k -bufsize {buf}k {audio}", "NVIDIA NVENC", f"{vbr}k", audio
        crf = VideoCfg.CRF_HI if vi.pixels >= 1920 * 1080 else VideoCfg.CRF_LO
        return f"-c:v libx264 -preset medium -crf {crf} {audio}", "CPU libx264", f"CRF {crf}", audio


# ==================== 静音检测 ====================
class NumpyDetector:
    def detect(self, path, fps, sr, th, margin, log_cb=None):
        if log_cb: log_cb("  📊 检测: NumPy (快速)")
        _, data = wavfile.read(path)
        cnt = data.shape[0]
        mx = max(float(np.max(data)), -float(np.min(data)))
        mx = mx if mx > 0 else 1.0
        spf = sr / fps
        fcnt = int(math.ceil(cnt / spf))
        loud = np.zeros(fcnt, dtype=np.int8)
        for i in range(fcnt):
            s, e = int(i * spf), min(int((i + 1) * spf), cnt)
            if e > s:
                v = max(float(np.max(data[s:e])), -float(np.min(data[s:e])))
                if v / mx >= th: loud[i] = 1
        inc = np.zeros(fcnt, dtype=np.int8)
        for i in range(fcnt):
            inc[i] = np.max(loud[max(0, i - margin):min(fcnt, i + 1 + margin)])
        chunks = [];
        st = 0;
        cur = bool(inc[0])
        for i in range(1, fcnt):
            if bool(inc[i]) != cur: chunks.append(Chunk(st, i, cur)); st = i; cur = bool(inc[i])
        chunks.append(Chunk(st, fcnt, cur))
        return chunks


class PydubDetector:
    def __init__(self, min_len=300):
        self.min_len = min_len

    def detect(self, path, fps, sr, th, margin, log_cb=None):
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent
        if log_cb: log_cb("  📊 检测: Pydub (精确)")
        audio = AudioSegment.from_wav(path)
        dur = len(audio)
        db = audio.dBFS + (20 * math.log10(th + 0.0001))
        if log_cb: log_cb(f"  📊 阈值: {db:.2f} dB")
        ranges = detect_nonsilent(audio, min_silence_len=self.min_len, silence_thresh=db, seek_step=int(1000 / fps))
        msf = 1000 / fps
        total = int(math.ceil(dur / msf))
        chunks = [];
        last = 0
        for s, e in ranges:
            sf, ef = max(0, int(s / msf) - margin), min(total, int(math.ceil(e / msf)) + margin)
            if sf > last: chunks.append(Chunk(last, sf, False))
            if chunks and chunks[-1].loud and sf <= chunks[-1].end:
                chunks[-1] = Chunk(chunks[-1].start, ef, True)
            else:
                chunks.append(Chunk(sf, ef, True))
            last = ef
        if last < total: chunks.append(Chunk(last, total, False))
        return chunks if chunks else [Chunk(0, total, True)]


# ==================== FFmpeg 处理 ====================
class FFmpegProc:
    def __init__(self, log_cb=None):
        self.log = log_cb or (lambda x: None)

    def process(self, inp, out, chunks, sil_spd, fps, tmp, enc_params, glob_spd):
        self.log(f"\n{'=' * 55}\n步骤1: 处理片段\n{'=' * 55}")
        segs = self._chunks(inp, chunks, sil_spd, fps, tmp)
        if not segs: self.log("  ❌ 无有效片段"); return False
        merged = self._merge(segs, tmp)
        if not merged: return False
        return self._final(merged, out, glob_spd, enc_params)

    def _chunks(self, inp, chunks, sil_spd, fps, tmp):
        segs = [];
        skip = 0
        for i, c in enumerate(chunks):
            s, e = c.time(fps);
            dur = e - s
            if dur < ProcCfg.MIN_DUR: continue
            spd = 1.0 if c.loud else sil_spd
            if spd >= 100: skip += 1; continue
            f = os.path.join(tmp, f"seg_{len(segs):06d}.mp4")
            if self._seg(inp, f, s, dur, spd): segs.append(f)
            if (i + 1) % ProcCfg.LOG_INTERVAL == 0:
                self.log(f"  进度: {i + 1}/{len(chunks)}, 生成: {len(segs)}, 跳过: {skip}")
        self.log(f"  ✅ 生成 {len(segs)} 片段, 跳过 {skip} 静音")
        return segs

    def _seg(self, inp, out, start, dur, spd):
        crf, pre = VideoCfg.CRF_SEG, VideoCfg.PRESET_SEG
        if abs(spd - 1.0) > 0.01:
            cmd = f'ffmpeg -hide_banner -y -i "{inp}" -ss {start} -t {dur} -vf "setpts={1 / spd}*PTS" -af "{atempo_filter(spd)}" -c:v libx264 -preset {pre} -crf {crf} -c:a {AudioCfg.CODEC} "{out}"'
        else:
            cmd = f'ffmpeg -hide_banner -y -i "{inp}" -ss {start} -t {dur} -c:v libx264 -preset {pre} -crf {crf} -c:a {AudioCfg.CODEC} "{out}"'
        return run_cmd(cmd) == 0 and os.path.exists(out)

    def _merge(self, segs, tmp):
        lst = os.path.join(tmp, "concat.txt")
        with open(lst, "w", encoding="utf-8") as f:
            for s in segs: f.write(f"file '{os.path.basename(s)}'\n")
        out = os.path.join(tmp, "merged.mp4")
        self.log("\n  合并片段...")
        if run_cmd(f'ffmpeg -hide_banner -y -f concat -safe 0 -i "{lst}" -c copy "{out}"') != 0:
            self.log("  ❌ 合并失败");
            return None
        self.log(f"  ✅ 合并完成 {get_dur(out):.2f}s")
        return out

    def _final(self, inp, out, spd, params):
        dur = get_dur(inp)
        self.log(f"\n{'=' * 55}\n步骤2: 全局加速 {spd}x + 编码\n{'=' * 55}")
        if abs(spd - 1.0) > 0.01:
            self.log(f"  PTS: {1 / spd:.4f}, 音频: {atempo_filter(spd)}")
            cmd = f'ffmpeg -hide_banner -y -i "{inp}" -vf "setpts={1 / spd}*PTS" -af "{atempo_filter(spd)}" {params} "{out}"'
            exp = dur / spd
        else:
            self.log("  无需加速")
            cmd = f'ffmpeg -hide_banner -y -i "{inp}" {params} "{out}"'
            exp = dur
        self.log(f"  预期: {exp:.2f}s\n  编码中...")
        if run_cmd(cmd, self.log) != 0: self.log("  ❌ 失败"); return False
        self.log(f"  ✅ 完成! 预期: {exp:.2f}s, 实际: {get_dur(out):.2f}s")
        return True


# ==================== 主处理类 ====================
class SilenceCut:
    def __init__(self, inp, out=None, sil_spd=999, glob_spd=1.1, margin=3, th=0.025, pydub=True, gpu=True, min_sil=300,
                 log_cb=None):
        self.inp = inp
        self.out = out or self._gen_out(inp)
        self.sil_spd = sil_spd;
        self.glob_spd = glob_spd;
        self.margin = margin
        self.th = th;
        self.pydub = pydub;
        self.gpu = gpu;
        self.min_sil = min_sil
        self.log = log_cb or (lambda x: None)
        self._tmp = None;
        self._vi = None;
        self._gpu_ok = False

    @staticmethod
    def _gen_out(p):
        b, e = os.path.splitext(p)
        return f"{b}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{e}"

    def _probe(self):
        cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams "{self.inp}"'
        r = subprocess.run(cmd, shell=True, capture_output=True, encoding="utf-8")
        vi = VideoInfo()
        try:
            d = json.loads(r.stdout)
            for s in d.get("streams", []):
                if s.get("codec_type") == "video":
                    fps = s.get("r_frame_rate", "25/1")
                    if "/" in fps:
                        n, dn = fps.split("/")
                        vi.fps = float(n) / float(dn) if float(dn) else 25
                    vi.w = s.get("width", VideoCfg.W)
                    vi.h = s.get("height", VideoCfg.H)
                    if "bit_rate" in s: vi.vbr = int(s["bit_rate"]) // 1000
                elif s.get("codec_type") == "audio":
                    vi.sr = int(s.get("sample_rate", AudioCfg.RATE))
            fmt = d.get("format", {})
            if vi.vbr == 0 and "bit_rate" in fmt: vi.vbr = int(fmt["bit_rate"]) // 1000
            vi.dur = float(fmt.get("duration", 0))
        except:
            pass
        return vi

    def run(self):
        self.log("=" * 60)
        self.log("🎬 SilenceCut 自动静音剪辑")
        self.log("=" * 60)
        self.log(f"📁 输入: {self.inp}")
        self.log(f"📁 输出: {self.out}")
        self.log(f"⏩ 静音速度: {self.sil_spd} (≥100删除)")
        self.log(f"🚀 全局加速: {self.glob_spd}x")
        self.log("=" * 60)

        self._tmp = os.path.splitext(self.inp)[0] + "_TEMP"
        if os.path.exists(self._tmp): rmtree(self._tmp)
        os.makedirs(self._tmp)

        try:
            self.log("\n[1/4] 分析视频...")
            self._vi = self._probe()
            self.log(f"  分辨率: {self._vi.res}, 帧率: {self._vi.fps:.2f}fps, 时长: {self._vi.dur:.2f}s")

            self.log("\n[2/4] 检测编码器...")
            if self.gpu:
                self._gpu_ok = Encoder.nvenc_ok()
                self.log(f"  GPU: {'可用 ✅' if self._gpu_ok else '不可用 ❌ (用CPU)'}")

            self.log("\n[3/4] 检测静音...")
            wav = os.path.join(self._tmp, "audio.wav")
            run_cmd(
                f'ffmpeg -hide_banner -y -i "{self.inp}" -vn -acodec pcm_s16le -ar {self._vi.sr} -ac {AudioCfg.CHANNELS} "{wav}"')
            det = PydubDetector(self.min_sil) if self.pydub else NumpyDetector()
            chunks = det.detect(wav, self._vi.fps, self._vi.sr, self.th, self.margin, self.log)
            loud = sum(1 for c in chunks if c.loud)
            self.log(f"  片段: {len(chunks)} (有声: {loud}, 静音: {len(chunks) - loud})")

            self.log("\n[4/4] 处理视频...")
            params, enc_type, br, audio = Encoder.get_params(self._vi, self.gpu and self._gpu_ok)
            self.log(I18n.get("enc_title"))
            self.log(f"{I18n.get('enc_type')} {enc_type}")
            self.log(f"{I18n.get('enc_bitrate')} {br}")
            self.log(f"{I18n.get('enc_audio')} {audio}")
            self.log(f"{I18n.get('enc_params')} {params}")
            self.log("=" * 60)

            ok = FFmpegProc(self.log).process(self.inp, self.out, chunks, self.sil_spd, self._vi.fps, self._tmp, params,
                                              self.glob_spd)
            if self._tmp and os.path.exists(self._tmp): rmtree(self._tmp)

            if ok:
                self.log(f"\n{'=' * 60}\n✅ 完成!\n📁 {self.out}\n{'=' * 60}")
                return self.out
            return ""
        except Exception as e:
            if self._tmp and os.path.exists(self._tmp): rmtree(self._tmp, ignore_errors=True)
            self.log(f"❌ 异常: {e}")
            raise


# ==================== 处理线程 ====================
class ProcThread(QThread):
    log_sig = pyqtSignal(str)
    prog_sig = pyqtSignal(int, int, str)
    done_sig = pyqtSignal(list)

    def __init__(self, files, cfg, out_dir, use_src, parallel, workers):
        super().__init__()
        self.files = files;
        self.cfg = cfg;
        self.out_dir = out_dir
        self.use_src = use_src;
        self.parallel = parallel;
        self.workers = workers
        self._stop = False

    def stop(self):
        self._stop = True

    def log(self, m):
        self.log_sig.emit(m)

    def _out_path(self, inp):
        n, e = os.path.splitext(os.path.basename(inp))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(os.path.dirname(inp) if self.use_src else self.out_dir, f"{n}_{ts}{e}")

    def _proc1(self, f):
        r = Result(inp=f)
        if not os.path.exists(f): r.err = "文件不存在"; return r
        t = time.time()
        try:
            out = self._out_path(f)
            o = SilenceCut(f, out, self.cfg.get("sil_spd", 999), self.cfg.get("glob_spd", 1.1),
                           self.cfg.get("margin", 3), self.cfg.get("th", 0.025),
                           self.cfg.get("pydub", True), self.cfg.get("gpu", True),
                           self.cfg.get("min_sil", 300), self.log).run()
            r.out = o;
            r.ok = bool(o)
        except Exception as e:
            r.err = str(e)
        r.time = time.time() - t
        return r

    def run(self):
        res = [];
        n = len(self.files)
        if self.parallel and n > 1:
            self.log(f"\n{'=' * 60}\n" + I18n.get("batch_parallel").format(n=n, w=self.workers) + f"\n{'=' * 60}")
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                fut = {ex.submit(self._proc1, f): f for f in self.files}
                done = 0
                for f in as_completed(fut):
                    if self._stop: break
                    done += 1;
                    r = f.result();
                    res.append(r)
                    self.prog_sig.emit(done, n, os.path.basename(r.inp))
        else:
            self.log(f"\n{'=' * 60}\n" + I18n.get("batch_serial").format(n=n) + f"\n{'=' * 60}")
            for i, f in enumerate(self.files):
                if self._stop: break
                self.prog_sig.emit(i + 1, n, os.path.basename(f))
                res.append(self._proc1(f))
        self.done_sig.emit(res)


# ==================== 文件列表 ====================
class FileList(QListWidget):
    EXT = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setMinimumHeight(100)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                p = u.toLocalFile()
                if p.lower().endswith(self.EXT): self.add(p)
            e.acceptProposedAction()

    def add(self, p):
        for i in range(self.count()):
            if self.item(i).data(Qt.ItemDataRole.UserRole) == p: return
        it = QListWidgetItem(os.path.basename(p))
        it.setData(Qt.ItemDataRole.UserRole, p)
        it.setToolTip(p)
        self.addItem(it)

    def files(self):
        return [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]

    def rem_sel(self):
        for it in self.selectedItems(): self.takeItem(self.row(it))


# ==================== 主窗口 ====================
class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread = None
        self._init_ui()
        self._init_menu()
        self._load()
        self._texts()
        for icon_name in ["icon6.ico", "icon2.ico", "icon3.ico", "icon6.png", "icon2.png", "icon3.png"]:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), icon_name)
            if os.path.exists(icon_path):
                from PyQt6.QtGui import QIcon
                self.setWindowIcon(QIcon(icon_path))
                break
        self.setStyleSheet(STYLE_SHEET)

    def _init_menu(self):
        mb = self.menuBar()
        self.m_set = mb.addMenu(I18n.get("menu_settings"))
        self.m_lang = QMenu(I18n.get("menu_language"), self)
        self.a_zh = QAction("中文", self, checkable=True)
        self.a_en = QAction("English", self, checkable=True)
        self.a_zh.triggered.connect(lambda: self._lang("zh"))
        self.a_en.triggered.connect(lambda: self._lang("en"))
        self.m_lang.addAction(self.a_zh)
        self.m_lang.addAction(self.a_en)
        self.m_set.addMenu(self.m_lang)
        self._menu_chk()

    def _menu_chk(self):
        self.a_zh.setChecked(I18n.current_lang == "zh")
        self.a_en.setChecked(I18n.current_lang == "en")

    def _lang(self, l):
        I18n.set_language(l);
        self._texts();
        self._menu_chk();
        self._save()

    def _init_ui(self):
        self.setWindowTitle(I18n.get("window_title"))
        self.setMinimumSize(750, 650)
        cw = QWidget();
        self.setCentralWidget(cw)
        ml = QVBoxLayout(cw);
        ml.setSpacing(10);
        ml.setContentsMargins(12, 12, 12, 12)

        # 文件列表
        self.g_file = QGroupBox()
        fl = QVBoxLayout(self.g_file)
        self.flist = FileList();
        fl.addWidget(self.flist)
        fb = QHBoxLayout()
        self.b_add = QPushButton();
        self.b_add.clicked.connect(self._add_files)
        self.b_rem = QPushButton();
        self.b_rem.clicked.connect(self.flist.rem_sel)
        self.b_clr = QPushButton();
        self.b_clr.clicked.connect(self.flist.clear)
        fb.addWidget(self.b_add);
        fb.addWidget(self.b_rem);
        fb.addWidget(self.b_clr);
        fb.addStretch()
        fl.addLayout(fb);
        ml.addWidget(self.g_file)

        # 参数区域
        pl = QHBoxLayout()
        self.g_par = QGroupBox();
        prl = QVBoxLayout(self.g_par)
        r1 = QHBoxLayout()
        self.l_sil = QLabel();
        r1.addWidget(self.l_sil)
        self.sp_sil = QDoubleSpinBox();
        self.sp_sil.setRange(0.1, 999);
        self.sp_sil.setValue(999);
        r1.addWidget(self.sp_sil)
        r1.addSpacing(20)
        self.l_glob = QLabel();
        r1.addWidget(self.l_glob)
        self.sp_glob = QDoubleSpinBox();
        self.sp_glob.setRange(0.5, 5.0);
        self.sp_glob.setValue(1.1);
        self.sp_glob.setSingleStep(0.1);
        r1.addWidget(self.sp_glob)
        r1.addStretch();
        prl.addLayout(r1)

        r2 = QHBoxLayout()
        self.l_mar = QLabel();
        r2.addWidget(self.l_mar)
        self.sp_mar = QSpinBox();
        self.sp_mar.setRange(0, 20);
        self.sp_mar.setValue(3);
        r2.addWidget(self.sp_mar)
        r2.addSpacing(20)
        self.l_th = QLabel();
        r2.addWidget(self.l_th)
        self.sp_th = QDoubleSpinBox();
        self.sp_th.setRange(0.001, 1.0);
        self.sp_th.setValue(0.025);
        self.sp_th.setSingleStep(0.005);
        self.sp_th.setDecimals(3);
        r2.addWidget(self.sp_th)
        r2.addSpacing(20)
        self.l_min = QLabel();
        r2.addWidget(self.l_min)
        self.sp_min = QSpinBox();
        self.sp_min.setRange(50, 2000);
        self.sp_min.setValue(300);
        self.sp_min.setSingleStep(50);
        r2.addWidget(self.sp_min)
        r2.addStretch();
        prl.addLayout(r2)
        pl.addWidget(self.g_par)

        # 编码设置
        self.g_enc = QGroupBox();
        el = QVBoxLayout(self.g_enc)
        er = QHBoxLayout()
        self.l_enc = QLabel();
        er.addWidget(self.l_enc)
        self.r_gpu = QRadioButton("GPU (NVENC)");
        self.r_cpu = QRadioButton("CPU (libx264)")
        self.r_gpu.setChecked(True)
        self.bg_enc = QButtonGroup();
        self.bg_enc.addButton(self.r_gpu);
        self.bg_enc.addButton(self.r_cpu)
        er.addWidget(self.r_gpu);
        er.addWidget(self.r_cpu);
        er.addStretch();
        el.addLayout(er)

        dr = QHBoxLayout()
        self.l_det = QLabel();
        dr.addWidget(self.l_det)
        self.r_pydub = QRadioButton("Pydub (精确)");
        self.r_numpy = QRadioButton("NumPy (快速)")
        self.r_pydub.setChecked(True)
        self.bg_det = QButtonGroup();
        self.bg_det.addButton(self.r_pydub);
        self.bg_det.addButton(self.r_numpy)
        dr.addWidget(self.r_pydub);
        dr.addWidget(self.r_numpy);
        dr.addStretch();
        el.addLayout(dr)

        mr = QHBoxLayout()
        self.l_mode = QLabel();
        mr.addWidget(self.l_mode)
        self.r_ser = QRadioButton();
        self.r_par = QRadioButton();
        self.r_par.setChecked(True)
        self.bg_mode = QButtonGroup();
        self.bg_mode.addButton(self.r_ser);
        self.bg_mode.addButton(self.r_par)
        mr.addWidget(self.r_ser);
        mr.addWidget(self.r_par)
        self.l_wrk = QLabel();
        mr.addWidget(self.l_wrk)
        self.sp_wrk = QSpinBox();
        self.sp_wrk.setRange(1, 8);
        self.sp_wrk.setValue(2);
        mr.addWidget(self.sp_wrk)
        mr.addStretch();
        el.addLayout(mr)
        pl.addWidget(self.g_enc);
        ml.addLayout(pl)

        # 输出设置 - 只保留一个复选框
        self.g_out = QGroupBox();
        ol = QVBoxLayout(self.g_out)
        self.chk_src = QCheckBox();
        self.chk_src.setChecked(True);
        self.chk_src.stateChanged.connect(self._out_mode);
        ol.addWidget(self.chk_src)
        self.w_cust = QWidget();
        cl = QHBoxLayout(self.w_cust);
        cl.setContentsMargins(0, 0, 0, 0)
        self.l_dir = QLabel();
        cl.addWidget(self.l_dir)
        self.e_dir = QLineEdit();
        self.e_dir.setText(get_downloads_folder());
        cl.addWidget(self.e_dir)
        self.b_brw = QPushButton();
        self.b_brw.clicked.connect(self._browse);
        cl.addWidget(self.b_brw)
        ol.addWidget(self.w_cust);
        self.w_cust.hide()
        self.l_hint = QLabel();
        self.l_hint.setObjectName("hint");
        self.l_hint.setWordWrap(True);
        ol.addWidget(self.l_hint)
        ml.addWidget(self.g_out)

        # 进度
        self.g_prog = QGroupBox();
        pgl = QVBoxLayout(self.g_prog)
        pgr = QHBoxLayout()
        self.prog = QProgressBar();
        self.prog.setTextVisible(True);
        self.prog.setFormat("%v / %m - %p%");
        pgr.addWidget(self.prog)
        self.l_cur = QLabel();
        pgr.addWidget(self.l_cur);
        pgl.addLayout(pgr)
        self.log = QTextEdit();
        self.log.setReadOnly(True);
        self.log.setMinimumHeight(150);
        pgl.addWidget(self.log)
        ml.addWidget(self.g_prog)

        # 按钮
        bl = QHBoxLayout()
        self.b_start = QPushButton();
        self.b_start.setStyleSheet("font-size:14px;font-weight:bold;padding:10px 30px");
        self.b_start.clicked.connect(self._start)
        self.b_stop = QPushButton();
        self.b_stop.setEnabled(False);
        self.b_stop.clicked.connect(self._stop)
        self.b_open = QPushButton();
        self.b_open.clicked.connect(self._open_dir)
        bl.addStretch();
        bl.addWidget(self.b_start);
        bl.addWidget(self.b_stop);
        bl.addWidget(self.b_open);
        bl.addStretch()
        ml.addLayout(bl)

    def _texts(self):
        self.setWindowTitle(I18n.get("window_title"))
        self.m_set.setTitle(I18n.get("menu_settings"));
        self.m_lang.setTitle(I18n.get("menu_language"))
        self.g_file.setTitle(I18n.get("file_list_title"))
        self.b_add.setText(I18n.get("btn_add"));
        self.b_rem.setText(I18n.get("btn_remove"));
        self.b_clr.setText(I18n.get("btn_clear"))
        self.g_par.setTitle(I18n.get("params_title"))
        self.l_sil.setText(I18n.get("silent_speed"));
        self.sp_sil.setToolTip(I18n.get("silent_speed_tip"))
        self.l_glob.setText(I18n.get("global_speed"));
        self.sp_glob.setToolTip(I18n.get("global_speed_tip"))
        self.l_mar.setText(I18n.get("frame_margin"));
        self.l_th.setText(I18n.get("threshold"));
        self.l_min.setText(I18n.get("min_silence"))
        self.g_enc.setTitle(I18n.get("encode_title"));
        self.l_enc.setText(I18n.get("encoder"));
        self.l_det.setText(I18n.get("detect"))
        self.l_mode.setText(I18n.get("mode"));
        self.r_ser.setText(I18n.get("serial"));
        self.r_par.setText(I18n.get("parallel"));
        self.l_wrk.setText(I18n.get("workers"))
        self.g_out.setTitle(I18n.get("output_title"))
        self.chk_src.setText(I18n.get("use_source"))
        self.l_dir.setText(I18n.get("output_dir"));
        self.b_brw.setText(I18n.get("browse"));
        self._hint()
        self.g_prog.setTitle(I18n.get("progress_title"));
        self.l_cur.setText(I18n.get("idle"))
        self.b_start.setText(I18n.get("btn_start"));
        self.b_stop.setText(I18n.get("btn_stop"));
        self.b_open.setText(I18n.get("btn_open"))

    def _out_mode(self, s):
        use = (s == Qt.CheckState.Checked.value)
        self.w_cust.setVisible(not use)
        self._hint()

    def _hint(self):
        self.l_hint.setText(I18n.get("hint_source") if self.chk_src.isChecked() else I18n.get("hint_custom"))

    def _add_files(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "选择视频" if I18n.current_lang == "zh" else "Select", "",
                                             "视频 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All (*)")
        for f in fs: self.flist.add(f)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "选择目录" if I18n.current_lang == "zh" else "Select")
        if d: self.e_dir.setText(d)

    def _open_dir(self):
        if self.chk_src.isChecked():
            fs = self.flist.files()
            if fs:
                d = os.path.dirname(fs[0])
            else:
                QMessageBox.warning(self, I18n.get("tip"), I18n.get("no_files"))
                return
        else:
            d = self.e_dir.text()
            if not d:
                d = os.path.join(str(Path.home()), "Videos", "SilenceCut_Output")
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        if platform.system() == "Windows":
            subprocess.run(["explorer", os.path.normpath(d)])
        elif platform.system() == "Darwin":
            subprocess.run(["open", d])
        else:
            subprocess.run(["xdg-open", d])

    def _cfg(self):
        return {"sil_spd": self.sp_sil.value(), "glob_spd": self.sp_glob.value(), "margin": self.sp_mar.value(),
                "th": self.sp_th.value(), "min_sil": self.sp_min.value(), "gpu": self.r_gpu.isChecked(),
                "pydub": self.r_pydub.isChecked()}

    def _start(self):
        fs = self.flist.files()
        if not fs: QMessageBox.warning(self, I18n.get("tip"), I18n.get("no_files")); return
        if not self.chk_src.isChecked():
            d = self.e_dir.text()
            if not d: QMessageBox.warning(self, I18n.get("tip"), I18n.get("no_dir")); return
            os.makedirs(d, exist_ok=True)
        else:
            d = ""
        self.log.clear();
        self.prog.setMaximum(len(fs));
        self.prog.setValue(0)
        self.b_start.setEnabled(False);
        self.b_stop.setEnabled(True)
        self.thread = ProcThread(fs, self._cfg(), d, self.chk_src.isChecked(), self.r_par.isChecked(),
                                 self.sp_wrk.value())
        self.thread.log_sig.connect(self._log);
        self.thread.prog_sig.connect(self._prog);
        self.thread.done_sig.connect(self._done)
        self.thread.start()

    def _stop(self):
        if self.thread: self.thread.stop(); self._log(f"\n{I18n.get('stopped')}")

    def _log(self, m):
        self.log.append(m); self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _prog(self, c, t, f):
        self.prog.setValue(c); self.l_cur.setText(f"{I18n.get('processing')} {f}")

    def _done(self, res):
        self.b_start.setEnabled(True);
        self.b_stop.setEnabled(False);
        self.l_cur.setText(I18n.get("complete"))
        ok = sum(1 for r in res if r.ok);
        n = len(res);
        t = sum(r.time for r in res)
        self._log(f"\n{'=' * 60}\n📊 {I18n.get('summary')}\n{'=' * 60}")
        self._log(f"{I18n.get('success')} {ok}/{n}");
        self._log(f"{I18n.get('failed')} {n - ok}/{n}")
        self._log(f"{I18n.get('total_time')} {t:.1f}s ({t / 60:.1f}min)")
        if n - ok > 0:
            self._log(f"\n{I18n.get('failed_list')}")
            for r in res:
                if not r.ok: self._log(f"  - {os.path.basename(r.inp)}: {r.err}")
        QMessageBox.information(self, I18n.get("done_title"), I18n.get("done_text").format(s=ok, t=n, e=t))

    def _save(self):
        s = {"lang": I18n.current_lang, "use_src": self.chk_src.isChecked(), "out_dir": self.e_dir.text(),
             **self._cfg(), "parallel": self.r_par.isChecked(), "workers": self.sp_wrk.value()}
        try:
            with open(Path.home() / ".silencecut.json", "w", encoding="utf-8") as f:
                json.dump(s, f)
        except:
            pass

    def _load(self):
        p = Path.home() / ".silencecut.json"
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    s = json.load(f)
                I18n.set_language(s.get("lang", "zh"))
                self.chk_src.setChecked(s.get("use_src", True))
                self.e_dir.setText(s.get("out_dir", get_downloads_folder()))
                self._out_mode(Qt.CheckState.Checked.value if s.get("use_src", True) else Qt.CheckState.Unchecked.value)
                self.sp_sil.setValue(s.get("sil_spd", 999));
                self.sp_glob.setValue(s.get("glob_spd", 1.1))
                self.sp_mar.setValue(s.get("margin", 3));
                self.sp_th.setValue(s.get("th", 0.025));
                self.sp_min.setValue(s.get("min_sil", 300))
                self.r_gpu.setChecked(s.get("gpu", True));
                self.r_cpu.setChecked(not s.get("gpu", True))
                self.r_pydub.setChecked(s.get("pydub", True));
                self.r_numpy.setChecked(not s.get("pydub", True))
                self.r_par.setChecked(s.get("parallel", True));
                self.r_ser.setChecked(not s.get("parallel", True))
                self.sp_wrk.setValue(s.get("workers", 2))
            except:
                pass

    def closeEvent(self, e):
        self._save()
        if self.thread and self.thread.isRunning():
            if QMessageBox.question(self, I18n.get("exit_title"), I18n.get("confirm_exit"),
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.No:
                e.ignore();
                return
            self.thread.stop();
            self.thread.wait(3000)
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei", 9))

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#222222"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f5f5f5"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#222222"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#222222"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#222222"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#0078d4"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    w = MainWin();
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()