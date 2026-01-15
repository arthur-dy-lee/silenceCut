# SilenceCut v2.1 - 自动静音剪辑工具

[English](README_EN.md) | 中文

<p align="center">
  <img src="icon1.png" width="128" height="128" alt="SilenceCut Logo">
</p>

<p align="center">
  <b>自动检测并删除视频中的静音片段，让你的视频更紧凑！</b>
</p>

## ✨ 功能特性

- 🎬 **智能静音检测** - 自动识别视频中的静音片段
- ✂️ **灵活处理方式** - 可删除静音或加速播放
- ⚡ **GPU 硬件加速** - 支持 NVIDIA NVENC 编码，速度更快
- 📁 **批量处理** - 支持多文件串行/并行处理
- 🖱️ **拖拽添加** - 直接拖拽视频文件到窗口
- 🌍 **中英文界面** - 支持中文和英文切换
- 💾 **参数记忆** - 自动保存用户设置
- 🎛️ **可视化配置** - 所有参数都可在界面调整

## 📸 界面预览

![SilenceCut 界面](screenshot.png)

## 🔧 环境要求

- **操作系统**: Windows 10/11
- **Python**: 3.8+ (如果从源码运行)
- **FFmpeg**: 必须安装并添加到系统 PATH
- **显卡**: NVIDIA GPU (可选，用于硬件加速编码)

## 📦 安装方法

### 方式一：下载可执行文件 (推荐)

1. 前往 [Releases](https://github.com/你的用户名/SilenceCut/releases) 页面
2. 下载最新版本的 `SilenceCut.exe`
3. 双击运行即可

### 方式二：从源码运行

```bash
# 1. 克隆仓库
git clone https://github.com/你的用户名/SilenceCut.git
cd SilenceCut

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行程序
python silence_cut_gui.py
```

### 方式三：自行打包 EXE

```bash
# 1. 安装打包工具
pip install pyinstaller

# 2. (可选) 将 icon1.png 转换为 icon1.ico 放在同目录

# 3. 运行打包脚本
build.bat
```

## 📖 使用说明

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **静音速度** | 999 | 静音片段的处理速度<br>≥100 = 完全删除静音<br>2-4 = 加速播放静音部分 |
| **视频全局加速** | 1.2 | 最终视频的整体加速倍率<br>1.0 = 原速<br>1.2 = 加速20% |
| **边界缓冲帧** | 3 | 静音/有声边界的缓冲帧数，防止切割生硬 |
| **静音阈值** | 0.025 | 音量低于此值视为静音 (0-1)，越小越敏感 |
| **最小静音(ms)** | 300 | 最短静音时长，低于此值不处理 |

### 编码设置

| 选项 | 说明 |
|------|------|
| GPU (NVENC) | 使用 NVIDIA 显卡硬件编码，速度快 |
| CPU (libx264) | 使用 CPU 软件编码，兼容性好 |
| Pydub (精确) | 使用 Pydub 库检测静音，精度高 |
| NumPy (快速) | 使用 NumPy 检测静音，速度快 |

### 输出设置

- **✅ 输出到源文件目录** (默认): 输出文件保存在源视频同目录，文件名添加时间戳
- **自定义目录**: 取消勾选后，可指定统一的输出目录

## 🛠️ 依赖项

```
PyQt6>=6.4.0
numpy>=1.21.0
scipy>=1.7.0
pydub>=0.25.0
```

## ❓ 常见问题

**Q: 提示找不到 FFmpeg？**
> 请下载 FFmpeg 并添加到系统环境变量 PATH 中。下载地址: https://ffmpeg.org/download.html

**Q: GPU 编码不可用？**
> 需要 NVIDIA 显卡并安装最新驱动。程序会自动检测，不可用时会切换到 CPU 编码。

**Q: 处理后视频音画不同步？**
> 尝试调整"边界缓冲帧"参数，或切换检测方式。

## 📄 开源协议

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
