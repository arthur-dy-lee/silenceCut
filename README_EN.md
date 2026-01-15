# SilenceCut v2.1 - Auto Silence Remover

English | [中文](README.md)

<p align="center">
  <img src="icon1.png" width="128" height="128" alt="SilenceCut Logo">
</p>

<p align="center">
  <b>Automatically detect and remove silent parts from videos, making your content more compact!</b>
</p>

## ✨ Features

- 🎬 **Smart Silence Detection** - Automatically identify silent segments in videos
- ✂️ **Flexible Processing** - Delete or speed up silent parts
- ⚡ **GPU Acceleration** - Support NVIDIA NVENC encoding for faster processing
- 📁 **Batch Processing** - Process multiple files in serial or parallel mode
- 🖱️ **Drag & Drop** - Simply drag video files into the window
- 🌍 **Multi-language** - Support Chinese and English interface
- 💾 **Settings Memory** - Automatically save user preferences
- 🎛️ **Visual Configuration** - All parameters adjustable in the UI

## 📸 Screenshot

![SilenceCut 界面](/png/GUI_en.png)

## 🔧 Requirements

- **OS**: Windows 10/11
- **Python**: 3.8+ (if running from source)
- **FFmpeg**: Must be installed and added to system PATH
- **GPU**: NVIDIA GPU (optional, for hardware encoding)

## 📦 Installation

### Option 1: Download Executable (Recommended)

1. Go to [Releases](https://github.com/arthur-dy-lee/silenceCut/releases) page
2. Download the latest `SilenceCut.exe`
3. Double-click to run

### Option 2: Run from Source

```bash
# 1. Clone repository
git clone https://github.com/your-username/SilenceCut.git
cd SilenceCut

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the program
python silence_cut_gui.py
```

### Option 3: Build EXE Yourself

```bash
# 1. Install packaging tool
pip install pyinstaller

# 2. (Optional) Convert icon1.png to icon1.ico and place in same directory

# 3. Run build script
build.bat
```

## 📖 Usage Guide

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Silent Speed** | 999 | Processing speed for silent segments<br>≥100 = completely remove<br>2-4 = speed up playback |
| **Global Speed** | 1.2 | Overall speed multiplier for final video<br>1.0 = original speed<br>1.2 = 20% faster |
| **Frame Margin** | 3 | Buffer frames at silence boundaries, prevents harsh cuts |
| **Threshold** | 0.025 | Volume below this is considered silent (0-1), lower = more sensitive |
| **Min Silence(ms)** | 300 | Minimum silence duration to process |

### Encoding Settings

| Option | Description |
|--------|-------------|
| GPU (NVENC) | Use NVIDIA hardware encoding, faster |
| CPU (libx264) | Use CPU software encoding, better compatibility |
| Pydub (Accurate) | Use Pydub library for detection, higher accuracy |
| NumPy (Fast) | Use NumPy for detection, faster speed |

### Output Settings

- **✅ Output to source directory** (default): Save output alongside source video with timestamp
- **Custom directory**: Uncheck to specify a custom output folder

## 🛠️ Dependencies

```
PyQt6>=6.4.0
numpy>=1.21.0
scipy>=1.7.0
pydub>=0.25.0
```

## ❓ FAQ

**Q: FFmpeg not found?**
> Download FFmpeg and add it to your system PATH. Download: https://ffmpeg.org/download.html

**Q: GPU encoding unavailable?**
> Requires NVIDIA GPU with latest drivers. The program auto-detects and falls back to CPU encoding if unavailable.

**Q: Audio/video out of sync after processing?**
> Try adjusting the "Frame Margin" parameter or switch detection method.

## 📄 License

Apache License

## 🤝 Contributing

Issues and Pull Requests are welcome!
