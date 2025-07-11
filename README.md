# Landmarks Video Generator

This project automatically generates short narrated videos of famous landmarks using state-of-the-art AI models for image generation (Stable Diffusion), text-to-speech (TTS), and video editing (MoviePy).

## Features

- **AI Image Generation:** Creates ultra-realistic images of landmarks using Stable Diffusion.
- **Text-to-Speech Narration:** Generates English audio descriptions for each landmark.
- **Automated Video Creation:** Combines images and narration into three themed videos.
- **Efficient Processing:** Skips already generated files for faster reruns.

## Landmarks Covered

- Hagia Sophia (Ayasofya)
- Sultanahmet Mosque
- Bosphorus Bridge
- Cappadocia
- Pamukkale
- Antalya
- Ephesus
- Mount Nemrut
- Rize Tea Plantations

## Requirements

- Python 3.8+
- [diffusers](https://github.com/huggingface/diffusers)
- [TTS](https://github.com/coqui-ai/TTS)
- [moviepy](https://zulko.github.io/moviepy/)
- torch

Install dependencies:

```bash
pip install torch diffusers TTS moviepy
```

## Usage

1. **Run the script:**
   ```bash
   python landmarks_video_generator.py
   ```
2. The script will generate images, audio, and three videos (`video1.mp4`, `video2.mp4`, `video3.mp4`).

## Notes

- The script is optimized for CPU but will use GPU if available.
- Images and audio are only generated if not already present.
- Each video consists of three landmark segments.

## Example Videos

You can view the generated videos by clicking the links below:

- [Video 1](video1.mp4)
- [Video 2](video2.mp4)
- [Video 3](video3.mp4)
