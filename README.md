# When AI Sees Litter

A Streamlit app that uses a YOLO model to detect three waste types: **Clear plastic bottle**, **Drink can**, **Styrofoam piece**. The app supports **Upload image** and **Camera**. Currently, it shows city-specific disposal guidance for **Shibuya** and SDG tiles for context.

## Features
- Two input modes: Upload image and Camera
- Loads weights from a public URL or a local file
- Adjustable thresholds: base confidence, IoU, per-class minimums, minimum box area, inference image size, optional TTA
- Disposal guidance cards for Shibuya with official links and images
- SDG tiles: 11, 12, 13
- Live video was removed

## Requirements
Use Python 3.10 or newer. Example `requirements.txt`:
