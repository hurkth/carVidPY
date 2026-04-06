# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a real-time face detection application for Raspberry Pi 4B using the CSI camera. It uses OpenCV's Haar Cascade classifier for efficient face detection and displays results with FPS metrics.

**Platform requirements:** Raspberry Pi 4B with CSI camera module (not portable to other systems).

## Running the Application

```bash
python3 face_detection.py
```

**Exit:** Press 'q' on the display window or Ctrl+C to shut down gracefully.

## Dependencies

Required packages (install on RPi):
```bash
sudo apt update
sudo apt install -y python3-opencv python3-picamera2 python3-numpy
# or via pip if needed:
pip3 install opencv-python picamera2 numpy
```

## Architecture

### Single-File Design

The application is implemented as a single Python file with one main class:

**`FaceDetectionApp`** — Handles all face detection logic:
- **Initialization:** Sets up the CSI camera with picamera2, loads the Haar Cascade classifier, configures resolution/FPS
- **Frame capture loop:** Continuously captures frames, detects faces, draws overlays, and handles display
- **Face detection:** Uses OpenCV's `detectMultiScale()` with tuned parameters (scaleFactor=1.1, minNeighbors=4) for RPi performance balance
- **Visualization:** Draws rectangles around detected faces and displays FPS/face count overlay
- **Cleanup:** Properly releases camera and window resources on shutdown

### Key Parameters

Camera configuration in `__init__`:
- **Resolution:** 640x480 (optimized for RPi performance; can be adjusted via constructor args)
- **FPS target:** 30 frames per second
- **Haar Cascade:** OpenCV's built-in `haarcascade_frontalface_default.xml`

Face detection tuning in `detect_faces()`:
- `scaleFactor=1.1` — Step size through image scales; lower = more accuracy but slower
- `minNeighbors=4` — Neighbors required for detection; lower = faster with more false positives
- `minSize=(30, 30)` & `maxSize=(300, 300)` — Constrains face size range

### Threading Import

The `Thread` import is included but not currently used. Potential future use case: offload face detection to a background thread to decouple capture/display.

## Development Notes

- **Camera warmup:** 2-second delay after initialization to stabilize image quality
- **Color space handling:** Converts between RGB (picamera2), RGB (OpenCV operations), and BGR (cv2.imshow) as needed
- **Error handling:** Exits immediately if Haar Cascade fails to load; uses try/finally to ensure resource cleanup on any shutdown path
- **FPS calculation:** Running average since start, not frame-by-frame (stable across the session)
