#!/usr/bin/env python3
"""
Real-time Face Detection with CSI Camera on Raspberry Pi 4B
Uses picamera2 and OpenCV Haar Cascade for efficient face detection.
"""

import cv2
import numpy as np
import sys
from picamera2 import Picamera2
from threading import Thread
import time

class FaceDetectionApp:
    def __init__(self, camera_width=640, camera_height=480, fps=30):
        """
        Initialize the face detection application.

        Args:
            camera_width: Camera resolution width (default 640 for RPi performance)
            camera_height: Camera resolution height (default 480)
            fps: Frames per second (default 30)
        """
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.fps = fps
        self.running = True

        # Load Haar Cascade classifier for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            print("Error: Could not load Haar Cascade classifier")
            sys.exit(1)

        # Initialize picamera2
        print("Initializing CSI camera...")
        self.camera = Picamera2()

        # Configure camera
        config = self.camera.create_preview_configuration(
            main={"format": "RGB888", "size": (camera_width, camera_height)}
        )
        self.camera.configure(config)
        self.camera.start()

        print(f"Camera initialized: {camera_width}x{camera_height} @ {fps} FPS")
        time.sleep(2)  # Allow camera to warm up

    def detect_faces(self, frame):
        """
        Detect faces in the given frame using Haar Cascade.

        Args:
            frame: Input image frame (BGR format)

        Returns:
            List of face rectangles (x, y, w, h)
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detect faces with adjusted parameters for speed/accuracy
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # How much smaller to step through scales
            minNeighbors=4,        # Minimum neighbors for detection (lower = faster, more false positives)
            minSize=(30, 30),      # Minimum face size
            maxSize=(300, 300)     # Maximum face size
        )

        return faces

    def draw_faces(self, frame, faces):
        """
        Draw rectangles around detected faces.

        Args:
            frame: Input image frame
            faces: List of face rectangles

        Returns:
            Frame with drawn rectangles
        """
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Optional: Add text label
            cv2.putText(
                frame,
                'Face',
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return frame

    def add_info_overlay(self, frame, face_count, fps_value):
        """
        Add information overlay to frame.

        Args:
            frame: Input image frame
            face_count: Number of faces detected
            fps_value: Current FPS

        Returns:
            Frame with overlay
        """
        # Add FPS and face count info
        info_text = f"Faces: {face_count} | FPS: {fps_value:.1f}"
        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return frame

    def run(self):
        """
        Main application loop for continuous face detection.
        Captures frames, detects faces, and displays results in real-time.
        """
        print("Starting face detection... (Press 'q' to quit)")

        frame_count = 0
        start_time = time.time()

        try:
            while self.running:
                # Capture frame from camera
                frame = self.camera.capture_array()

                # Convert RGBA to RGB if necessary
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

                # Detect faces
                faces = self.detect_faces(frame)

                # Draw detected faces
                frame = self.draw_faces(frame, faces)

                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps_value = frame_count / elapsed_time if elapsed_time > 0 else 0

                # Add overlay information
                frame = self.add_info_overlay(frame, len(faces), fps_value)

                # Display the frame (convert RGB to BGR for OpenCV display)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('Face Detection - CSI Camera', frame_bgr)

                # Check for exit command (q key)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nShutting down...")
                    self.running = False

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            self.running = False

        finally:
            self.cleanup()

    def cleanup(self):
        """
        Clean up resources: close camera and display windows.
        """
        print("Cleaning up resources...")
        self.camera.stop()
        self.camera.close()
        cv2.destroyAllWindows()
        print("Resources released successfully")


def main():
    """
    Main entry point for the application.
    """
    app = FaceDetectionApp(
        camera_width=640,
        camera_height=480,
        fps=30
    )
    app.run()


if __name__ == "__main__":
    main()
