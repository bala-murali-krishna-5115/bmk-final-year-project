import logging
import cv2
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class CameraManager:
    """
    Webcam initialization, frame capture, and configuration manager.
    Provides robust handling for camera connections, frame reading, and resource release.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        self.camera_index = camera_index
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_active = False

    def start_camera(self) -> bool:
        """
        Initializes and opens the specified camera device.
        Configures frame resolution and target FPS.

        Returns:
            bool: True if camera started successfully, False otherwise.
        """
        if self._is_active and self.cap is not None and self.cap.isOpened():
            logger.info(f"Camera index {self.camera_index} is already running.")
            return True

        logger.info(f"Initializing camera at index {self.camera_index}...")
        try:
            # OpenCV DirectShow on Windows or default backend
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if cv2.CAP_DSHOW else cv2.CAP_ANY)
            
            if not self.cap.isOpened():
                # Fallback to standard CAP_ANY backend if DSHOW fails
                self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera at index {self.camera_index}.")
                self._is_active = False
                return False

            # Set resolution and FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Test frame read to confirm active capture
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error(f"Camera at index {self.camera_index} opened but failed to read initial frame.")
                self.release_camera()
                return False

            self._is_active = True
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Camera index {self.camera_index} started successfully ({actual_w}x{actual_h}).")
            return True

        except Exception as e:
            logger.error(f"Exception encountered while starting camera: {str(e)}")
            self.release_camera()
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Captures the next image frame from the active webcam.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success_flag, BGR_image_array)
        """
        if not self._is_active or self.cap is None or not self.cap.isOpened():
            logger.warning("Attempted to read frame from inactive or uninitialized camera.")
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                logger.warning("Failed to retrieve valid frame from camera.")
                return False, None
            
            return True, frame

        except Exception as e:
            logger.error(f"Exception during frame read: {str(e)}")
            return False, None

    def release_camera(self) -> None:
        """Releases the camera device and cleans up resources."""
        if self.cap is not None:
            logger.info(f"Releasing camera index {self.camera_index}...")
            self.cap.release()
            self.cap = None
        self._is_active = False
        logger.info("Camera released.")

    def is_active(self) -> bool:
        """Returns True if the camera is initialized and open."""
        return self._is_active and self.cap is not None and self.cap.isOpened()

    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Returns current frame width and height."""
        if self.is_active() and self.cap is not None:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return self.target_width, self.target_height

    def __del__(self):
        """Ensure resources are released on garbage collection."""
        self.release_camera()
