import pytest
import numpy as np
from computer_vision.camera.camera_manager import CameraManager

def test_camera_manager_initialization():
    """Test CameraManager parameter initialization."""
    cam = CameraManager(camera_index=0, width=1280, height=720, fps=15)
    assert cam.camera_index == 0
    assert cam.target_width == 1280
    assert cam.target_height == 720
    assert cam.target_fps == 15
    assert not cam.is_active()

def test_read_frame_inactive_camera():
    """Test reading from an uninitialized camera returns (False, None)."""
    cam = CameraManager(camera_index=0)
    success, frame = cam.read_frame()
    assert success is False
    assert frame is None

def test_invalid_camera_index_graceful_failure():
    """Test starting an invalid camera index (e.g. 999) fails gracefully without crashing."""
    cam = CameraManager(camera_index=999)
    success = cam.start_camera()
    assert success is False
    assert not cam.is_active()
    
    # Reading from invalid camera must return False, None
    ret, frame = cam.read_frame()
    assert ret is False
    assert frame is None
    
    # Release must be safe to call even when failed
    cam.release_camera()
    assert not cam.is_active()

def test_camera_lifecycle_if_available():
    """
    Attempts to initialize physical camera 0.
    If a physical camera exists, verifies active status, frame retrieval, and clean release.
    If no camera exists on the system, gracefully asserts start_camera returns False.
    """
    cam = CameraManager(camera_index=0)
    success = cam.start_camera()
    if success:
        assert cam.is_active()
        ret, frame = cam.read_frame()
        assert ret is True
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3  # Height x Width x Channels (BGR)
        w, h = cam.get_frame_dimensions()
        assert w > 0 and h > 0
        cam.release_camera()
        assert not cam.is_active()
    else:
        # No physical camera attached in test environment
        assert not cam.is_active()
