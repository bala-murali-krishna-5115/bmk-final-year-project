/**
 * Camera & Video Stream Manager
 * Handles HTML5 getUserMedia webcam capture, frame capture, and landmark overlay rendering.
 */

class CameraManager {
  constructor(videoElementId, overlayCanvasId) {
    this.video = document.getElementById(videoElementId);
    this.canvas = document.getElementById(overlayCanvasId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.stream = null;
    this.isStreaming = false;
  }

  async startCamera() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user"
        },
        audio: false
      });

      this.video.srcObject = this.stream;
      await this.video.play();
      this.isStreaming = true;

      if (this.canvas) {
        this.canvas.width = this.video.videoWidth || 640;
        this.canvas.height = this.video.videoHeight || 480;
      }

      this.drawScanningGuide();
      return true;
    } catch (err) {
      console.warn("Webcam access error or permission denied:", err);
      return false;
    }
  }

  stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
      this.isStreaming = false;
    }
    if (this.ctx) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  captureFrameBase64() {
    if (!this.isStreaming || !this.video) return null;

    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = this.video.videoWidth || 640;
    captureCanvas.height = this.video.videoHeight || 480;
    const ctx = captureCanvas.getContext('2d');
    ctx.drawImage(this.video, 0, 0, captureCanvas.width, captureCanvas.height);

    return captureCanvas.toDataURL('image/jpeg', 0.85);
  }

  drawScanningGuide() {
    if (!this.ctx || !this.isStreaming) return;

    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.clearRect(0, 0, w, h);

    // Subtle face alignment ellipse
    this.ctx.save();
    this.ctx.strokeStyle = "rgba(6, 182, 212, 0.4)";
    this.ctx.lineWidth = 2;
    this.ctx.setLineDash([6, 6]);
    this.ctx.beginPath();
    this.ctx.ellipse(w / 2, h / 2, w * 0.22, h * 0.35, 0, 0, 2 * Math.PI);
    this.ctx.stroke();

    // Symmetry central axis
    this.ctx.strokeStyle = "rgba(139, 92, 246, 0.3)";
    this.ctx.lineWidth = 1.5;
    this.ctx.setLineDash([4, 4]);
    this.ctx.beginPath();
    this.ctx.moveTo(w / 2, h * 0.15);
    this.ctx.lineTo(w / 2, h * 0.85);
    this.ctx.stroke();
    this.ctx.restore();
  }

  drawBoundingBox(bbox) {
    if (!this.ctx || !bbox) return;
    this.ctx.save();
    this.ctx.strokeStyle = "#10B981";
    this.ctx.lineWidth = 3;
    this.ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);
    this.ctx.restore();
  }
}

window.CameraManager = CameraManager;
