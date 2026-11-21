"""
IntelliCrash Edge - PiCam Crash Evidence Recorder
=================================================
Keeps a rolling, TIME-based video buffer (default 15 s). When a crash is confirmed the recorder keeps filming
for another `post_seconds` (default 15 s) and then writes ONE clip = pre-crash + post-crash footage
(default 30 s) as an MP4 in ./evidence.

Why the clip used to be only 4-5 s long:
  * the old buffer held "the last 225 frames", and the clip was written at a fixed 15 fps. If the camera
    delivered fewer than 15 frames/s (busy Pi) the clip played back faster than real time;
  * the buffer only holds what was recorded since start-up, so a crash a few seconds after launch (typical in a
    demo or a replay) produced a few-second clip;
  * nothing recorded AFTER the impact.
Now every frame carries a timestamp, the clip is written at the MEASURED frame rate (so its length equals the real
time it covers) and post-crash footage is appended.

save_crash_clip() returns the final path immediately and finishes the file in a background thread (the inference
loop is never blocked). Use wait_for_clip(path) / the "_tmp" naming to know when it is ready.
"""

import os
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")        # the H.264 -> MPEG-4 fallback would otherwise print harmless FFmpeg errors
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
import time
import threading
from collections import deque
from pathlib import Path
from datetime import datetime
import numpy as np

GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


class PiCamCrashRecorder:
    """Rolling circular video buffer for collision event recording (pre-crash + post-crash)."""

    def __init__(
        self,
        buffer_seconds: int = 15,
        fps: int = 15,
        resolution: tuple = (640, 480),
        output_dir: str = None,
        enabled: bool = True,
        post_seconds: int = 15,
    ):
        self.buffer_seconds = buffer_seconds      # pre-crash seconds kept
        self.post_seconds = post_seconds          # post-crash seconds recorded after the impact
        self.fps = fps
        self.resolution = resolution
        self.frame_buffer = deque()               # (timestamp, packed_frame); trimmed by TIME, not by count
        self._keep_seconds = buffer_seconds + post_seconds + 3
        self.enabled = enabled
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._done = {}                           # final clip path -> threading.Event
        self.camera_backend = "none"

        self.output_dir = Path(output_dir or Path(__file__).resolve().parent / "evidence")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import cv2
            self._cv2 = cv2
        except Exception:
            self._cv2 = None

        if not self.enabled:
            print("[INFO] PiCam Crash Recorder is disabled.")
            return

        self._init_camera()
        if self.camera_backend != "none":
            self.start()

    # ------------------------------------------------------------------ camera
    def _init_camera(self):
        # 1. picamera2 (Raspberry Pi OS Bookworm)
        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(main={"size": self.resolution, "format": "RGB888"})
            self.picam2.configure(config)
            self.picam2.start()
            self.camera_backend = "picamera2"
            print(f"{GREEN}[OK] PiCam initialized via picamera2 ({self.resolution[0]}x{self.resolution[1]} @ {self.fps} fps){RESET}")
            return
        except Exception:
            pass

        # 2. OpenCV VideoCapture (V4L2 / USB camera)
        try:
            import cv2
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                self.camera_backend = "opencv"
                print(f"{GREEN}[OK] Camera initialized via OpenCV VideoCapture(0){RESET}")
                return
            self.cap.release()
        except Exception:
            pass

        # 3. No camera: synthetic telematics clip
        self.camera_backend = "synthetic"
        print(f"{YELLOW}[INFO] Physical camera not detected. Evidence clips will be a synthetic telematics HUD video.{RESET}")

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="PiCamCaptureThread")
        self._thread.start()

    def _pack(self, frame):
        """JPEG-compress the frame: ~30 KB instead of ~900 KB, so 33 s of video costs about 15 MB of RAM."""
        if self._cv2 is not None:
            ok, enc = self._cv2.imencode(".jpg", frame, [int(self._cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                return enc.tobytes()
        return frame

    def _unpack(self, packed):
        if isinstance(packed, (bytes, bytearray)):
            return self._cv2.imdecode(np.frombuffer(packed, np.uint8), self._cv2.IMREAD_COLOR)
        return packed

    def _capture_loop(self):
        frame_interval = 1.0 / self.fps
        while self._running:
            loop_start = time.time()
            frame = None
            if self.camera_backend == "picamera2":
                try:
                    frame = self.picam2.capture_array()
                except Exception:
                    frame = None
            elif self.camera_backend == "opencv":
                try:
                    ret, raw = self.cap.read()
                    if ret:
                        frame = raw
                except Exception:
                    frame = None

            if frame is not None:
                packed = self._pack(frame)
                now = time.time()
                with self._lock:
                    self.frame_buffer.append((now, packed))
                    cutoff = now - self._keep_seconds
                    while self.frame_buffer and self.frame_buffer[0][0] < cutoff:
                        self.frame_buffer.popleft()

            time.sleep(max(0.001, frame_interval - (time.time() - loop_start)))

    # -------------------------------------------------------------- crash clip
    def save_crash_clip(
        self,
        event_timestamp: str = None,
        severity: str = "SEVERE",
        p_final: float = 0.85,
        gps_coords: tuple = None,
        post_seconds: float = None,
    ) -> str:
        """Returns the FINAL clip path immediately; the file appears when the post-crash footage is encoded
        (about post_seconds + a few seconds later)."""
        if not self.enabled or self._cv2 is None:
            if self._cv2 is None:
                print("[WARN] OpenCV not installed. Skipping MP4 encoding.")
            return ""

        ts = event_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"crash_evidence_{severity.lower()}_{ts}.mp4"
        post = self.post_seconds if post_seconds is None else post_seconds
        ev = threading.Event()
        self._done[str(out_path)] = ev
        t_crash = time.time()
        threading.Thread(target=self._finalize, args=(out_path, ev, t_crash, post, ts, severity, p_final, gps_coords),
                         daemon=True, name="CrashClipWriter").start()
        print(f"{GREEN}[VIDEO] Recording {self.buffer_seconds} s before + {post:.0f} s after the impact -> {out_path.name}{RESET}")
        return str(out_path)

    def wait_for_clip(self, path, timeout: float = 30.0) -> bool:
        ev = self._done.get(str(path))
        return bool(ev and ev.wait(timeout))

    def _finalize(self, out_path, ev, t_crash, post, ts, severity, p_final, gps):
        cv2 = self._cv2
        try:
            end = t_crash + post
            while time.time() < end and self._running:
                time.sleep(0.25)
            with self._lock:
                items = [(t, f) for (t, f) in self.frame_buffer if t_crash - self.buffer_seconds <= t <= end]

            frames, writer_fps = [], float(self.fps)
            if self.camera_backend == "synthetic" or len(items) < 2 * self.fps:
                total = int((self.buffer_seconds + post) * self.fps)
                frames = self._generate_telematics_clip(ts, severity, p_final, gps, total)
                note = "synthetic"
            else:
                t0, t1 = items[0][0], items[-1][0]
                if t1 > t0:
                    writer_fps = max(2.0, min(30.0, (len(items) - 1) / (t1 - t0)))   # measured, so length = real time
                for (t, packed) in items:
                    fr = self._unpack(packed)
                    if fr is None:
                        continue
                    fr = fr.copy()
                    dt = t - t_crash
                    cv2.putText(fr, f"IntelliCrash  {severity}  T{dt:+.1f}s", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                    cv2.putText(fr, f"IntelliCrash  {severity}  T{dt:+.1f}s", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    frames.append(fr)
                note = "camera"

            if not frames:
                return
            h, w = frames[0].shape[:2]
            tmp = out_path.with_name(out_path.stem + "_tmp.mp4")
            writer = None
            for code in ("avc1", "mp4v"):
                wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*code), writer_fps, (w, h))
                if wr.isOpened():
                    writer = wr
                    break
                wr.release()
            if writer is None:
                print("[WARN] Could not open a video writer.")
                return
            for fr in frames:
                writer.write(fr)
            writer.release()
            os.replace(tmp, out_path)
            secs = len(frames) / writer_fps
            print(f"{GREEN}[OK] Crash video saved: {out_path.name} | {len(frames)} frames | {secs:.1f} s ({note}) | "
                  f"{out_path.stat().st_size / 1e6:.1f} MB{RESET}")
        except Exception as e:
            print(f"{YELLOW}[WARN] Crash clip failed: {e}{RESET}")
        finally:
            ev.set()

    def _generate_telematics_clip(self, ts, severity, p_final, gps, total_frames):
        """Synthetic forensic HUD clip, used only when no camera produced frames."""
        cv2 = self._cv2
        frames = []
        width, height = self.resolution
        for i in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (20, 24, 33)
            cv2.line(frame, (0, int(height * 0.6)), (width, int(height * 0.6)), (50, 60, 75), 1)
            cv2.line(frame, (int(width * 0.5), int(height * 0.6)), (int(width * 0.2), height), (80, 90, 110), 2)
            cv2.line(frame, (int(width * 0.5), int(height * 0.6)), (int(width * 0.8), height), (80, 90, 110), 2)
            cv2.rectangle(frame, (15, 15), (width - 15, 95), (35, 45, 60), -1)
            cv2.rectangle(frame, (15, 15), (width - 15, 95), (70, 90, 120), 1)
            cv2.putText(frame, "INTELLICRASH BLACKBOX FORENSIC RECORDER", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, f"STATUS: [CRASH EVENT] | SEVERITY: {severity} | P: {p_final:.2f}", (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)
            cv2.putText(frame, f"GPS: {(f'{gps[0]:.5f} N, {gps[1]:.5f} E' if gps else 'no fix')} | TS: {ts} | FRAME: {i+1}/{total_frames}", (25, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
            if total_frames * self.buffer_seconds // max(1, (self.buffer_seconds + self.post_seconds)) - 4 <= i <= total_frames * self.buffer_seconds // max(1, (self.buffer_seconds + self.post_seconds)) + 6:
                cv2.circle(frame, (width // 2, height // 2), 70, (0, 0, 220), -1)
                cv2.putText(frame, "IMPACT", (width // 2 - 45, height // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frames.append(frame)
        return frames

    def stop(self, wait_for_clips: bool = True):
        """Stop the camera. Clips still being finished (post-crash footage + encoding) are allowed to complete first,
        otherwise a replay that ends, or Ctrl+C right after a crash, would lose or shorten the evidence video."""
        if wait_for_clips:
            pending = [ev for ev in list(self._done.values()) if not ev.is_set()]
            if pending:
                print(f"{YELLOW}[VIDEO] Finishing the crash video before shutdown (up to {self.post_seconds + 20:.0f} s)...{RESET}", flush=True)
                deadline = time.time() + self.post_seconds + 20
                for ev in pending:
                    ev.wait(max(0.0, deadline - time.time()))
        self._running = False
        if hasattr(self, "picam2"):
            try:
                self.picam2.stop()
            except Exception:
                pass
        if hasattr(self, "cap"):
            try:
                self.cap.release()
            except Exception:
                pass


if __name__ == "__main__":
    print("Testing PiCamCrashRecorder standalone (pre 15 s + post 15 s = 30 s clip)...")
    recorder = PiCamCrashRecorder(buffer_seconds=15, post_seconds=15, fps=15, enabled=True)
    time.sleep(5)
    clip = recorder.save_crash_clip(severity="SEVERE", p_final=0.92)
    print("Clip will be written to:", clip)
    recorder.wait_for_clip(clip, timeout=40)
    recorder.stop()
