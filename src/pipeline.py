from collections import deque
import sys
import tempfile
import time
from pathlib import Path
from PIL import Image

from src.ffmpeg import FFmpegService
from src.upscaler import DirectMLUpscaler, Preset

DEBUG_KEEP_TEMP: bool = False

class Pipeline:
    """Steuert den Verarbeitungsablauf und gibt transparente Fortschrittsdaten aus."""

    def __init__(self, model_path: Path, preset: Preset):
        self.model_path = model_path
        self.preset = preset
        self.upscaler = DirectMLUpscaler(model_path, preset)

    def run(self, input_video: Path, output_video: Path, base_dir: Path) -> None:
        fps, total_frames, encoder = FFmpegService.get_video_info(input_video)

        if DEBUG_KEEP_TEMP:
            temp_path = base_dir / "temp"
            temp_path.mkdir(exist_ok=True)
            self._execute(input_video, output_video, temp_path, fps, encoder)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                self._execute(input_video, output_video, Path(temp_dir), fps, encoder)

    def _execute(
        self, 
        input_video: Path, 
        output_video: Path, 
        temp_path: Path, 
        fps: float, 
        encoder: str
    ) -> None:
        frames_in = temp_path / "in"
        frames_out = temp_path / "out"
        frames_in.mkdir(exist_ok=True)
        frames_out.mkdir(exist_ok=True)

        # Phase 1: Extraktion
        print("\nPhase 1/3")
        print("Extrahiere Frames...")
        FFmpegService.extract_frames(input_video, frames_in)

        input_files = sorted(frames_in.glob("frame_*.png"))
        total_extracted = len(input_files)
        print(f"✓ {total_extracted} Frames extrahiert")

        # Phase 2: Upscaling
        print("\nPhase 2/3")
        print("Upscaling...")

        frame_times = deque(maxlen=100)
        prev_time = time.perf_counter()

        for idx, frame_file in enumerate(input_files, start=1):
            with Image.open(frame_file) as img:
                scaled_img = self.upscaler.upscale_image(img)
                scaled_img.save(frames_out / frame_file.name)

            now = time.perf_counter()
            frame_times.append(now - prev_time)
            prev_time = now

            avg_dt = sum(frame_times) / len(frame_times)
            current_fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
            eta_sec = (total_extracted - idx) * avg_dt

            self._render_progress(idx, total_extracted, current_fps, eta_sec)

        sys.stdout.write("\n")
        sys.stdout.flush()

        # Phase 3: Video zusammensetzen
        print("\nPhase 3/3")
        print("Video zusammensetzen...")
        FFmpegService.assemble_video(frames_out, input_video, output_video, fps, encoder)

        print("✓ Audio übernommen")
        print("✓ Untertitel übernommen")
        print("✓ Metadaten übernommen")

    @staticmethod
    def _render_progress(current: int, total: int, fps: float, eta_seconds: float) -> None:
        percent = (current / total) * 100
        bar_len = 24
        filled = int(bar_len * current // total)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        eta_str = time.strftime("%H:%M:%S", time.gmtime(max(0.0, eta_seconds)))

        sys.stdout.write(
            f"\r{bar}  {current} / {total} Frames  ({percent:5.1f}%)  {fps:4.1f} FPS  ETA {eta_str}"
        )
        sys.stdout.flush()