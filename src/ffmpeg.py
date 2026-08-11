import json
import subprocess
from pathlib import Path

# Zuordnung von Quell-Codecs zu FFmpeg-Encodern
CODEC_MAP: dict[str, str] = {
    "h264": "libx264",
    "hevc": "libx265",
    "av1": "libsvtav1",
    "vp9": "libvpx-vp9",
}

DEFAULT_CODEC = "libx264"
VIDEO_CRF = "18"
PIXEL_FORMAT = "yuv420p"
AUDIO_CODEC = "copy"
SUBTITLE_CODEC = "copy"

class FFmpegError(Exception):
    """Fehler bei der Ausführung von FFmpeg oder FFprobe."""
    pass

class FFmpegService:
    """Verwaltet alle FFmpeg- und FFprobe-Prozesse."""

    @staticmethod
    def get_video_info(video_path: Path) -> tuple[float, int, str]:
        """Liest FPS, Frame-Anzahl und Quell-Codec in einem einzigen FFprobe-Aufruf."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=r_frame_rate,nb_read_packets,nb_frames,codec_name",
            "-of", "json",
            str(video_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)["streams"][0]
            
            num, den = map(int, data["r_frame_rate"].split("/"))
            fps = num / den if den != 0 else 30.0
            
            total_frames = int(data.get("nb_read_packets") or data.get("nb_frames") or 0)
            if total_frames == 0:
                raise ValueError("Frame-Anzahl konnte nicht ermittelt werden.")
            
            raw_codec = data.get("codec_name", "").lower()
            encoder = CODEC_MAP.get(raw_codec, DEFAULT_CODEC)
                
            return fps, total_frames, encoder
        except Exception as e:
            raise FFmpegError(f"Fehler bei der Videoanalyse von '{video_path.name}': {e}")

    @staticmethod
    def extract_frames(video_path: Path, output_dir: Path) -> None:
        """Extrahiert Frames als PNG-Dateien."""
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-q:v", "1",
            str(output_dir / "frame_%08d.png")
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            stderr_out = e.stderr.strip() if e.stderr else "Keine Details verfügbar."
            raise FFmpegError(f"Frame-Extraktion fehlgeschlagen:\n{stderr_out}")

    @staticmethod
    def assemble_video(
        frames_dir: Path, 
        original_video: Path, 
        output_path: Path, 
        fps: float, 
        encoder: str
    ) -> None:
        """Erstellt das Video und übernimmt Audio, Untertitel und Metadaten unverändert."""
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%08d.png"),
            "-i", str(original_video),
            "-map", "0:v:0",
            "-map", "1:a?",
            "-map", "1:s?",
            "-map_metadata", "1",
            "-c:v", encoder,
            "-pix_fmt", PIXEL_FORMAT,
            "-crf", VIDEO_CRF,
            "-c:a", AUDIO_CODEC,
            "-c:s", SUBTITLE_CODEC,
            str(output_path)
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            stderr_out = e.stderr.strip() if e.stderr else "Keine Details verfügbar."
            raise FFmpegError(f"Video-Rekonstruktion fehlgeschlagen:\n{stderr_out}")