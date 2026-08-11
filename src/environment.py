import shutil
import sys
from pathlib import Path
import onnxruntime as ort

class EnvironmentError(Exception):
    """Fehler bei System- und Umgebungsvoraussetzungen."""
    pass

class EnvironmentChecker:
    """Prüft Systemkomponenten vor der Ausführung."""

    MIN_PYTHON = (3, 11)
    REQUIRED_FOLDERS = ("input", "models", "output", "logs")

    @classmethod
    def validate_all(cls, base_dir: Path) -> None:
        cls._check_python_version()
        cls._check_ffmpeg()
        cls._check_directml()
        cls._check_directories_and_permissions(base_dir)

    @classmethod
    def _check_python_version(cls) -> None:
        if sys.version_info < cls.MIN_PYTHON:
            raise EnvironmentError(
                f"Python {cls.MIN_PYTHON[0]}.{cls.MIN_PYTHON[1]} oder neuer erforderlich.\n"
                f"Installierte Version: {sys.version.split()[0]}"
            )

    @classmethod
    def _check_ffmpeg(cls) -> None:
        missing = [cmd for cmd in ("ffmpeg", "ffprobe") if not shutil.which(cmd)]
        if missing:
            raise EnvironmentError(
                f"Folgende Programme wurden im PATH nicht gefunden: {', '.join(missing)}\n"
                "Bitte installieren Sie FFmpeg und fügen Sie es zum System-PATH hinzu."
            )

    @classmethod
    def _check_directml(cls) -> None:
        providers = ort.get_available_providers()
        if "DmlExecutionProvider" not in providers:
            raise EnvironmentError(
                "DirectML-Unterstützung fehlt in ONNX Runtime.\n"
                "Bitte installieren Sie das Paket 'onnxruntime-directml'."
            )

    @classmethod
    def _check_directories_and_permissions(cls, base_dir: Path) -> None:
        for folder_name in cls.REQUIRED_FOLDERS:
            folder = base_dir / folder_name
            folder.mkdir(exist_ok=True)
            test_file = folder / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                raise EnvironmentError(
                    f"Keine Schreibrechte im Ordner '{folder_name}/': {e}"
                )