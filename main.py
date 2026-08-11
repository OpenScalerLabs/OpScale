import sys
import traceback
from pathlib import Path

from src.environment import EnvironmentChecker, EnvironmentError
from src.ffmpeg import FFmpegError
from src.pipeline import Pipeline
from src.upscaler import PRESETS, ModelError


def get_base_dir() -> Path:
    """Ermittelt das Basisverzeichnis für Python-Skripte und PyInstaller-EXEs."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent.resolve()
    return Path(__file__).parent.resolve()


def print_error(title: str, message: str) -> None:
    """Gibt eine formatierte Fehlermeldung auf der Konsole aus."""
    print(f"\n❌ FEHLER: {title}")
    print("-" * 50)
    print(f"{message}")
    print("-" * 50)


def _log_error(log_file: Path, message: str) -> None:
    """Protokolliert Fehlerdetails in der Log-Datei."""
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{message}\n" + "=" * 50 + "\n")
    except Exception:
        pass


def main() -> None:
    base_dir = get_base_dir()
    log_file = base_dir / "logs" / "opscale_error.log"

    print("==================================================")
    print("      OpScale – The Open ONNX Video Upscaler")
    print("==================================================")

    # 1. Systemprüfung
    sys.stdout.write("Systemprüfung ... ")
    sys.stdout.flush()
    try:
        EnvironmentChecker.validate_all(base_dir)
        print("OK")
    except EnvironmentError as e:
        print("FEHLER")
        print_error("Systemprüfung fehlgeschlagen", str(e))
        sys.exit(1)

    input_dir = base_dir / "input"
    models_dir = base_dir / "models"
    output_dir = base_dir / "output"

    # 2. Eingabevideo automatisch wählen
    videos = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov")
    ]
    if not videos:
        print_error(
            "Kein Eingabevideo gefunden",
            "Bitte legen Sie ein Video (.mp4, .mkv, .avi, .mov) in den Ordner 'input/'.",
        )
        sys.exit(1)
    selected_video = videos[0]

    # 3. ONNX-Modell auswählen
    models = sorted(list(models_dir.glob("*.onnx")))
    if not models:
        print_error(
            "Kein ONNX-Modell gefunden",
            "Bitte legen Sie mindestens eine '.onnx'-Datei in den Ordner 'models/'.",
        )
        sys.exit(1)

    selected_model_idx = 0
    if len(models) > 1:
        print("\nVerfügbare Modelle:")
        for idx, m in enumerate(models, start=1):
            print(f"  [{idx}] {m.name}")
        try:
            choice = int(input("\nModell wählen (Nummer): ")) - 1
            if 0 <= choice < len(models):
                selected_model_idx = choice
        except ValueError:
            pass
    selected_model = models[selected_model_idx]

    # 4. Qualitätsstufe auswählen
    print("\nQualitätsstufen:")
    for lvl, p in PRESETS.items():
        print(f"  [{lvl}] {p.name}")

    preset_level = 2
    try:
        user_input = input("\nStufe wählen (1-4, Standard [2]): ").strip()
        if user_input:
            val = int(user_input)
            if val in PRESETS:
                preset_level = val
    except ValueError:
        pass
    selected_preset = PRESETS[preset_level]

    # 5. Pipeline initialisieren, Konfiguration ausgeben und starten
    try:
        pipeline = Pipeline(selected_model, selected_preset)

        providers_str = ", ".join(pipeline.upscaler.get_active_providers())
        gpu_name = pipeline.upscaler.gpu_name
        dev_id = pipeline.upscaler.device_id
        scale_val = pipeline.upscaler.scale_factor
        tile_val = pipeline.upscaler.model_tile_size

        print("\nKonfiguration:")
        print(f"  GPU:        {gpu_name}")
        print(f"  Provider:   {providers_str}")
        print(f"  Device ID:  {dev_id}")
        print(f"  Eingabe:    {selected_video.name}")
        print(f"  Modell:     {selected_model.name}")
        print(f"  Input-Tile: {tile_val}x{tile_val}px")
        print(f"  Scale:      {scale_val}x")
        print(f"  Quality:    {selected_preset.name}")

        output_filename = (
            f"{selected_video.stem}_{selected_model.stem}_{scale_val}x.mp4"
        )
        output_video = output_dir / output_filename

        pipeline.run(selected_video, output_video, base_dir)
        print(f"\n✔ Fertiggestellt! Ausgabedatei: {output_video.name}")

    except KeyboardInterrupt:
        print("\n\nAbbruch durch Benutzer.")
        print("Temporäre Dateien werden entfernt...")
        sys.exit(0)
    except (ModelError, FFmpegError) as e:
        _log_error(log_file, str(e))
        print_error("Verarbeitungsfehler", str(e))
        sys.exit(1)
    except Exception as e:
        tb = traceback.format_exc()
        _log_error(log_file, f"{e}\n{tb}")
        print_error(
            "Unerwarteter Systemfehler",
            f"{e}\n\nDetails wurden in '{log_file.relative_to(base_dir)}' protokolliert.",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()