import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort

class ModelError(Exception):
    """Fehler beim Laden oder Ausführen des ONNX-Modells."""
    pass

@dataclass(frozen=True)
class Preset:
    level: int
    name: str
    tile_pad: int

PRESETS: dict[int, Preset] = {
    1: Preset(1, "Sicher            (2+ GB VRAM)", 12),
    2: Preset(2, "Balanced          (4+ GB VRAM)", 16),
    3: Preset(3, "Performance       (6+ GB VRAM)", 20),
    4: Preset(4, "Max Performance   (8+ GB VRAM)", 24),
}

# Windows DXGI Ctypes-Strukturen zur GPU-Erkennung
class LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", wintypes.WCHAR * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", LUID),
        ("Flags", wintypes.UINT),
    ]

class DirectMLUpscaler:
    """Verwaltet ONNX Inferenz ohne CPU-Fallback und führt automatische Kachel-Verarbeitung durch."""

    def __init__(self, model_path: Path, preset: Preset):
        self.model_path = model_path
        self.preset = preset
        self.device_id, self.gpu_name = self._detect_best_gpu()
        self.session = self._load_session()
        self.scale_factor, self.model_tile_size = self._detect_model_properties()

    @staticmethod
    def _detect_best_gpu() -> tuple[int, str]:
        """Ermittelt die leistungsfähigste dedizierte GPU via Windows DXGI API."""
        try:
            dxgi = ctypes.windll.dxgi
            factory = ctypes.c_void_p()
            iid_factory1 = (ctypes.c_ubyte * 16)(
                0x03, 0x05, 0x0A, 0x77, 0x5F, 0x22, 0xBA, 0x47,
                0x97, 0x1D, 0x05, 0xD9, 0x9F, 0xB2, 0x15, 0x69
            )
            if dxgi.CreateDXGIFactory1(ctypes.byref(iid_factory1), ctypes.byref(factory)) != 0:
                return 0, "DirectML Graphics Device"

            def get_vtbl_method(obj_p, index, restype, *argtypes):
                vtbl = ctypes.cast(obj_p, ctypes.POINTER(ctypes.c_void_p))[0]
                func_p = ctypes.cast(vtbl, ctypes.POINTER(ctypes.c_void_p))[index]
                return ctypes.WINFUNCTYPE(restype, *argtypes)(func_p)

            enum_adapters1 = get_vtbl_method(
                factory, 10, ctypes.c_long, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p)
            )
            release = get_vtbl_method(factory, 2, wintypes.ULONG, ctypes.c_void_p)

            best_id = 0
            best_vram = -1
            best_name = "DirectML Graphics Device"

            adapter_idx = 0
            while True:
                adapter = ctypes.c_void_p()
                if enum_adapters1(factory, adapter_idx, ctypes.byref(adapter)) != 0 or not adapter:
                    break

                get_desc1 = get_vtbl_method(
                    adapter, 10, ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(DXGI_ADAPTER_DESC1)
                )
                adapter_release = get_vtbl_method(adapter, 2, wintypes.ULONG, ctypes.c_void_p)

                desc = DXGI_ADAPTER_DESC1()
                if get_desc1(adapter, ctypes.byref(desc)) == 0:
                    if not (desc.Flags & 2):  # Software-Render-Knoten ausschließen
                        vram = desc.DedicatedVideoMemory
                        name = desc.Description.strip()
                        if vram > best_vram:
                            best_vram = vram
                            best_id = adapter_idx
                            best_name = name

                adapter_release(adapter)
                adapter_idx += 1

            release(factory)
            return best_id, best_name
        except Exception:
            return 0, "DirectML Graphics Device"

    def _load_session(self) -> ort.InferenceSession:
        """Initialisiert die Inferenz-Session exklusiv auf der ausgewählten DirectML GPU."""
        try:
            providers = [("DmlExecutionProvider", {"device_id": self.device_id})]
            return ort.InferenceSession(str(self.model_path), providers=providers)
        except Exception as e:
            raise ModelError(
                f"ONNX-Modell '{self.model_path.name}' konnte nicht über DirectML (Device ID {self.device_id}) geladen werden.\n"
                f"Details: {e}"
            )

    def get_active_providers(self) -> list[str]:
        """Gibt die von ONNX Runtime tatsächlich genutzten Execution Provider zurück."""
        return self.session.get_providers()

    def _detect_model_properties(self) -> tuple[int, int]:
        """Ermittelt Skalierungsfaktor und Eingabe-Tilegröße direkt aus den ONNX-Metadaten."""
        try:
            in_shape = self.session.get_inputs()[0].shape
            out_shape = self.session.get_outputs()[0].shape

            if len(in_shape) >= 4 and len(out_shape) >= 4:
                in_h = in_shape[2]
                out_h = out_shape[2]
                if isinstance(in_h, int) and isinstance(out_h, int) and in_h > 0 and out_h > 0:
                    scale = int(out_h // in_h)
                    if scale >= 1:
                        return scale, in_h
        except Exception:
            pass

        try:
            input_meta = self.session.get_inputs()[0]
            output_meta = self.session.get_outputs()[0]
            dummy_input = np.zeros((1, 3, 64, 64), dtype=np.float32)
            outputs = self.session.run([output_meta.name], {input_meta.name: dummy_input})
            out_shape = outputs[0].shape
            scale = int(out_shape[2] // 64)
            if scale >= 1:
                return scale, 512
        except Exception as e:
            raise ModelError(
                f"Eigenschaften des Modells '{self.model_path.name}' konnten nicht ermittelt werden.\n"
                f"Ursache: {e}"
            )

        raise ModelError(f"Ungültige Dimensionen im Modell '{self.model_path.name}'.")

    def upscale_image(self, img: Image.Image) -> Image.Image:
        """Zerlegt ein Bild in Kacheln exakt passend zur Modell-Eingabegröße und fügt sie nahtlos zusammen."""
        w, h = img.size
        s = self.scale_factor
        m = self.model_tile_size

        pad = min(self.preset.tile_pad, (m - 1) // 2) if m > 1 else 0
        step = max(1, m - 2 * pad)

        if w < m or h < m:
            work_w, work_h = max(w, m), max(h, m)
            working_img = Image.new("RGB", (work_w, work_h))
            working_img.paste(img, (0, 0))
            if w < work_w:
                right_edge = img.crop((w - 1, 0, w, h)).resize((work_w - w, h))
                working_img.paste(right_edge, (w, 0))
            if h < work_h:
                bottom_edge = working_img.crop((0, h - 1, work_w, h)).resize((work_w, work_h - h))
                working_img.paste(bottom_edge, (0, h))
        else:
            work_w, work_h = w, h
            working_img = img

        output_img = Image.new("RGB", (w * s, h * s))

        for y in range(0, h, step):
            ch = min(step, h - y)
            y1 = max(0, min(work_h - m, y - pad))
            y2 = y1 + m

            for x in range(0, w, step):
                cw = min(step, w - x)
                x1 = max(0, min(work_w - m, x - pad))
                x2 = x1 + m

                tile = working_img.crop((x1, y1, x2, y2))
                upscaled_tile = self._process_tile(tile)

                offset_x = x - x1
                offset_y = y - y1

                crop_x1 = offset_x * s
                crop_y1 = offset_y * s
                crop_x2 = (offset_x + cw) * s
                crop_y2 = (offset_y + ch) * s

                valid_tile = upscaled_tile.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                output_img.paste(valid_tile, (x * s, y * s))

        return output_img

    def _process_tile(self, tile: Image.Image) -> Image.Image:
        """In-Place Speicherverarbeitung für DirectML-Inferenz."""
        img_np = np.array(tile, dtype=np.float32)
        img_np /= 255.0
        
        img_tensor = np.ascontiguousarray(np.transpose(img_np, (2, 0, 1))[np.newaxis, ...])

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        outputs = self.session.run([output_name], {input_name: img_tensor})

        out_tensor = outputs[0][0]
        np.clip(out_tensor * 255.0, 0, 255, out=out_tensor)
        out_np = np.transpose(out_tensor.astype(np.uint8), (1, 2, 0))
        
        return Image.fromarray(out_np)