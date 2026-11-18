"""Run the pinned u2netp weights on CPU. The bytes are already hashed."""

from __future__ import annotations

import hashlib
import threading

_LOCK = threading.Lock()
_CACHE: dict[str, object] = {}


class EngineError(Exception):
    pass


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def _build(model_bytes: bytes):
    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image
        from rembg.sessions.u2netp import U2netpSession
    except SystemExit as exc:
        raise EngineError("The background library stopped while loading.") from exc
    except Exception as exc:
        raise EngineError("The background library could not be loaded.") from exc

    class Pinned(U2netpSession):
        def __init__(self, raw: bytes):
            self.model_name = "u2netp"
            self.inner_session = ort.InferenceSession(
                raw,
                sess_options=ort.SessionOptions(),
                providers=["CPUExecutionProvider"],
            )
            active = list(self.inner_session.get_providers())
            if active != ["CPUExecutionProvider"]:
                raise EngineError("The local session did not stay on CPU.")

        def predict(self, img, *args, **kwargs):
            ort_outs = self.inner_session.run(
                None,
                self.normalize(
                    img,
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                    (320, 320),
                ),
            )
            pred = ort_outs[0][:, 0, :, :]
            high = float(np.max(pred))
            low = float(np.min(pred))
            if high > low:
                pred = (pred - low) / (high - low)
            else:
                pred = np.zeros(pred.shape, dtype=pred.dtype)
            pred = np.squeeze(pred)
            mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
            mask = mask.resize(img.size, Image.Resampling.LANCZOS)
            return [mask]

    return Pinned(model_bytes)


def _session(model_bytes: bytes):
    digest = hashlib.sha256(model_bytes).hexdigest()
    with _LOCK:
        cached = _CACHE.get("session")
        if isinstance(cached, tuple) and cached[0] == digest:
            return cached[1]
        session = _build(model_bytes)
        _CACHE["session"] = (digest, session)
        return session


def cut_rgba(image, model_bytes: bytes):
    """Return an RGBA image. Pixel size matches the stored input. EXIF is not applied."""
    session = _session(model_bytes)
    rgb = image.convert("RGB")
    with _LOCK:
        mask = session.predict(rgb)[0]
    rgba = rgb.convert("RGBA")
    rgba.putalpha(mask)
    return rgba
