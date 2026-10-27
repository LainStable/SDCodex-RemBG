"""SDCodex-RemBG backend — high-precision neural background removal and replacement."""

import base64
import io
import json
import mimetypes
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from urllib.parse import quote, unquote
from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter, ImageOps
import numpy as np

rembg_bp = Blueprint("rembg_bp", __name__, url_prefix="/rembg/api")


def _core_session_missing() -> bool:
    try:
        from sdcodex import auth as core_auth
    except Exception:
        return False  # standalone deployment
    try:
        return core_auth.current_user() is None
    except Exception:
        return True


@rembg_bp.before_request
def _rembg_auth_guard():
    if _core_session_missing():
        return jsonify({"error": "unauthorized"}), 401
    return None

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".jfif"
}

_sessions = {}
_session_errors = {}
_model_cache_warning = ""
_session_lock = threading.Lock()

# Batch state
_batch_lock = threading.Lock()
_batch_cancel = threading.Event()
_batch_pause = threading.Event()
_batch_status = {
    "running": False,
    "state": "idle",
    "total": 0,
    "processed": 0,
    "errors": 0,
    "current_file": "",
    "items": [],
    "started_at": 0,
    "message": "",
}


def _plugin_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _workspace_root():
    cur = _plugin_dir()
    for _ in range(5):
        if os.path.isdir(os.path.join(cur, "plugins")) and os.path.isdir(os.path.join(cur, "backend")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(os.path.join(_plugin_dir(), "..", ".."))


def get_input_dir():
    ws = _workspace_root()
    cand = os.environ.get("REMBG_INPUT") or os.environ.get("GALLERY_ROOT") or os.path.join(ws, "downloads")
    if not os.path.isdir(cand):
        os.makedirs(cand, exist_ok=True)
    return os.path.abspath(cand)


def get_output_dir():
    ws = _workspace_root()
    cand = os.environ.get("REMBG_OUTPUT") or os.path.join(ws, "rembg_output")
    if not os.path.isdir(cand):
        os.makedirs(cand, exist_ok=True)
    return os.path.abspath(cand)


def get_model_cache_dir():
    """Use the plugin's persistent model volume for ONNX model weights."""
    global _model_cache_warning
    _model_cache_warning = ""
    cache_dir = os.environ.get("U2NET_HOME") or os.environ.get("HFH") or "/data/huggingface"
    cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as exc:
        _model_cache_warning = f"Could not use configured model cache {cache_dir}: {exc}"
        cache_dir = os.path.abspath(os.path.expanduser("~/.u2net"))
        os.makedirs(cache_dir, exist_ok=True)
    # rembg checks this variable when it initializes a session.
    os.environ["U2NET_HOME"] = cache_dir
    return cache_dir


def _free_bytes(path):
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def _is_out_of_space(exc):
    text = str(exc).lower()
    return "no space left" in text or "errno 28" in text


def _fallback_cache_dir():
    """A model cache on a volume that still has room.

    The configured cache is normally a mounted data volume. When that volume
    is full the model download dies with ENOSPC and the plugin looks broken,
    so fall back to a local cache that can still be written.
    """
    for candidate in ("~/.u2net", "/tmp/u2net", "/var/tmp/u2net"):
        path = os.path.abspath(os.path.expanduser(candidate))
        try:
            os.makedirs(path, exist_ok=True)
            probe = os.path.join(path, ".write-probe")
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("ok")
            os.remove(probe)
        except OSError:
            continue
        if _free_bytes(path) > 64 * 1024 * 1024:
            return path
    return ""


def get_model_session(model_name):
    """Retrieve or initialize a cached rembg model session."""
    name = (model_name or "birefnet-general").strip().lower()
    # Normalize aliases
    alias_map = {
        "birefnet": "birefnet-general",
        "birefnet-general": "birefnet-general",
        "birefnet-portrait": "birefnet-portrait",
        "u2net": "u2net",
        "isnet": "isnet-general-use",
        "isnet-general-use": "isnet-general-use",
        "silueta": "silueta",
    }
    target_name = alias_map.get(name, name)

    with _session_lock:
        if target_name in _sessions:
            return _sessions[target_name], target_name
        try:
            get_model_cache_dir()
            from rembg import new_session
            sess = new_session(target_name)
            _sessions[target_name] = sess
            _session_errors.pop(target_name, None)
            return sess, target_name
        except Exception as exc:
            # The model volume may simply be full. Retry once against a local
            # cache so the plugin still works while the disk is being sorted.
            fallback = _fallback_cache_dir() if _is_out_of_space(exc) else ""
            if fallback:
                try:
                    os.environ["U2NET_HOME"] = fallback
                    from rembg import new_session
                    sess = new_session(target_name)
                    _sessions[target_name] = sess
                    _session_errors.pop(target_name, None)
                    global _model_cache_warning
                    _model_cache_warning = (
                        f"Model cache moved to {fallback} — the configured volume is full"
                    )
                    return sess, target_name
                except Exception as inner:
                    exc = inner
            # Do not silently substitute another model: the caller needs to
            # report the model that actually ran (or that inference is unavailable).
            _session_errors[target_name] = f"{type(exc).__name__}: {exc}"
            return None, target_name
    return None, target_name


def pil_to_base64(img, fmt="PNG"):
    buf = io.BytesIO()
    save_fmt = "PNG" if fmt.upper() in ("PNG", "MASK") else "WEBP"
    img.save(buf, format=save_fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if save_fmt == "PNG" else "image/webp"
    return f"data:{mime};base64,{b64}"


def base64_to_pil(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGBA")


def hex_to_rgb(hex_code, default=(15, 19, 28)):
    if not hex_code:
        return default
    s = hex_code.strip().lstrip("#")
    if len(s) == 3:
        s = "".join([c * 2 for c in s])
    if len(s) == 6:
        try:
            return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            pass
    return default


def apply_matte_refinements(cutout_img, threshold=0.85, erosion=0.0, feather=0.0, despill=False):
    """Refine cutout alpha channel with threshold, erosion/dilation, and feathering."""
    if cutout_img.mode != "RGBA":
        cutout_img = cutout_img.convert("RGBA")

    r, g, b, a = cutout_img.split()

    # 1. Alpha Thresholding
    if threshold is not None and threshold > 0:
        thresh_val = int(threshold * 255)
        # Smooth threshold curve or hard binarization
        a_arr = np.array(a, dtype=np.uint8)
        mask = a_arr >= thresh_val
        a_arr = np.where(mask, a_arr, 0)
        a = Image.fromarray(a_arr, mode="L")

    # 2. Erosion / Dilation
    if erosion > 0:
        filter_size = max(3, int(round(erosion * 2)) | 1)
        a = a.filter(ImageFilter.MinFilter(filter_size))
    elif erosion < 0:
        filter_size = max(3, int(round(abs(erosion) * 2)) | 1)
        a = a.filter(ImageFilter.MaxFilter(filter_size))

    # 3. Feathering
    if feather > 0:
        a = a.filter(ImageFilter.GaussianBlur(radius=feather))

    # 4. Despill (Suppress green/cyan fringe on edges)
    if despill:
        arr_rgb = np.stack([np.array(r), np.array(g), np.array(b)], axis=-1).astype(np.float32)
        arr_a = np.array(a, dtype=np.float32) / 255.0
        # Reduce green/cyan overflow where G > max(R, B)
        rb_max = np.maximum(arr_rgb[..., 0], arr_rgb[..., 2])
        spill = np.maximum(0, arr_rgb[..., 1] - rb_max)
        arr_rgb[..., 1] -= spill * 0.7 * (1.0 - arr_a * 0.5)
        arr_rgb = np.clip(arr_rgb, 0, 255).astype(np.uint8)
        r = Image.fromarray(arr_rgb[..., 0])
        g = Image.fromarray(arr_rgb[..., 1])
        b = Image.fromarray(arr_rgb[..., 2])

    return Image.merge("RGBA", (r, g, b, a))


# --------------------------------------------------------------------------- #
#  Endpoints
# --------------------------------------------------------------------------- #

@rembg_bp.get("/status")
def rembg_status():
    cuda_avail = False
    device_name = "CPU"
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    try:
        import rembg  # noqa: F401
        rembg_available = True
    except Exception:
        rembg_available = False
    return jsonify({
        "engine": "BiRefNet / RemBG Neural Matting",
        "device": device_name,
        "cuda": cuda_avail,
        "active_models": list(_sessions.keys()),
        "rembg_available": rembg_available,
        "input_dir": get_input_dir(),
        "output_dir": get_output_dir(),
        "model_cache_dir": get_model_cache_dir(),
        "model_cache_warning": _model_cache_warning,
        "model_errors": dict(_session_errors),
    })


@rembg_bp.get("/models")
def list_models():
    """Available local ONNX models and their upstream download sources."""
    models = [
        ("birefnet-general", "BiRefNet General", "Recommended general model", "BiRefNet-general-epoch_244.onnx", True),
        ("birefnet-general-lite", "BiRefNet General Lite", "Lightweight general model", "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx", False),
        ("birefnet-portrait", "BiRefNet Portrait", "Human portraits", "BiRefNet-portrait-epoch_150.onnx", False),
        ("birefnet-dis", "BiRefNet DIS", "Dichotomous image segmentation", "BiRefNet-DIS-epoch_590.onnx", False),
        ("birefnet-hrsod", "BiRefNet HRSOD", "High-resolution salient objects", "BiRefNet-HRSOD_DHU-epoch_115.onnx", False),
        ("birefnet-cod", "BiRefNet COD", "Concealed object detection", "BiRefNet-COD-epoch_125.onnx", False),
        ("birefnet-massive", "BiRefNet Massive", "Broad dataset model", "BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.onnx", False),
        ("bria-rmbg", "BRIA RMBG 2.0", "High quality; review license terms", "bria-rmbg-2.0.onnx", False),
        ("u2net", "U2Net", "General purpose", "u2net.onnx", False),
        ("u2netp", "U2NetP", "Lightweight U2Net", "u2netp.onnx", False),
        ("u2net_human_seg", "U2Net Human Segmentation", "Human segmentation", "u2net_human_seg.onnx", False),
        ("u2net_cloth_seg", "U2Net Cloth Segmentation", "Clothing parsing", "u2net_cloth_seg.onnx", False),
        ("isnet-general-use", "IS-Net General", "General purpose", "isnet-general-use.onnx", False),
        ("isnet-anime", "IS-Net Anime", "Anime character segmentation", "isnet-anime.onnx", False),
        ("silueta", "Silueta", "Compact model", "silueta.onnx", False),
    ]
    return jsonify({"models": [
        {"id": model_id, "name": name, "desc": desc, "default": default,
         "source": f"https://github.com/danielgatis/rembg/releases/download/v0.0.0/{filename}"}
        for model_id, name, desc, filename, default in models
    ]})


@rembg_bp.errorhandler(Exception)
def api_exception(error):
    if isinstance(error, HTTPException):
        return jsonify({"ok": False, "error": error.description}), error.code
    current_app.logger.exception("Unhandled RemBG API error")
    return jsonify({"ok": False, "error": f"{type(error).__name__}: {error}"}), 500


@rembg_bp.post("/process")
def process_image():
    t0 = time.time()
    data = request.get_json(silent=True) or {}

    # Image source: file upload, base64 payload, or disk path
    input_pil = None
    orig_name = "image.png"

    if "file" in request.files:
        f = request.files["file"]
        orig_name = f.filename or "image.png"
        try:
            input_pil = Image.open(f.stream).convert("RGBA")
        except Exception:
            return jsonify({"ok": False, "error": "Uploaded file is not a readable image"}), 400
    elif data.get("image"):
        orig_name = data.get("name", "uploaded.png")
        input_pil = base64_to_pil(data["image"])
    elif data.get("path"):
        path = unquote(data["path"])
        root = os.path.realpath(get_input_dir())
        full_path = os.path.realpath(path if os.path.isabs(path) else os.path.join(root, path))
        if os.path.commonpath((root, full_path)) != root:
            return jsonify({"ok": False, "error": "Image path is outside the input directory"}), 400
        if os.path.isfile(full_path):
            orig_name = os.path.basename(full_path)
            try:
                input_pil = Image.open(full_path).convert("RGBA")
            except Exception:
                return jsonify({"ok": False, "error": "File is not a readable image"}), 400

    if input_pil is None:
        return jsonify({"ok": False, "error": "No valid input image provided"}), 400

    orig_width, orig_height = input_pil.size
    model_name = data.get("model", "birefnet-general")
    target_mode = data.get("target_mode", "transparent")  # transparent, color, composite, mask
    bgcolor = data.get("bgcolor", "#0F131C")
    output_format = data.get("format", "png").lower()  # png, webp, mask
    save_output = bool(data.get("save", False))

    threshold = float(data.get("threshold", 0.85)) if data.get("threshold") is not None else 0.85
    erosion = float(data.get("erosion", -0.5)) if data.get("erosion") is not None else -0.5
    feather = float(data.get("feather", 1.5)) if data.get("feather") is not None else 1.5
    despill = bool(data.get("despill", True))

    # Neural Background Removal
    try:
        from rembg import remove
        sess, actual_model = get_model_session(model_name)
        kwargs = {}
        if sess is None:
            detail = _session_errors.get(actual_model)
            raise RuntimeError(f"Could not initialize model '{actual_model}'" + (f": {detail}" if detail else ""))
        kwargs["session"] = sess

        # Pass PIL image to rembg
        cutout = remove(input_pil, **kwargs)
    except Exception as e:
        # Fallback: simple contrast/alpha fallback if model fails
        return jsonify({"ok": False, "error": f"Inference error: {str(e)}"}), 500

    # Post-process matte
    refined = apply_matte_refinements(
        cutout,
        threshold=threshold,
        erosion=erosion,
        feather=feather,
        despill=despill,
    )

    alpha_channel = refined.split()[3]
    mask_img = Image.merge("RGB", (alpha_channel, alpha_channel, alpha_channel))

    # Target Mode Compositing
    if output_format == "mask" or target_mode == "mask":
        final_img = mask_img
    elif target_mode in ("color", "composite"):
        rgb_color = hex_to_rgb(bgcolor)
        bg = Image.new("RGBA", refined.size, rgb_color + (255,))
        final_img = Image.alpha_composite(bg, refined)
    else:  # transparent or default
        final_img = refined

    t_ms = int((time.time() - t0) * 1000)

    # Save to disk if requested. A full output volume must not take the
    # preview down with it: the cutout is still returned, with the save
    # failure reported next to it.
    saved_path = None
    save_error = ""
    if save_output:
        out_dir = get_output_dir()
        stem = os.path.splitext(os.path.basename(orig_name))[0]
        ext = "png" if output_format in ("png", "mask") or target_mode == "transparent" else "webp"
        target_filename = f"{stem}_cutout_{int(time.time() * 1000)}.{ext}"
        saved_path = os.path.join(out_dir, target_filename)
        try:
            final_img.save(saved_path)
        except OSError as exc:
            save_error = f"{type(exc).__name__}: {exc}"
            saved_path = None
            current_app.logger.warning("RemBG could not save %s: %s", target_filename, save_error)

    # Return base64 preview representations
    preview_url = pil_to_base64(final_img, fmt=output_format)
    mask_url = pil_to_base64(mask_img, fmt="PNG")
    orig_url = pil_to_base64(input_pil, fmt="PNG")

    return jsonify({
        "ok": True,
        "width": orig_width,
        "height": orig_height,
        "inference_ms": t_ms,
        "model": actual_model,
        "image": preview_url,
        "mask": mask_url,
        "original": orig_url,
        "saved_path": saved_path,
        "saved_filename": os.path.basename(saved_path) if saved_path else None,
        "save_error": save_error,
    })


@rembg_bp.get("/queue")
def list_queue():
    """List source images available for processing from input directory."""
    items = []
    try:
        in_dir = get_input_dir()
        for f in sorted(os.listdir(in_dir)):
            full = os.path.join(in_dir, f)
            if os.path.isfile(full) and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                stat = os.stat(full)
                items.append({
                    "name": f,
                    "path": f,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                })
    except Exception as exc:
        return jsonify({"items": [], "total": 0, "dir": locals().get("in_dir", ""), "error": f"{type(exc).__name__}: {exc}"}), 500
    return jsonify({"items": items[:50], "total": len(items), "dir": in_dir})


def get_staging_dir():
    """Scratch space for files the browser uploads one at a time.

    A batch used to arrive as one multipart request holding every file. A
    proxy in front of the app (bot protection, body-size limit) rejects that
    with a 403, so the browser now stages the files one request at a time and
    the batch starts from the staged paths.
    """
    path = os.path.join(get_output_dir(), ".staging")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        path = os.path.join(tempfile.gettempdir(), "rembg-staging")
        os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def _clear_staging_dir(keep=()):
    """Drop leftovers from an earlier batch (never the files in use)."""
    keep = {os.path.realpath(p) for p in keep}
    try:
        for name in os.listdir(get_staging_dir()):
            full = os.path.join(get_staging_dir(), name)
            if os.path.realpath(full) in keep:
                continue
            try:
                if os.path.isfile(full):
                    os.remove(full)
            except OSError:
                pass
    except OSError:
        pass


@rembg_bp.post("/batch/stage")
def stage_batch_file():
    """Accept one image for a later batch and return where it was staged."""
    app = current_app._get_current_object()
    name = ""
    image_bytes = None

    if "file" in request.files:
        uploaded = request.files["file"]
        name = secure_filename(uploaded.filename or "")
        image_bytes = uploaded.read()
    else:
        data = request.get_json(silent=True) or {}
        name = secure_filename(data.get("name") or "")
        raw = data.get("image") or ""
        if not raw:
            return jsonify({"ok": False, "error": "no image was sent"}), 400
        try:
            if raw.startswith("data:"):
                raw = raw.split(",", 1)[1] if "," in raw else raw
            image_bytes = base64.b64decode(raw, validate=False)
        except Exception:
            return jsonify({"ok": False, "error": "image is not valid base64"}), 400

    extension = os.path.splitext(name)[1].lower()
    if extension not in IMAGE_EXTENSIONS:
        return jsonify({"ok": False, "error": "unsupported image type"}), 400
    if not image_bytes:
        return jsonify({"ok": False, "error": "the image was empty"}), 400

    safe_name = name or f"image{extension}"
    staged_path = os.path.join(get_staging_dir(), f"{uuid.uuid4().hex}_{safe_name}")
    try:
        with open(staged_path, "wb") as handle:
            handle.write(image_bytes)
    except OSError as exc:
        app.logger.warning("RemBG could not stage %s: %s", safe_name, exc)
        return jsonify({"ok": False, "error": f"Could not stage the file: {exc}"}), 500
    return jsonify({"ok": True, "path": staged_path, "name": safe_name})


@rembg_bp.post("/batch")
def start_batch():
    global _batch_status
    with _batch_lock:
        if _batch_status["running"]:
            return jsonify({"ok": False, "error": "A batch job is already running"}), 409
        app = current_app._get_current_object()

        uploaded_files = request.files.getlist("files")
        if request.mimetype == "multipart/form-data":
            try:
                data = json.loads(request.form.get("options", "{}"))
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "options must contain valid JSON"}), 400
            if not isinstance(data, dict):
                return jsonify({"ok": False, "error": "options must be an object"}), 400
        else:
            data = request.get_json(silent=True) or {}
        items = data.get("items", [])
        if not isinstance(items, list):
            return jsonify({"ok": False, "error": "items must be a list"}), 400
        if not uploaded_files and any(
            not isinstance(item, str) and not (isinstance(item, dict) and item.get("path"))
            for item in items
        ):
            return jsonify({"ok": False, "error": "items must contain image paths"}), 400
        staging_dir = None
        staged_paths = []
        if not uploaded_files:
            staged_paths = [
                item["path"] for item in items
                if isinstance(item, dict) and item.get("path")
            ]
            # Anything left in the staging area belongs to an earlier job.
            _clear_staging_dir(keep=staged_paths)

        def cleanup_staging():
            if not staging_dir:
                return
            try:
                shutil.rmtree(staging_dir)
            except OSError:
                app.logger.exception("Could not remove RemBG batch upload directory %s", staging_dir)

        if uploaded_files:
            staging_dir = tempfile.mkdtemp(prefix="rembg-batch-")
            items = []
            try:
                for index, uploaded_file in enumerate(uploaded_files, start=1):
                    original_name = secure_filename(uploaded_file.filename or "")
                    extension = os.path.splitext(original_name)[1].lower()
                    if extension not in IMAGE_EXTENSIONS:
                        continue
                    safe_name = original_name or f"image{extension}"
                    staged_path = os.path.join(staging_dir, f"{index:04d}_{safe_name}")
                    uploaded_file.save(staged_path)
                    items.append({"path": staged_path, "name": safe_name})
            except Exception:
                cleanup_staging()
                raise
            if not items:
                cleanup_staging()
                return jsonify({"ok": False, "error": "No supported image files were uploaded"}), 400
        model_name = data.get("model", "birefnet-general")
        target_mode = data.get("target_mode", "transparent")
        bgcolor = data.get("bgcolor", "#0F131C")
        output_format = str(data.get("format", "png")).lower()
        try:
            threshold = float(data.get("threshold", 0.85))
            erosion = float(data.get("erosion", -0.5))
            feather = float(data.get("feather", 1.5))
        except (TypeError, ValueError):
            cleanup_staging()
            return jsonify({"ok": False, "error": "threshold, erosion, and feather must be numbers"}), 400
        despill = bool(data.get("despill", True))

        if not items:
            # Auto-populate from input dir
            in_dir = get_input_dir()
            for f in sorted(os.listdir(in_dir)):
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                    items.append(f)

        if not items:
            return jsonify({"ok": False, "error": "No images found for batch"}), 400

        _batch_cancel.clear()
        _batch_status = {
            "running": True,
            "state": "running",
            "total": len(items),
            "processed": 0,
            "errors": 0,
            "current_file": "",
            "items": [],
            "started_at": time.time(),
            "message": "",
        }

        def run_batch():
            global _batch_status
            try:
                from rembg import remove
                sess, actual_model = get_model_session(model_name)
                if sess is None:
                    raise RuntimeError(f"Could not initialize model '{actual_model}': {_session_errors.get(actual_model, 'session unavailable')}")
                out_dir = get_output_dir()
                in_dir = get_input_dir()

                for item in items:
                    if _batch_cancel.is_set():
                        break
                    # A paused batch waits here until it is resumed or stopped.
                    while _batch_pause.is_set() and not _batch_cancel.is_set():
                        _batch_status["state"] = "paused"
                        time.sleep(0.2)
                    if _batch_cancel.is_set():
                        break
                    _batch_status["state"] = "running"
                    uploaded_name = item.get("name") if isinstance(item, dict) else None
                    item_path = item["path"] if isinstance(item, dict) else item
                    # Dict items are staged uploads: either this request's own
                    # temp directory or the persistent staging area the browser
                    # filled one file at a time.
                    source_root = (staging_dir or get_staging_dir()) if isinstance(item, dict) else in_dir
                    source_root = os.path.realpath(source_root)
                    full_in = os.path.realpath(item_path if os.path.isabs(item_path) else os.path.join(source_root, item_path))
                    if os.path.commonpath((source_root, full_in)) != source_root:
                        _batch_status["errors"] += 1
                        _batch_status["items"].append({"file": os.path.basename(str(item)), "status": "error", "error": "Path is outside the allowed directory"})
                        continue
                    fname = uploaded_name or os.path.basename(full_in)
                    _batch_status["current_file"] = fname

                    try:
                        with Image.open(full_in) as raw:
                            img = raw.convert("RGBA")
                        cutout = remove(img, session=sess)
                        refined = apply_matte_refinements(cutout, threshold, erosion, feather, despill)

                        if target_mode == "mask" or output_format == "mask":
                            alpha = refined.getchannel("A")
                            res = Image.merge("RGB", (alpha, alpha, alpha))
                        elif target_mode in ("color", "composite"):
                            bg = Image.new("RGBA", refined.size, hex_to_rgb(bgcolor) + (255,))
                            res = Image.alpha_composite(bg, refined)
                        else:
                            res = refined

                        stem = os.path.splitext(fname)[0]
                        out_ext = "webp" if output_format == "webp" and target_mode != "transparent" else "png"
                        out_path = os.path.join(out_dir, f"{stem}_cutout_{int(time.time() * 1000)}.{out_ext}")
                        res.save(out_path, format=out_ext.upper())

                        _batch_status["processed"] += 1
                        _batch_status["items"].append({
                            "file": fname,
                            "status": "done",
                            "output": out_path,
                            # The staged original, while the job still owns it:
                            # a studio page re-opened mid-batch can show the
                            # before/after wipe again.
                            "source": f"/rembg/api/batch/staged/{quote(os.path.basename(full_in))}",
                            # A URL the studio can show straight away, so the
                            # batch results are visible instead of only on disk.
                            "url": f"/rembg/api/file/{quote(os.path.basename(out_path))}",
                        })
                    except Exception as e:
                        _batch_status["errors"] += 1
                        _batch_status["items"].append({"file": fname, "status": "error", "error": str(e)})
            except Exception as e:
                _batch_status["errors"] += max(0, len(items) - len(_batch_status["items"]))
                _batch_status["items"].append({"file": "batch", "status": "error", "error": str(e)})
                _batch_status["message"] = str(e)
                _batch_status["state"] = "error"
            finally:
                cancelled = _batch_cancel.is_set()
                _batch_status["running"] = False
                _batch_status["current_file"] = ""
                if _batch_status["state"] == "running":
                    processed = _batch_status["processed"]
                    errors = _batch_status["errors"]
                    total = len(items)
                    if cancelled:
                        _batch_status["state"] = "cancelled"
                    elif errors and not processed:
                        # Every image failed. Reporting "complete" here made a
                        # dead batch look finished, so the state says it failed
                        # and the per-item errors carry the reason.
                        _batch_status["state"] = "error"
                        _batch_status["message"] = f"All {errors} image{'' if errors == 1 else 's'} failed"
                    elif errors:
                        _batch_status["message"] = f"{errors} of {total} images failed"
                    else:
                        _batch_status["message"] = f"Saved {processed} cutout{'' if processed == 1 else 's'}"
                cleanup_staging()
                # The staged uploads were this job's own scratch space.
                for path in staged_paths:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                if staged_paths:
                    _clear_staging_dir()

        threading.Thread(target=run_batch, daemon=True).start()
        return jsonify({"ok": True, "total": len(items)})


@rembg_bp.get("/batch/staged/<path:name>")
def staged_batch_file(name):
    """Serve a file the browser staged for the running batch."""
    clean = os.path.basename(unquote(name or "").strip())
    if not clean:
        return "File not found", 404
    root = os.path.realpath(get_staging_dir())
    target = os.path.realpath(os.path.join(root, clean))
    if os.path.commonpath((root, target)) != root or not os.path.isfile(target):
        return "File not found", 404
    mime, _ = mimetypes.guess_type(target)
    return send_file(target, mimetype=mime or "application/octet-stream", conditional=True)


@rembg_bp.get("/batch/status")
def batch_status():
    """Batch state for the studio page and the core plugin-task float."""
    payload = dict(_batch_status)
    processed = payload.get("processed", 0)
    total = payload.get("total", 0)
    payload["active"] = bool(payload.get("running"))
    payload["current"] = processed
    # While running the detail is the file in flight; once it is over, name the
    # first failure so the core's task float shows why nothing was saved.
    detail = payload.get("current_file") or ""
    if not detail:
        for item in payload.get("items") or []:
            if item.get("status") == "error":
                detail = f"{item.get('file', 'image')}: {item.get('error', 'failed')}"
                break
    payload["detail"] = detail
    if not payload.get("message"):
        if payload.get("errors"):
            payload["message"] = f"{payload.get('errors')} of {total} images failed"
        else:
            payload["message"] = f"{processed} of {total} images" if total else ""
    return jsonify(payload)


@rembg_bp.post("/batch/pause")
def pause_batch():
    """Pause or resume a running batch (the task float's pause button)."""
    with _batch_lock:
        if not _batch_status["running"]:
            return jsonify({"ok": True, "state": _batch_status["state"], "paused": False})
        if _batch_pause.is_set():
            _batch_pause.clear()
            _batch_status["state"] = "running"
            return jsonify({"ok": True, "state": "running", "paused": False})
        _batch_pause.set()
        _batch_status["state"] = "paused"
        return jsonify({"ok": True, "state": "paused", "paused": True})


@rembg_bp.post("/batch/cancel")
def cancel_batch():
    _batch_cancel.set()
    _batch_pause.clear()
    return jsonify({"ok": True})


@rembg_bp.get("/outputs")
def list_outputs():
    out_dir = get_output_dir()
    files = []
    try:
        for f in sorted(os.listdir(out_dir), reverse=True):
            full = os.path.join(out_dir, f)
            if os.path.isfile(full) and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                stat = os.stat(full)
                files.append({
                    "name": f,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "url": f"/rembg/api/file/{f}",
                })
    except Exception:
        pass
    return jsonify({"outputs": files[:50], "total": len(files)})


@rembg_bp.get("/file/<path:filename>")
def stream_file(filename):
    clean = unquote(filename).strip("/\\")
    out_root = os.path.realpath(get_output_dir())
    in_root = os.path.realpath(get_input_dir())
    out_path = os.path.realpath(os.path.join(out_root, clean))
    in_path = os.path.realpath(os.path.join(in_root, clean))
    if not (os.path.commonpath((out_root, out_path)) == out_root or
            os.path.commonpath((in_root, in_path)) == in_root):
        return "File not found", 404

    target = out_path if os.path.isfile(out_path) else in_path
    if not os.path.isfile(target):
        return "File not found", 404

    mime, _ = mimetypes.guess_type(target)
    return send_file(target, mimetype=mime or "image/png")
