"""SDCodex-RemBG backend — high-precision neural background removal and replacement."""

import base64
import io
import json
import mimetypes
import os
import re
import threading
import time
from urllib.parse import unquote
from flask import Blueprint, jsonify, request, send_file
from PIL import Image, ImageFilter, ImageOps
import numpy as np

rembg_bp = Blueprint("rembg_bp", __name__, url_prefix="/rembg/api")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".jfif"
}

_sessions = {}
_session_lock = threading.Lock()

# Batch state
_batch_lock = threading.Lock()
_batch_cancel = threading.Event()
_batch_status = {
    "running": False,
    "total": 0,
    "processed": 0,
    "errors": 0,
    "current_file": "",
    "items": [],
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
    target_name = alias_map.get(name, "u2net")

    with _session_lock:
        if target_name in _sessions:
            return _sessions[target_name], target_name
        try:
            from rembg import new_session
            sess = new_session(target_name)
            _sessions[target_name] = sess
            return sess, target_name
        except Exception:
            # Fallback to default u2net session if specific model fails (e.g. offline)
            if target_name != "u2net":
                try:
                    from rembg import new_session
                    sess = new_session("u2net")
                    _sessions["u2net"] = sess
                    return sess, "u2net"
                except Exception:
                    pass
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

    return jsonify({
        "engine": "BiRefNet / RemBG Neural Matting",
        "device": device_name,
        "cuda": cuda_avail,
        "active_models": list(_sessions.keys()) or ["birefnet-general (ready)"],
        "input_dir": get_input_dir(),
        "output_dir": get_output_dir(),
    })


@rembg_bp.get("/models")
def list_models():
    return jsonify({
        "models": [
            {
                "id": "birefnet-general",
                "name": "BiRefNet General",
                "desc": "Recommended for complex limbs & hair (Bilateral Reference)",
                "default": True,
            },
            {
                "id": "birefnet-portrait",
                "name": "BiRefNet Portrait",
                "desc": "Fine strand alpha matting tuned for portraits",
            },
            {
                "id": "u2net",
                "name": "U2Net Ultra Fast",
                "desc": "Low latency general salient object detection",
            },
            {
                "id": "isnet-general-use",
                "name": "IS-Net / DIS",
                "desc": "High edge sharpness for fine mechanical & structure edges",
            },
            {
                "id": "silueta",
                "name": "Silueta (Compact)",
                "desc": "Lightweight model for low memory / fast inference",
            },
        ]
    })


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
        input_pil = Image.open(f.stream).convert("RGBA")
    elif data.get("image"):
        orig_name = data.get("name", "uploaded.png")
        input_pil = base64_to_pil(data["image"])
    elif data.get("path"):
        path = unquote(data["path"])
        if os.path.isabs(path) and os.path.isfile(path):
            full_path = path
        else:
            full_path = os.path.join(get_input_dir(), path)
        if os.path.isfile(full_path):
            orig_name = os.path.basename(full_path)
            input_pil = Image.open(full_path).convert("RGBA")

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
        if sess:
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
    if target_mode == "color":
        rgb_color = hex_to_rgb(bgcolor)
        bg = Image.new("RGBA", refined.size, rgb_color + (255,))
        final_img = Image.alpha_composite(bg, refined)
    elif target_mode == "mask":
        final_img = mask_img
    else:  # transparent or default
        final_img = refined

    t_ms = int((time.time() - t0) * 1000)

    # Save to disk if requested
    saved_path = None
    if save_output:
        out_dir = get_output_dir()
        stem = os.path.splitext(orig_name)[0]
        ext = "png" if output_format == "png" or target_mode == "transparent" else "webp"
        target_filename = f"{stem}_cutout_{int(time.time())}.{ext}"
        saved_path = os.path.join(out_dir, target_filename)
        final_img.save(saved_path)

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
    })


@rembg_bp.get("/queue")
def list_queue():
    """List source images available for processing from input directory."""
    in_dir = get_input_dir()
    items = []
    try:
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
    except Exception:
        pass
    return jsonify({"items": items[:50], "total": len(items), "dir": in_dir})


@rembg_bp.post("/batch")
def start_batch():
    global _batch_status
    with _batch_lock:
        if _batch_status["running"]:
            return jsonify({"ok": False, "error": "A batch job is already running"}), 409

        data = request.get_json(silent=True) or {}
        items = data.get("items", [])
        model_name = data.get("model", "birefnet-general")
        target_mode = data.get("target_mode", "transparent")
        bgcolor = data.get("bgcolor", "#0F131C")

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
            "total": len(items),
            "processed": 0,
            "errors": 0,
            "current_file": "",
            "items": [],
        }

        def run_batch():
            global _batch_status
            from rembg import remove
            sess, _ = get_model_session(model_name)
            out_dir = get_output_dir()
            in_dir = get_input_dir()

            for idx, item in enumerate(items):
                if _batch_cancel.is_set():
                    break
                full_in = item if os.path.isabs(item) else os.path.join(in_dir, item)
                fname = os.path.basename(full_in)
                _batch_status["current_file"] = fname

                try:
                    with Image.open(full_in) as raw:
                        img = raw.convert("RGBA")
                    cutout = remove(img, session=sess) if sess else remove(img)
                    refined = apply_matte_refinements(cutout)

                    if target_mode == "color":
                        bg = Image.new("RGBA", refined.size, hex_to_rgb(bgcolor) + (255,))
                        res = Image.alpha_composite(bg, refined)
                    else:
                        res = refined

                    stem = os.path.splitext(fname)[0]
                    out_path = os.path.join(out_dir, f"{stem}_cutout.png")
                    res.save(out_path)

                    _batch_status["processed"] += 1
                    _batch_status["items"].append({"file": fname, "status": "done", "output": out_path})
                except Exception as e:
                    _batch_status["errors"] += 1
                    _batch_status["items"].append({"file": fname, "status": "error", "error": str(e)})

            _batch_status["running"] = False
            _batch_status["current_file"] = ""

        threading.Thread(target=run_batch, daemon=True).start()
        return jsonify({"ok": True, "total": len(items)})


@rembg_bp.get("/batch/status")
def batch_status():
    return jsonify(_batch_status)


@rembg_bp.post("/batch/cancel")
def cancel_batch():
    _batch_cancel.set()
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
    out_path = os.path.join(get_output_dir(), clean)
    in_path = os.path.join(get_input_dir(), clean)

    target = out_path if os.path.isfile(out_path) else in_path
    if not os.path.isfile(target):
        return "File not found", 404

    mime, _ = mimetypes.guess_type(target)
    return send_file(target, mimetype=mime or "image/png")
