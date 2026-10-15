"""SDCodex-RemBG entrypoint (Stitch UI + slim JSON API)."""

import os
import sys

plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)


def init_plugin(app, db, plugin_info=None):
    """Register the rembg_bp blueprint."""
    from rembg_backend import rembg_bp

    if "rembg_bp" not in app.blueprints:
        app.register_blueprint(rembg_bp)
    return True
