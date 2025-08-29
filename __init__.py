from __future__ import annotations

# Lightweight init to avoid importing heavy deps at ComfyUI startup
# Nodes are implemented in nodes_hunyuan_foley.py with delayed imports

from .nodes_hunyuan_foley import HunyuanVideoFoleyGenerateAudio, VideoAudioMerger

NODE_CLASS_MAPPINGS = {
    "HunyuanVideoFoleyGenerateAudio": HunyuanVideoFoleyGenerateAudio,
    "VideoAudioMerger": VideoAudioMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanVideoFoleyGenerateAudio": "HunyuanVideo-Foley Generate Audio",
    "VideoAudioMerger": "Video Audio Merger",
}

# Optional: expose a web directory if needed in the future
# WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

