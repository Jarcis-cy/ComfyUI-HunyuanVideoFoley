import os
import sys
import warnings

import torch


def main():
    # ensure repo root on path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # 1) import
    warnings.filterwarnings("default")
    mod = __import__(
        "custom_nodes.ComfyUI-HunyuanVideoFoley.vendor.models.synchformer.synchformer",
        fromlist=["Synchformer"],
    )
    Synchformer = getattr(mod, "Synchformer")

    # 2) build model (no ckpt)
    model = Synchformer()

    # 3) build dummy inputs
    vis = torch.randn(1, 1, 8, 3, 224, 224)  # (B,S,Tv,C,H,W)
    aud = torch.randn(1, 1, 1, 128, 66)      # (B,S,_,Fa,Ta)

    # 4) forward visual
    vfeats = model.extract_vfeats(vis)
    assert isinstance(vfeats, torch.Tensor), "vfeats must be a tensor"
    assert vfeats.dim() == 4 and vfeats.shape[0] == 1 and vfeats.shape[1] == 1, f"bad vfeats shape: {vfeats.shape}"
    print("vfeats.shape:", tuple(vfeats.shape))

    # 5) forward audio
    afeats = model.extract_afeats(aud)
    assert isinstance(afeats, torch.Tensor), "afeats must be a tensor"
    assert afeats.dim() == 4 and afeats.shape[0] == 1 and afeats.shape[1] == 1, f"bad afeats shape: {afeats.shape}"
    print("afeats.shape:", tuple(afeats.shape))

    # 6) value sanity (range doesn't matter for random init / placeholder, just finite)
    assert torch.isfinite(vfeats).all(), "vfeats contains non-finite values"
    assert torch.isfinite(afeats).all(), "afeats contains non-finite values"

    # 7) optional compare_v_a (requires einops)
    try:
        if hasattr(model, "compare_v_a"):
            # project to flattened time for both streams according to Synchformer.compare_v_a expectations
            B, S, tv, D = vfeats.shape
            B, S, ta, D = afeats.shape
            vflat = vfeats.view(B, S * tv, D)
            aflat = afeats.view(B, S * ta, D)
            logits = model.compare_v_a(vfeats, afeats)
            assert isinstance(logits, torch.Tensor), "compare_v_a must return a tensor"
            print("compare_v_a logits shape:", tuple(logits.shape))
    except Exception as e:
        print("[warning] compare_v_a not executed:", e)

    print("All checks passed.")


if __name__ == "__main__":
    main()

