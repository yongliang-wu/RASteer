"""Table-3 metrics: per (setting, method, concept) CS / CA / FID.

CS  = 100 x mean CLIP(ViT-B/32) cosine(image, concept name).
CA  = zero-shot accuracy: fraction of images whose CLIP-best label among the
      side's 4 concept names is the generated concept.
FID = clean-fid Inception features + torch-linalg Frechet distance against the
      unedited (orig) generations of the same concept (same prompts, seed 42).

Layout: results/table3/<method>/<setting>/<concept>/<prompt>/42-*.png
        results/table3/orig/<concept>/...
Writes results/table3/metrics.json
Usage: python table3_eval.py <method1> <method2> ...
"""
import glob
import json
import os
import sys

import numpy as np
import torch

# Base-model aware: set T3_ROOT=results/table3_sd21 (or _sdxl) for the
# cross-model runs; defaults to the SD-1.4 Table-3 tree.
ROOT = os.environ.get("T3_ROOT", "results/table3")
CONCEPTS = {
    "snoopy": "Snoopy", "mickey": "Mickey Mouse", "spongebob": "Spongebob",
    "pikachu": "Pikachu", "vangogh": "Van Gogh", "picasso": "Picasso",
    "rembrandt": "Rembrandt", "hokusai": "Hokusai",
}
INSTANCE = ["snoopy", "mickey", "spongebob", "pikachu"]
STYLE = ["vangogh", "picasso", "rembrandt", "hokusai"]
SETTINGS = {
    "e_snoopy": INSTANCE, "e_snoopy_mickey": INSTANCE,
    "e_snoopy_mickey_spongebob": INSTANCE,
    "e_vangogh": STYLE, "e_picasso": STYLE, "e_rembrandt": STYLE,
}

_clip = {}
def clip_model():
    if not _clip:
        import clip
        m, p = clip.load("ViT-B/32", device="cuda")
        m.eval()
        _clip.update(model=m, prep=p)
    return _clip["model"], _clip["prep"]


def imgs_of(d):
    return sorted(glob.glob(os.path.join(d, "**", "*.png"), recursive=True))


@torch.no_grad()
def cs_ca(d, concept_key, side):
    import clip
    from PIL import Image
    model, prep = clip_model()
    labels = [CONCEPTS[k] for k in side]
    tgt = labels.index(CONCEPTS[concept_key])
    txt = clip.tokenize([f"an image of {l}" for l in labels]).to("cuda")
    ft = model.encode_text(txt).float()
    ft /= ft.norm(dim=-1, keepdim=True)
    paths = imgs_of(d)
    sims, correct = [], 0
    B = 64
    for i in range(0, len(paths), B):
        x = torch.stack([prep(Image.open(p).convert("RGB")) for p in paths[i:i + B]]).to("cuda")
        fi = model.encode_image(x).float()
        fi /= fi.norm(dim=-1, keepdim=True)
        s = fi @ ft.T                       # [B, 4]
        sims += s[:, tgt].tolist()
        correct += int((s.argmax(-1) == tgt).sum())
    n = len(paths)
    return round(100 * float(np.mean(sims)), 2), round(100 * correct / max(1, n), 2), n


_inc = {}
def feats(d, cache):
    if os.path.exists(cache):
        return np.load(cache)
    from cleanfid.fid import build_feature_extractor, get_folder_features
    if not _inc:
        _inc["m"] = build_feature_extractor("clean", "cuda")
    f = np.asarray(get_folder_features(d, _inc["m"], num_workers=8, batch_size=64,
                   device=torch.device("cuda"), mode="clean", description="",
                   verbose=False), dtype=np.float64)
    np.save(cache, f)
    return f


def fid(a, b):
    def ms(X):
        X = torch.from_numpy(X).double()
        mu = X.mean(0); Xc = X - mu
        return mu, Xc.T @ Xc / (X.shape[0] - 1)
    def sq(s):
        w, V = torch.linalg.eigh(s)
        return (V * torch.clamp(w, min=0).sqrt()) @ V.T
    (m1, s1), (m2, s2) = ms(a), ms(b)
    sv = torch.linalg.svdvals(sq(s1) @ sq(s2))
    return float((m1 - m2) @ (m1 - m2) + torch.trace(s1) + torch.trace(s2) - 2 * sv.sum())


def main(methods):
    os.makedirs(f"{ROOT}/_feats", exist_ok=True)
    metrics_path = f"{ROOT}/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            out = json.load(f)
    else:
        out = {}

    out["orig"] = {}
    for c in CONCEPTS:
        d = f"{ROOT}/orig/{c}"
        side = INSTANCE if c in INSTANCE else STYLE
        cs, ca, n = cs_ca(d, c, side)
        out["orig"][c] = {"CS": cs, "CA": ca, "n": n}
        print("orig", c, out["orig"][c], flush=True)
    for m in methods:
        out[m] = {}
        for s, side in SETTINGS.items():
            out[m][s] = {}
            for c in side:
                d = f"{ROOT}/{m}/{s}/{c}"
                if not os.path.isdir(d):
                    out[m][s][c] = None
                    continue
                cs, ca, n = cs_ca(d, c, side)
                f_gen = feats(d, f"{ROOT}/_feats/{m}_{s}_{c}.npy")
                f_org = feats(f"{ROOT}/orig/{c}", f"{ROOT}/_feats/orig_{c}.npy")
                out[m][s][c] = {"CS": cs, "CA": ca,
                                "FID": round(fid(f_org, f_gen), 2), "n": n}
                print(m, s, c, out[m][s][c], flush=True)
    # Re-read under an exclusive lock right before writing: two evals that
    # finish together must not drop each other's methods (lost update).
    import fcntl
    with open(metrics_path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        merged = {}
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                merged = json.load(f)
        for k, v in out.items():
            if k in methods or k not in merged:
                merged[k] = v
        tmp_path = metrics_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(merged, f, indent=1)
        os.replace(tmp_path, metrics_path)
    print("WROTE", metrics_path)


if __name__ == "__main__":
    main(sys.argv[1:])
