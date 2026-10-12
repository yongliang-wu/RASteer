"""COCO-3k extra metrics: paired LPIPS vs orig + KID (unbiased, from cached features).
Usage: python coco3k_extra.py <cond1> <cond2> ...   (orig excluded automatically from LPIPS)
Reads feats_*.npy caches from results/coco3k (created by coco3k_eval.py).
Writes results/coco3k/metrics_extra.json
"""
import os, sys, glob, json
import numpy as np

ROOT = os.environ.get("COCO3K_ROOT", "results/coco3k")
REAL_FEAT = f"{ROOT}/feats_real.npy"


def paired_lpips(cond):
    import torch, lpips
    from PIL import Image
    dev = "cuda"
    fn = lpips.LPIPS(net="alex", verbose=False).to(dev)

    def load(p, size=256):
        im = Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)

    rel = lambda d: {os.path.relpath(p, d): p for p in glob.glob(os.path.join(d, "**", "*.png"), recursive=True)}
    a, b = rel(f"{ROOT}/orig"), rel(f"{ROOT}/{cond}")
    keys = sorted(set(a) & set(b))
    vals = []
    B = 32
    import torch as T
    with T.no_grad():
        for i in range(0, len(keys), B):
            ks = keys[i:i + B]
            x0 = T.stack([load(a[k]) for k in ks]).to(dev)
            x1 = T.stack([load(b[k]) for k in ks]).to(dev)
            vals.extend(fn(x0, x1).flatten().tolist())
    v = np.asarray(vals)
    return {"n_pairs": len(v), "lpips_mean": round(float(v.mean()), 4),
            "lpips_sem": round(float(v.std(ddof=1) / np.sqrt(len(v))), 5)}


def kid(featA, featB, n_subsets=100, subset_size=1000, seed=0):
    """Unbiased KID (MMD^2, cubic polynomial kernel), cleanfid-style subset averaging."""
    rng = np.random.RandomState(seed)
    d = featA.shape[1]
    m = min(subset_size, featA.shape[0], featB.shape[0])
    scores = []
    for _ in range(n_subsets):
        x = featA[rng.choice(featA.shape[0], m, replace=False)]
        y = featB[rng.choice(featB.shape[0], m, replace=False)]
        Kxx = (x @ x.T / d + 1) ** 3
        Kyy = (y @ y.T / d + 1) ** 3
        Kxy = (x @ y.T / d + 1) ** 3
        np.fill_diagonal(Kxx, 0); np.fill_diagonal(Kyy, 0)
        mmd = Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (m * (m - 1)) - 2 * Kxy.mean()
        scores.append(mmd)
    return float(np.mean(scores))


def main(conds):
    real = np.load(REAL_FEAT)
    orig = np.load(f"{ROOT}/feats_orig.npy")
    out = {}
    for c in conds:
        feats = np.load(f"{ROOT}/feats_{c}.npy")
        row = {"KIDx1000_real": round(1000 * kid(real, feats), 3),
               "KIDx1000_drift": 0.0 if c == "orig" else round(1000 * kid(orig, feats), 3)}
        if c != "orig":
            row.update(paired_lpips(c))
        out[c] = row
        print(c, json.dumps(row), flush=True)
    import fcntl
    with open(f"{ROOT}/metrics_extra.json.lock", "w") as _lk:
        fcntl.flock(_lk, fcntl.LOCK_EX)
        try:
            prev = json.load(open(f"{ROOT}/metrics_extra.json"))
        except Exception:
            prev = {}
        prev.update(out)
        json.dump(prev, open(f"{ROOT}/metrics_extra.json", "w"), indent=1)
    print("WROTE", f"{ROOT}/metrics_extra.json")


if __name__ == "__main__":
    main(sys.argv[1:])
