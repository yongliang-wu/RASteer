"""COCO-3k eval: CLIP caption score + FID_real (vs matched real images) + FID* (vs orig gen).
Usage: python coco3k_eval.py <cond1> <cond2> ...   ('orig' must be among generated dirs)
Writes results/coco3k/metrics.json"""
import os, sys, json, glob
import numpy as np, pandas as pd, torch

ROOT = os.environ.get("COCO3K_ROOT", "results/coco3k")
REAL = "cache/coco_real/images3k"

def build_real_subset():
    df = pd.read_csv("data/coco/coco_30k.csv")
    df = df[~df["prompt"].str.lower().str.contains("horse")].head(3000)
    os.makedirs(REAL, exist_ok=True)
    missing = []
    for cid in df["coco_id"]:
        dst = f"{REAL}/{cid:012d}.jpg"
        if os.path.exists(dst): continue
        src = f"cache/coco_real/val2014/COCO_val2014_{cid:012d}.jpg"
        if os.path.exists(src): os.link(src, dst)
        else: missing.append(int(cid))
    if missing:
        import urllib.request
        for cid in missing:
            for split in ("train2014", "val2014"):
                try:
                    urllib.request.urlretrieve(
                        f"http://images.cocodataset.org/{split}/COCO_{split}_{cid:012d}.jpg",
                        f"{REAL}/{cid:012d}.jpg")
                    break
                except Exception: continue
    n = len(glob.glob(f"{REAL}/*.jpg"))
    print(f"real subset: {n}/3000")
    return n

def feats(path, cache):
    if os.path.exists(cache): return np.load(cache)
    from cleanfid.fid import get_folder_features, build_feature_extractor
    global _M
    if "_M" not in globals(): _M = build_feature_extractor("clean", "cuda")
    f = np.asarray(get_folder_features(path, _M, num_workers=8, batch_size=64,
        device=torch.device("cuda"), mode="clean", description="", verbose=False), dtype=np.float64)
    np.save(cache, f); return f

def ms(X):
    X = torch.from_numpy(X).double(); mu = X.mean(0); Xc = X - mu
    return mu, Xc.T @ Xc / (X.shape[0] - 1)

def sqmat(s):
    w, V = torch.linalg.eigh(s); return (V * torch.clamp(w, min=0).sqrt()) @ V.T

def fid(a, b):
    (m1,s1),(m2,s2) = a,b
    sv = torch.linalg.svdvals(sqmat(s1) @ sqmat(s2))
    return float((m1-m2) @ (m1-m2) + torch.trace(s1) + torch.trace(s2) - 2*sv.sum())

def clip_score(folder):
    import clip
    from PIL import Image
    model, prep = clip.load("ViT-B/32", device="cuda"); model.eval()
    items = []
    for d in sorted(glob.glob(os.path.join(folder, "*"))):
        if os.path.isdir(d):
            cap = os.path.basename(d)
            items += [(p, cap) for p in glob.glob(os.path.join(d, "*.png"))]
    sims = []
    with torch.no_grad():
        for i in range(0, len(items), 64):
            b = items[i:i+64]
            x = torch.stack([prep(Image.open(p).convert("RGB")) for p,_ in b]).to("cuda")
            t = clip.tokenize([c for _,c in b], truncate=True).to("cuda")
            fi = model.encode_image(x).float(); fi /= fi.norm(dim=-1, keepdim=True)
            ft = model.encode_text(t).float(); ft /= ft.norm(dim=-1, keepdim=True)
            sims += (fi*ft).sum(-1).tolist()
    return 100.0 * float(np.mean(sims)), len(sims)

def main(conds):
    build_real_subset()
    M, out = {}, {}
    M["real"] = ms(feats(REAL, f"{ROOT}/feats_real.npy"))
    M["orig"] = ms(feats(f"{ROOT}/orig", f"{ROOT}/feats_orig.npy"))
    for c in conds:
        if c not in M:
            M[c] = ms(feats(f"{ROOT}/{c}", f"{ROOT}/feats_{c}.npy"))
    print(f"[sanity] FID(orig,orig)={fid(M['orig'],M['orig']):.4f}")
    for c in conds:
        cs, n = clip_score(f"{ROOT}/{c}")
        out[c] = {"n": n, "CLIP": round(cs,2),
                  "FID_real": round(fid(M["real"], M[c]),2),
                  "FID_star": 0.0 if c=="orig" else round(fid(M["orig"], M[c]),2)}
        print(c, json.dumps(out[c]))
    import fcntl
    with open(f"{ROOT}/metrics.json.lock", "w") as _lk:
        fcntl.flock(_lk, fcntl.LOCK_EX)
        try:
            prev = json.load(open(f"{ROOT}/metrics.json"))
        except Exception:
            prev = {}
        prev.update(out)
        json.dump(prev, open(f"{ROOT}/metrics.json", "w"), indent=1)
    print("WROTE", f"{ROOT}/metrics.json")

if __name__ == "__main__":
    main(sys.argv[1:] or ["orig"])
