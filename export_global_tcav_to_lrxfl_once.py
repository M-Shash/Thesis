# export_global_tcav_to_lrxfl_once.py
# Export LR-XFL artifacts from GLOBAL TCAVs exactly once (no duplication).
# Outputs:
#   {shared_dir}/concepts/concept_names.npy
#   {shared_dir}/tables/imagenet.csv   (columns: feature000..feature{F-1},label)
#
# If multiple callers run this concurrently, only one becomes "leader" and writes;
# others detect work is done and return immediately.

from __future__ import annotations
import os
from typing import Dict, Tuple, List, Iterable, Callable
import numpy as np
import pandas as pd

# =========================
# Single-writer primitives
# =========================
def try_become_leader(shared_dir: str) -> bool:
    """Atomically try to become the writer. Returns True if we should write, else False."""
    mdir = os.path.join(shared_dir, "_markers")
    os.makedirs(mdir, exist_ok=True)
    leader_path = os.path.join(mdir, "export_leader.lock")
    # Atomic create; fails if exists
    try:
        fd = os.open(leader_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False

def mark_done(shared_dir: str) -> None:
    mdir = os.path.join(shared_dir, "_markers")
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "export_done.ok"), "w") as f:
        f.write("done\n")

def is_done(shared_dir: str) -> bool:
    return os.path.exists(os.path.join(shared_dir, "_markers", "export_done.ok"))

# ================================
# Global concept dictionary (C×K)
# ================================
def build_global_concept_order(
    label_to_name: Dict[int, str],
    clusters_per_class: int,
) -> Tuple[List[Tuple[int, int]], List[str]]:
    all_keys: List[Tuple[int, int]] = []
    names: List[str] = []
    for cid in sorted(label_to_name.keys()):
        cname = label_to_name[cid]
        for k in range(int(clusters_per_class)):
            all_keys.append((cid, k))
            names.append(f"{cname}-C{k:02d}")
    return all_keys, names

def ensure_concepts_file(shared_dir: str, concept_names: List[str]) -> str:
    cdir = os.path.join(shared_dir, "concepts")
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, "concept_names.npy")
    if not os.path.exists(path):
        np.save(path, np.array(concept_names, dtype=object))
    else:
        existing = np.load(path, allow_pickle=True)
        if len(existing) != len(concept_names) or any(a != b for a, b in zip(existing, concept_names)):
            raise SystemExit(
                "[ERROR] concept_names.npy mismatch. Keep class set and K consistent."
            )
    return path

# =========================================
# TCAV>0 binarization for GLOBAL evaluation
# =========================================
def tcav_bits_for_image(
    pil_img,
    cavs: Dict[Tuple[int, int], np.ndarray],                 # {(class_id, cluster_id): cav_vec}
    get_grad_vec: Callable[[object, int], np.ndarray],       # (pil_img, inception_class_idx) -> grad vec
    class_to_inception_idx: Dict[int, int],                  # {class_id: inception_idx}
    all_keys: List[Tuple[int, int]],                         # global concept order
) -> np.ndarray:
    unique_cids = sorted({cid for cid, _ in all_keys})
    grads_per_cid: Dict[int, np.ndarray | None] = {}
    for cid in unique_cids:
        inc_idx = class_to_inception_idx.get(cid)
        if inc_idx is None:
            grads_per_cid[cid] = None
            continue
        g = get_grad_vec(pil_img, inc_idx)
        g = g / (np.linalg.norm(g) + 1e-12)
        grads_per_cid[cid] = g

    bits = np.zeros((len(all_keys),), dtype=np.float32)
    for j, (cid, kid) in enumerate(all_keys):
        g = grads_per_cid.get(cid)
        cav = cavs.get((cid, kid))
        if g is None or cav is None:
            bits[j] = 0.0
            continue
        cav = cav / (np.linalg.norm(cav) + 1e-12)
        bits[j] = 1.0 if float(np.dot(g, cav)) > 0.0 else 0.0
    return bits

# ===================================
# Export once (no duplicates, no append)
# ===================================
def export_global_tcav_binary_once(
    *,
    shared_dir: str,
    label_to_name: Dict[int, str],                 # {class_id:int -> class_name:str}
    clusters_per_class: int,                       # K (uniform)
    global_cavs: Dict[str, List[float]] | Dict[Tuple[int,int], np.ndarray],
    class_to_inception_idx_namekey: Dict[str, int],   # {class_name:str -> inception_idx}
    get_grad_vec_G: Callable[[object, int], np.ndarray],  # global gradient fn
    image_iter_G: Iterable[Tuple[object, int]],     # yields (PIL.Image, label_id)
    overwrite: bool = False,
) -> None:
    """
    Write LR-XFL inputs *once* for a shared test set:
      - If files already exist and overwrite=False, do nothing.
      - Otherwise, attempt to become leader and write them.

    The CSV is written in *write mode* (not append) to avoid duplicate rows.
    """
    # If already done and not overwriting -> no-op
    if (not overwrite) and is_done(shared_dir):
        print("[SKIP] export already completed.")
        return

    # Only one writer proceeds
    if not try_become_leader(shared_dir):
        # Another process will write; if you want to block until done, you can spin here.
        print("[SKIP] another process is exporting; skipping.")
        return

    # Normalize global_cavs keys to (class_id, cluster_id)
    cavs: Dict[Tuple[int, int], np.ndarray] = {}
    if len(global_cavs) > 0:
        sample_key = next(iter(global_cavs.keys()))
        if isinstance(sample_key, tuple):
            for (cid, kid), vec in global_cavs.items():  # type: ignore
                cavs[(int(cid), int(kid))] = np.array(vec, dtype=np.float32)
        else:
            for key, vec in global_cavs.items():  # type: ignore
                cl, cid = key.split("_")
                cavs[(int(cl), int(cid))] = np.array(vec, dtype=np.float32)

    # Global concept order & names
    all_keys, names = build_global_concept_order(label_to_name, clusters_per_class)
    ensure_concepts_file(shared_dir, names)

    # Build id->inception_idx map
    name_to_id = {v: k for k, v in label_to_name.items()}
    class_to_inception_idx: Dict[int, int] = {}
    for name, idx in class_to_inception_idx_namekey.items():
        if name in name_to_id:
            class_to_inception_idx[name_to_id[name]] = int(idx)

    # Compute all rows (TCAV>0) for the shared test set
    rows: List[np.ndarray] = []
    lbls: List[int] = []
    for pil_img, y in image_iter_G:
        bits = tcav_bits_for_image(
            pil_img=pil_img,
            cavs=cavs,
            get_grad_vec=get_grad_vec_G,
            class_to_inception_idx=class_to_inception_idx,
            all_keys=all_keys,
        )
        rows.append(bits)
        lbls.append(int(y))
    if not rows:
        print("[WARN] no images yielded by image_iter_G; nothing to export.")
        mark_done(shared_dir)
        return

    X = np.stack(rows, axis=0)
    y = np.array(lbls, dtype=int)

    # Write CSV in *write mode* (no append)
    tdir = os.path.join(shared_dir, "tables")
    os.makedirs(tdir, exist_ok=True)
    csv_path = os.path.join(tdir, "imagenet.csv")
    feat_cols = [f"feature{i:03d}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df["label"] = y
    df.to_csv(csv_path, index=False)  # overwrite

    mark_done(shared_dir)
    print(f"[DONE] wrote {csv_path} with shape {X.shape} and concept_names.npy with F={len(names)}")
