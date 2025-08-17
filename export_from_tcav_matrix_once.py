# export_from_tcav_matrix_once.py  — mapping-aware version
from __future__ import annotations
import os, json
from typing import Dict, Iterable, List, Tuple, Optional
import numpy as np
import pandas as pd

def _try_leader(shared_dir: str) -> bool:
    mdir = os.path.join(shared_dir, "_markers"); os.makedirs(mdir, exist_ok=True)
    p = os.path.join(mdir, "export_leader.lock")
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd); return True
    except FileExistsError:
        return False

def _mark_done(shared_dir: str) -> None:
    mdir = os.path.join(shared_dir, "_markers"); os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "export_done.ok"), "w") as f: f.write("done\n")

def _is_done(shared_dir: str) -> bool:
    return os.path.exists(os.path.join(shared_dir, "_markers", "export_done.ok"))

def _build_concepts(label_to_name: Dict[int,str], K: int) -> tuple[list[tuple[int,int]], list[str], list[str]]:
    all_keys, names = [], []
    class_ids_sorted = sorted(label_to_name.keys())
    class_names_sorted = [label_to_name[cid] for cid in class_ids_sorted]
    for cid in class_ids_sorted:
        cname = label_to_name[cid]
        for k in range(int(K)):
            all_keys.append((cid, k))
            names.append(f"{cname}-C{k:02d}")
    return all_keys, names, class_names_sorted

def _ensure_concepts_file(shared_dir: str, concept_names: list[str]) -> None:
    cdir = os.path.join(shared_dir, "concepts"); os.makedirs(cdir, exist_ok=True)
    p = os.path.join(cdir, "concept_names.npy")
    if not os.path.exists(p):
        np.save(p, np.array(concept_names, dtype=str))
    else:
        prev = np.load(p, allow_pickle=True)
        if len(prev) != len(concept_names) or any(a != b for a, b in zip(prev, concept_names)):
            raise SystemExit("[ERROR] concept_names.npy mismatch with current concept order.")

def _sanitize_header(name: str) -> str:
    # keep it simple & CSV-safe (no commas/newlines); LR-XFL doesn't need names, this is for the optional readable CSV
    return name.replace(",", "_").replace("\n", " ").strip()

def _write_tables_and_maps(shared_dir: str, X: np.ndarray, y: np.ndarray, concept_names: list[str]) -> tuple[str, str]:
    tdir = os.path.join(shared_dir, "tables"); os.makedirs(tdir, exist_ok=True)
    cdir = os.path.join(shared_dir, "concepts"); os.makedirs(cdir, exist_ok=True)

    # 1) Primary file expected by LR-XFL: feature000..feature{F-1}, label
    feat_cols = [f"feature{i:03d}" for i in range(X.shape[1])]
    df_feat = pd.DataFrame(X, columns=feat_cols); df_feat["label"] = y.astype(int)
    csv_path = os.path.join(tdir, "imagenet.csv")
    df_feat.to_csv(csv_path, index=False)

    # 2) Human-readable variant (headers are concept names)
    name_cols = [_sanitize_header(n) for n in concept_names]
    df_named = pd.DataFrame(X, columns=name_cols); df_named["label"] = y.astype(int)
    csv_named_path = os.path.join(tdir, "imagenet_with_names.csv")
    df_named.to_csv(csv_named_path, index=False)

    # 3) Explicit mapping: feature000 -> concept_name (CSV + JSON)
    map_rows = [{"column": f"feature{i:03d}", "concept_name": concept_names[i]} for i in range(len(concept_names))]
    map_csv = os.path.join(cdir, "feature_map.csv")
    map_json = os.path.join(cdir, "feature_map.json")
    pd.DataFrame(map_rows).to_csv(map_csv, index=False)
    with open(map_json, "w") as f:
        json.dump(map_rows, f, indent=2)

    return csv_path, csv_named_path

def export_global_tcav_from_matrix_once(
    *,
    shared_dir: str,
    label_to_name: Dict[int, str],
    clusters_per_class: int,
    tcav_matrix: np.ndarray,        # (F, C) or (C, F)
    label_iter: Iterable[int],      # one label per image
    threshold: float = 0.60,        # single cutoff for binarization & saliency
    concept_names_override: Optional[list[str]] = None,
    overwrite: bool = False,
) -> None:
    if (not overwrite) and _is_done(shared_dir):
        print("[SKIP] export already completed."); return
    if not _try_leader(shared_dir):
        print("[SKIP] another process is exporting; skipping."); return

    C = len(label_to_name)
    F_expected = C * int(clusters_per_class)

    M = np.array(tcav_matrix, dtype=np.float32)
    if M.shape == (F_expected, C):
        pass
    elif M.shape == (C, F_expected):
        M = M.T
    else:
        raise ValueError(f"tcav_matrix shape {M.shape} != (F={F_expected}, C={C}) or (C,F).")

    # concept names (deterministic or provided)
    if concept_names_override is None:
        _, concept_names, class_names_sorted = _build_concepts(label_to_name, clusters_per_class)
    else:
        concept_names = list(concept_names_override)
        class_names_sorted = [label_to_name[c] for c in sorted(label_to_name)]
        if len(concept_names) != F_expected:
            raise ValueError(f"concept_names length {len(concept_names)} != F_expected {F_expected}")
    _ensure_concepts_file(shared_dir, concept_names)

    # salient mask (not strictly required for LR-XFL, but handy)
    mask = (M > float(threshold))  # (F, C) bool
    cdir = os.path.join(shared_dir, "concepts"); os.makedirs(cdir, exist_ok=True)
    np.save(os.path.join(cdir, "salient_mask.npy"), mask.astype(bool))
    pd.DataFrame(mask.astype(int), index=concept_names, columns=[label_to_name[c] for c in sorted(label_to_name)])\
      .to_csv(os.path.join(cdir, "salient_concepts.csv"))

    # rows: pick the column for each image label
    Xc = mask.astype(np.float32)   # (F, C)
    rows, labels = [], []
    for y in label_iter:
        y = int(y)
        rows.append(Xc[:, y])
        labels.append(y)
    X = np.stack(rows, axis=0) if rows else np.zeros((0, F_expected), dtype=np.float32)
    y = np.array(labels, dtype=int)

    # WRITE + MAP (this guarantees header↔concept alignment is recorded)
    csv_path, csv_named_path = _write_tables_and_maps(shared_dir, X, y, concept_names)

    _mark_done(shared_dir)
    print(f"[DONE] wrote:\n - {csv_path}\n - {csv_named_path}\n - concepts/feature_map.(csv|json)\n - concepts/concept_names.npy\nthreshold={threshold}, shape={X.shape}")

