"""
track_a.py
----------
Track A: COLMAP-based Structure-from-Motion + (optional) Multi-View Stereo
for all four datasets.

This is a thin Python driver around the COLMAP command-line interface.
It uses CPU-only feature extraction and matching, which is the right choice
on a machine without an NVIDIA GPU.

Pipeline per dataset:
    1. Stage images into a workspace folder (COLMAP wants a flat dir).
    2. colmap feature_extractor          (CPU SIFT)
    3. colmap exhaustive_matcher         (CPU)
    4. colmap mapper                     (incremental SfM, CPU)
    5. colmap model_converter -> .ply    (sparse cloud)
    6. (optional, --dense)
       colmap image_undistorter
       colmap patch_match_stereo          (CPU is very slow; can be hours)
       colmap stereo_fusion -> .ply       (dense cloud)
    7. Convert the .ply output into our submission .txt format.
    8. For Box/Entrance: read back the COLMAP-estimated poses and report
       how they compare to the given poses (after similarity-aligning).

Usage:
    python src/track_a.py Box
    python src/track_a.py all
    python src/track_a.py Statue --dense
    python src/track_a.py Fountain --colmap-exe "C:/Tools/COLMAP/colmap.bat"

Requires COLMAP on PATH (or pass --colmap-exe).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (                                                   # noqa: E402
    list_dataset_images,
    load_K,
    save_xyz_txt,
)


# ----------------------------------------------------------------------------
# Dataset registry
# ----------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    name: str
    image_dir: Path
    prefix: str
    k_file: Path | None                 # may be None if K is hardcoded
    image_size_wh: tuple[int, int]
    poses_file: Path | None = None      # only Box / Entrance
    extra_args: list[str] = field(default_factory=list)


DATASETS = {
    "Box": DatasetSpec(
        name="Box",
        image_dir=Path("data/Box"),
        prefix="box",
        k_file=Path("data/Box/K.txt"),
        image_size_wh=(1920, 1080),
        poses_file=Path("data/Box/boxInput.txt"),
    ),
    "Entrance": DatasetSpec(
        name="Entrance",
        image_dir=Path("data/Entrance"),
        prefix="entrance",
        k_file=Path("data/Entrance/K.txt"),
        image_size_wh=(1920, 1080),
        poses_file=Path("data/Entrance/entranceInput.txt"),
    ),
    "Statue": DatasetSpec(
        name="Statue",
        image_dir=Path("data/Statue"),
        prefix="statue",
        k_file=Path("data/Statue/K.txt"),
        image_size_wh=(1920, 1080),
    ),
    "Fountain": DatasetSpec(
        name="Fountain",
        image_dir=Path("data/Fountain"),
        prefix="fountain",
        k_file=Path("data/Fountain/K.txt"),
        image_size_wh=(3072, 2048),
    ),
}

# K used if a dataset has no K.txt - matches Box/Entrance/Statue
DEFAULT_K_1080P = np.array([
    [960.0,   0.0, 960.0],
    [  0.0, 960.0, 540.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)


# ----------------------------------------------------------------------------
# COLMAP helpers
# ----------------------------------------------------------------------------

def find_colmap(explicit: str | None) -> str:
    """Locate the COLMAP binary.  Returns a string suitable for subprocess."""
    if explicit:
        if not Path(explicit).exists():
            raise FileNotFoundError(f"--colmap-exe not found: {explicit}")
        return explicit
    # PATH lookup
    found = shutil.which("colmap") or shutil.which("colmap.bat") \
            or shutil.which("COLMAP.bat") or shutil.which("colmap.exe")
    if not found:
        raise RuntimeError(
            "COLMAP not found on PATH.  Add the COLMAP folder to your system "
            "PATH or pass --colmap-exe 'C:/path/to/COLMAP/colmap.bat'."
        )
    return found


def run_cmd(cmd: list[str], log_path: Path | None = None) -> None:
    """Run a subprocess command, streaming output, and raise on non-zero exit."""
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                logf.write(line)
            proc.wait()
    else:
        proc = subprocess.run(cmd)
    rc = proc.returncode if log_path is None else proc.returncode
    dt = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"Command failed (exit {rc}) after {dt:.1f}s")
    print(f"<<< done in {dt:.1f}s")


def k_to_pinhole_params(K: np.ndarray) -> str:
    """COLMAP PINHOLE expects 'fx,fy,cx,cy' as a comma string."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return f"{fx},{fy},{cx},{cy}"


# ----------------------------------------------------------------------------
# Image staging
# ----------------------------------------------------------------------------

def stage_images(spec: DatasetSpec, project_root: Path,
                 workspace: Path) -> Path:
    """Copy the dataset images into <workspace>/images/.

    We copy (not symlink) for Windows compatibility.  Only image files are
    copied - any .txt files in the source folder are left behind.
    """
    src = project_root / spec.image_dir
    dst = workspace / "images"
    dst.mkdir(parents=True, exist_ok=True)
    paths = list_dataset_images(src, spec.prefix)
    if not paths:
        raise FileNotFoundError(
            f"No images matching '{spec.prefix}<N>.[png|jpg]' under {src}")
    for p in paths:
        target = dst / p.name
        if not target.exists() or target.stat().st_size != p.stat().st_size:
            shutil.copy2(p, target)
    print(f"Staged {len(paths)} images into {dst}")
    return dst


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

def run_sparse(colmap: str, spec: DatasetSpec, project_root: Path,
               workspace: Path, K: np.ndarray) -> Path:
    """Run COLMAP feature extraction, matching, and sparse mapping.

    Returns the path to the produced sparse model directory (e.g.
    workspace/sparse/0).
    """
    images_dir = stage_images(spec, project_root, workspace)
    db_path = workspace / "database.db"
    if db_path.exists():
        db_path.unlink()   # fresh DB each run, avoids stale-feature issues

    log_dir = workspace / "logs"

    # 1) Feature extraction (CPU, single shared PINHOLE camera with our K)
    run_cmd([
        colmap, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path",    str(images_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model",  "PINHOLE",
        "--ImageReader.camera_params", k_to_pinhole_params(K),
    ], log_path=log_dir / "feature_extractor.log")

    # 2) Exhaustive matching (CPU)
    run_cmd([
        colmap, "exhaustive_matcher",
        "--database_path", str(db_path),
    ], log_path=log_dir / "exhaustive_matcher.log")

    # 3) Mapper (sparse incremental SfM)
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([
        colmap, "mapper",
        "--database_path", str(db_path),
        "--image_path",    str(images_dir),
        "--output_path",   str(sparse_dir),
    ], log_path=log_dir / "mapper.log")

    # Find produced model(s) - COLMAP creates sparse/0, sparse/1, ...
    sub_models = sorted([d for d in sparse_dir.iterdir() if d.is_dir()],
                        key=lambda p: int(p.name) if p.name.isdigit() else 999)
    if not sub_models:
        raise RuntimeError(
            f"COLMAP mapper produced no models in {sparse_dir}.  "
            "Check the log files in {log_dir} for the reason.")
    # Pick the largest model (most registered images)
    def model_size(d: Path) -> int:
        images_bin = d / "images.bin"
        return images_bin.stat().st_size if images_bin.exists() else 0
    best = max(sub_models, key=model_size)
    print(f"Selected sparse model: {best}")
    return best


def export_sparse(colmap: str, sparse_model: Path,
                  workspace: Path, out_name: str) -> tuple[Path, Path]:
    """Convert the COLMAP sparse model to PLY (binary) and TXT (plain text).

    Returns (ply_path, txt_dir).  txt_dir contains cameras.txt, images.txt,
    points3D.txt for downstream parsing.
    """
    ply_path = workspace / f"{out_name}_sparse.ply"
    run_cmd([
        colmap, "model_converter",
        "--input_path",  str(sparse_model),
        "--output_path", str(ply_path),
        "--output_type", "PLY",
    ])
    txt_dir = workspace / "sparse_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([
        colmap, "model_converter",
        "--input_path",  str(sparse_model),
        "--output_path", str(txt_dir),
        "--output_type", "TXT",
    ])
    return ply_path, txt_dir


def run_dense(colmap: str, sparse_model: Path, workspace: Path,
              out_name: str) -> Path:
    """Run dense MVS.  Returns path to the fused dense .ply.

    NOTE: patch_match_stereo on CPU is *very* slow.  Expect hours for big
    image sets.  COLMAP must be built with CPU-PatchMatch support (the
    'colmap-windows-no-cuda' release does).
    """
    dense_dir = workspace / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    log_dir = workspace / "logs"

    # 1) Undistort & prepare dense workspace
    run_cmd([
        colmap, "image_undistorter",
        "--image_path",  str(workspace / "images"),
        "--input_path",  str(sparse_model),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
        "--max_image_size", "1600",
    ], log_path=log_dir / "image_undistorter.log")

    # 2) Patch-match stereo (the slow step on CPU)
    run_cmd([
        colmap, "patch_match_stereo",
        "--workspace_path",    str(dense_dir),
        "--workspace_format",  "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        # CPU-mode-only knobs to keep it from running for ever
        "--PatchMatchStereo.window_radius",  "5",
        "--PatchMatchStereo.num_iterations", "3",
        "--PatchMatchStereo.cache_size",     "16",
    ], log_path=log_dir / "patch_match_stereo.log")

    # 3) Fusion
    fused_ply = dense_dir / "fused.ply"
    run_cmd([
        colmap, "stereo_fusion",
        "--workspace_path",   str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type",       "geometric",
        "--output_path",      str(fused_ply),
    ], log_path=log_dir / "stereo_fusion.log")

    if not fused_ply.exists():
        raise RuntimeError(f"stereo_fusion did not produce {fused_ply}")

    # Copy to a friendlier name
    out_ply = workspace / f"{out_name}_dense.ply"
    shutil.copy2(fused_ply, out_ply)
    return out_ply


# ----------------------------------------------------------------------------
# PLY reader (just enough to grab vertices for submission TXT)
# ----------------------------------------------------------------------------

def read_ply_vertices(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Minimal ASCII/binary-little-endian PLY vertex reader.

    Returns (points (N,3), colors (N,3) uint8 or None).
    Handles only the subset of PLY that COLMAP emits.
    """
    with open(path, "rb") as f:
        # Read header
        line = f.readline().decode("ascii").strip()
        if line != "ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = None
        n_vert = None
        props: list[tuple[str, str]] = []   # (type, name)
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n_vert = int(line.split()[2])
            elif line.startswith("property"):
                parts = line.split()
                # property <type> <name>     (no list props expected)
                props.append((parts[1], parts[2]))
            elif line == "end_header":
                break
        assert fmt is not None and n_vert is not None

        # Build dtype
        type_map = {
            "float":   "<f4", "float32": "<f4",
            "double":  "<f8", "float64": "<f8",
            "uchar":   "u1",  "uint8":   "u1",
            "char":    "i1",  "int8":    "i1",
            "ushort":  "<u2", "uint16":  "<u2",
            "short":   "<i2", "int16":   "<i2",
            "uint":    "<u4", "uint32":  "<u4",
            "int":     "<i4", "int32":   "<i4",
        }

        if fmt.startswith("binary_little_endian"):
            dt = np.dtype([(name, type_map[t]) for t, name in props])
            data = np.frombuffer(f.read(dt.itemsize * n_vert), dtype=dt)
        elif fmt == "ascii":
            # Read remaining text
            rest = f.read().decode("ascii")
            rows = [r.split() for r in rest.strip().splitlines()[:n_vert]]
            arrs = {}
            for col, (t, name) in enumerate(props):
                arr = np.array([r[col] for r in rows], dtype=type_map[t])
                arrs[name] = arr
            data = arrs
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")

    def col(name: str) -> np.ndarray:
        return np.asarray(data[name])

    points = np.stack([col("x"), col("y"), col("z")], axis=1).astype(np.float64)
    colors = None
    prop_names = {p[1] for p in props}
    if {"red", "green", "blue"}.issubset(prop_names):
        colors = np.stack([col("red"), col("green"), col("blue")],
                          axis=1).astype(np.uint8)
    return points, colors


# ----------------------------------------------------------------------------
# Top-level per-dataset runner
# ----------------------------------------------------------------------------

def run_dataset(name: str, colmap: str, project_root: Path,
                dense: bool, force: bool) -> None:
    spec = DATASETS[name]
    workspace = project_root / "output" / f"colmap_{name.lower()}"
    if force and workspace.exists():
        print(f"--force given; removing existing workspace {workspace}")
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # Resolve intrinsics
    k_path = project_root / spec.k_file if spec.k_file else None
    if k_path and k_path.exists():
        K = load_K(k_path)
        print(f"Using K from {k_path}:\n{K}")
    else:
        K = DEFAULT_K_1080P.copy()
        print(f"No K.txt found at {k_path}; using default 1920x1080 FOV-90 K.")

    print(f"\n========== {name}: SPARSE ==========")
    sparse_model = run_sparse(colmap, spec, project_root, workspace, K)
    ply_sparse, txt_dir = export_sparse(colmap, sparse_model, workspace,
                                        out_name=name.lower())

    # Convert sparse .ply to submission .txt
    out_root = project_root / "output"
    out_root.mkdir(parents=True, exist_ok=True)
    pts, cols = read_ply_vertices(ply_sparse)
    submit_ply = out_root / f"{name.lower()}_track_a_sparse.ply"
    submit_txt = out_root / f"{name.lower()}_track_a_sparse.txt"
    shutil.copy2(ply_sparse, submit_ply)
    save_xyz_txt(submit_txt, pts, cols if cols is not None else None)
    print(f"Sparse: {len(pts):,} points -> {submit_ply}, {submit_txt}")

    if dense:
        print(f"\n========== {name}: DENSE ==========")
        print("Warning: patch_match_stereo on CPU is slow.  This may take "
              "30 minutes to several hours.")
        ply_dense = run_dense(colmap, sparse_model, workspace, name.lower())
        pts_d, cols_d = read_ply_vertices(ply_dense)
        submit_ply_d = out_root / f"{name.lower()}_track_a_dense.ply"
        submit_txt_d = out_root / f"{name.lower()}_track_a_dense.txt"
        shutil.copy2(ply_dense, submit_ply_d)
        save_xyz_txt(submit_txt_d, pts_d, cols_d if cols_d is not None else None)
        print(f"Dense: {len(pts_d):,} points -> {submit_ply_d}, {submit_txt_d}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    choices = list(DATASETS.keys()) + ["all"]
    p.add_argument("dataset", choices=choices,
                   help="Dataset name or 'all'.")
    p.add_argument("--project-root", default=".")
    p.add_argument("--colmap-exe", default=None,
                   help="Path to colmap.bat/colmap.exe if not on PATH.")
    p.add_argument("--dense", action="store_true",
                   help="Also run dense MVS (CPU, very slow).")
    p.add_argument("--force", action="store_true",
                   help="Delete the workspace folder before running.")
    args = p.parse_args()

    project_root = Path(args.project_root).resolve()
    colmap = find_colmap(args.colmap_exe)
    print(f"Using COLMAP at: {colmap}")

    names = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for n in names:
        run_dataset(n, colmap, project_root, dense=args.dense, force=args.force)


if __name__ == "__main__":
    main()
