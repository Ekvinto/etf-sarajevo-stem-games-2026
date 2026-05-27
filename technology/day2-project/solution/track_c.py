"""
track_c.py
----------
Track C: Custom OpenCV pipeline using KNOWN camera poses.
Applies to the Box and Entrance datasets.

Algorithm overview:
    1. Parse camera poses (boxInput.txt / entranceInput.txt) -> P_i = K @ [R_i | t_i].
    2. SIFT-detect features in every image.
    3. For every pair (i, j) with |i - j| <= MAX_PAIR_GAP:
        a. FLANN-match descriptors with Lowe's ratio test.
        b. Filter outliers with a fundamental-matrix RANSAC.
        c. Triangulate surviving matches using cv2.triangulatePoints.
        d. Reject points with negative depth (behind either camera).
        e. Reject points whose reprojection error in EITHER view exceeds
           REPROJ_PX_THRESHOLD.
        f. Reject implausibly distant points.
    4. (Optional multi-view step) Promote 2-view tracks to multi-view tracks
       when the same feature is seen by 3+ cameras; re-triangulate with all
       views via a linear DLT for better accuracy.
    5. Deduplicate by voxel binning so we don't emit the same physical
       surface point 50 times.
    6. Sample the colour of each surviving point from one source image.
    7. Save .ply (binary, for MeshLab) and .txt (X Y Z R G B, for submission).

Usage (from project root):
    python src/track_c.py Box
    python src/track_c.py Entrance
    python src/track_c.py Box --max-images 6      # quick smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Allow running as "python src/track_c.py" (no install needed)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (                                                   # noqa: E402
    CameraPose,
    build_projection_matrix,
    list_dataset_images,
    load_K,
    parse_cameras_file,
    pose_to_opencv,
    project_point,
    sanity_check_pose,
    save_ply,
    save_xyz_txt,
)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

@dataclass
class Config:
    # SIFT
    sift_n_features: int = 0          # 0 = no cap; SIFT finds what it finds
    sift_contrast_threshold: float = 0.04
    sift_edge_threshold: float = 10.0

    # Matching
    ratio_test: float = 0.75          # Lowe's ratio; lower = stricter
    min_matches_per_pair: int = 30    # skip a pair if fewer raw matches

    # Pair selection
    max_pair_gap: int = 4             # 1 = only consecutive cams; larger = wider baselines

    # Triangulation filtering
    reproj_px_threshold: float = 4.0  # max reprojection error in EITHER view
    min_triangulation_angle_deg: float = 1.5   # discard near-collinear rays
    min_depth: float = 1.0             # in front of camera, in world units
    max_depth: float = 5000.0          # discard runaway points

    # Multi-view promotion
    enable_multiview: bool = True
    min_track_length: int = 3          # only promote tracks seen by >= N cams
    multiview_reproj_px_threshold: float = 6.0  # slightly looser after DLT

    # Deduplication
    voxel_size: float = 2.0            # world units; smaller = denser final cloud
    enable_dedup: bool = True

    # IO
    output_dir: Path = Path("output")


# ----------------------------------------------------------------------------
# Dataset registry
# ----------------------------------------------------------------------------

DATASETS = {
    "Box": {
        "image_dir":    Path("data/Box"),
        "prefix":       "box",
        "poses_file":   Path("data/Box/boxInput.txt"),
        # K is the global one - same calibration as Entrance
        "k_file":       Path("data/Box/K.txt"),  # we'll fall back to a default
    },
    "Entrance": {
        "image_dir":    Path("data/Entrance"),
        "prefix":       "entrance",
        "poses_file":   Path("data/Entrance/entranceInput.txt"),
        "k_file":       Path("data/Entrance/K.txt"),
    },
}

# Default K for the 1920x1080, FOV-90 cameras when no K.txt is alongside the data
DEFAULT_K_1080P = np.array([
    [960.0,   0.0, 960.0],
    [  0.0, 960.0, 540.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------

def load_dataset(name: str, project_root: Path) -> tuple[
        list[Path], list[CameraPose], np.ndarray]:
    """Locate images, poses, and intrinsics for a named dataset."""
    cfg = DATASETS[name]
    img_dir   = project_root / cfg["image_dir"]
    poses_fp  = project_root / cfg["poses_file"]
    k_fp      = project_root / cfg["k_file"]

    if not img_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {img_dir}")
    if not poses_fp.exists():
        raise FileNotFoundError(f"Poses file not found: {poses_fp}")

    image_paths = list_dataset_images(img_dir, cfg["prefix"])
    if not image_paths:
        raise FileNotFoundError(
            f"No images matching '{cfg['prefix']}<N>.png' under {img_dir}")

    cams = parse_cameras_file(poses_fp)

    # Cross-check counts
    if len(image_paths) != len(cams):
        print(f"  WARNING: found {len(image_paths)} images but {len(cams)} "
              f"camera poses.  Will only use the first {min(len(image_paths), len(cams))}.")
        n = min(len(image_paths), len(cams))
        image_paths = image_paths[:n]
        cams = cams[:n]

    if k_fp.exists():
        K = load_K(k_fp)
        print(f"  K loaded from {k_fp.name}.")
    else:
        K = DEFAULT_K_1080P.copy()
        print(f"  No K.txt in dataset folder; using default 1920x1080 FOV-90 K.")
    print(f"  K =\n{K}")

    return image_paths, cams, K


def detect_features(image_paths: list[Path], cfg: Config
                    ) -> tuple[list[list[cv2.KeyPoint]],
                               list[np.ndarray],
                               list[np.ndarray]]:
    """Detect SIFT features in every image.

    Returns:
        keypoints_per_image: list of OpenCV KeyPoint lists
        descriptors_per_image: list of (M, 128) float32 arrays
        images_bgr: list of color images (uint8, BGR) for later color sampling
    """
    sift = cv2.SIFT_create(
        nfeatures=cfg.sift_n_features,
        contrastThreshold=cfg.sift_contrast_threshold,
        edgeThreshold=cfg.sift_edge_threshold,
    )

    keypoints: list[list[cv2.KeyPoint]] = []
    descriptors: list[np.ndarray] = []
    images: list[np.ndarray] = []

    for p in tqdm(image_paths, desc="SIFT", unit="img"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image {p}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, desc = sift.detectAndCompute(gray, None)
        if desc is None:
            desc = np.zeros((0, 128), dtype=np.float32)
        keypoints.append(list(kps))
        descriptors.append(desc.astype(np.float32))
        images.append(img)

    total = sum(len(k) for k in keypoints)
    print(f"  Total SIFT features across all images: {total}")
    return keypoints, descriptors, images


def match_pair(desc_a: np.ndarray, desc_b: np.ndarray,
               ratio: float) -> list[cv2.DMatch]:
    """FLANN-based knn match + Lowe's ratio test."""
    if len(desc_a) < 2 or len(desc_b) < 2:
        return []
    # FLANN with KD-tree for float descriptors (SIFT)
    index_params  = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(desc_a, desc_b, k=2)
    good: list[cv2.DMatch] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def filter_with_fundamental(pts_a: np.ndarray, pts_b: np.ndarray,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC-fit fundamental matrix; return the inlier mask + filtered pts."""
    if len(pts_a) < 8:
        return pts_a, pts_b, np.zeros(len(pts_a), dtype=bool)
    F, mask = cv2.findFundamentalMat(
        pts_a, pts_b,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=2.0,
        confidence=0.999,
    )
    if mask is None:
        return pts_a, pts_b, np.zeros(len(pts_a), dtype=bool)
    inliers = mask.ravel().astype(bool)
    return pts_a[inliers], pts_b[inliers], inliers


def triangulate_pair(P_a: np.ndarray, P_b: np.ndarray,
                     pts_a: np.ndarray, pts_b: np.ndarray) -> np.ndarray:
    """Triangulate 2D correspondences into world XYZ via cv2.triangulatePoints.

    Returns (N, 3) array of world points.
    """
    if len(pts_a) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    # cv2 expects shape (2, N)
    pts4 = cv2.triangulatePoints(P_a, P_b, pts_a.T.astype(np.float64),
                                          pts_b.T.astype(np.float64))
    pts3 = (pts4[:3] / pts4[3]).T  # (N, 3)
    return pts3


def triangulation_angle(C_a: np.ndarray, C_b: np.ndarray,
                        X: np.ndarray) -> np.ndarray:
    """Angle (degrees) between the two viewing rays X->C_a and X->C_b for
    each 3D point X.  Small angles -> ill-conditioned triangulation.
    """
    v1 = C_a[None, :] - X    # (N, 3)
    v2 = C_b[None, :] - X
    n1 = np.linalg.norm(v1, axis=1) + 1e-12
    n2 = np.linalg.norm(v2, axis=1) + 1e-12
    cos = np.clip((v1 * v2).sum(axis=1) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World coordinates of the camera optical center: C = -R^T @ t."""
    return -R.T @ t


def filter_triangulated(
        pts3d: np.ndarray,
        pts_a: np.ndarray, pts_b: np.ndarray,
        P_a: np.ndarray, P_b: np.ndarray,
        R_a: np.ndarray, t_a: np.ndarray,
        R_b: np.ndarray, t_b: np.ndarray,
        cfg: Config,
) -> np.ndarray:
    """Apply all per-point filters; return boolean keep-mask of shape (N,)."""
    N = pts3d.shape[0]
    if N == 0:
        return np.zeros(0, dtype=bool)

    # Cheirality: depth in front of each camera must be positive
    cam_pts_a = (R_a @ pts3d.T + t_a[:, None]).T   # (N, 3)
    cam_pts_b = (R_b @ pts3d.T + t_b[:, None]).T
    keep = (cam_pts_a[:, 2] > cfg.min_depth) & (cam_pts_a[:, 2] < cfg.max_depth)
    keep &= (cam_pts_b[:, 2] > cfg.min_depth) & (cam_pts_b[:, 2] < cfg.max_depth)

    # Reprojection error in both views
    proj_a = project_point(P_a, pts3d)
    proj_b = project_point(P_b, pts3d)
    err_a = np.linalg.norm(proj_a - pts_a, axis=1)
    err_b = np.linalg.norm(proj_b - pts_b, axis=1)
    keep &= (err_a < cfg.reproj_px_threshold) & (err_b < cfg.reproj_px_threshold)

    # Triangulation angle - reject near-collinear rays
    C_a = camera_center(R_a, t_a)
    C_b = camera_center(R_b, t_b)
    ang = triangulation_angle(C_a, C_b, pts3d)
    keep &= ang > cfg.min_triangulation_angle_deg

    return keep


def linear_triangulate_multiview(Ps: list[np.ndarray],
                                 pts: list[np.ndarray]) -> np.ndarray:
    """DLT linear triangulation from M >= 2 views of the SAME 3D point.

    Args:
        Ps:  list of M projection matrices (3x4)
        pts: list of M (u, v) image coords

    Returns:
        (3,) world point
    """
    A: list[np.ndarray] = []
    for P, (u, v) in zip(Ps, pts):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A_mat = np.stack(A, axis=0)
    _, _, Vt = np.linalg.svd(A_mat)
    X_h = Vt[-1]
    X = X_h[:3] / X_h[3]
    return X


def voxel_downsample(points: np.ndarray, colors: np.ndarray | None,
                     voxel_size: float) -> tuple[np.ndarray, np.ndarray | None]:
    """Bin points into voxels and keep one representative per voxel (the mean).

    Returns the downsampled (points, colors).

    Uses a collision-free encoding by viewing the (int32, int32, int32) voxel
    coordinate triples as a single structured-dtype entry, then np.unique on it.
    """
    if len(points) == 0 or voxel_size <= 0:
        return points, colors

    # Quantise to voxel grid.  int32 is plenty for any realistic scene
    # (-2.1e9 .. 2.1e9 voxel indices).
    coords = np.floor(points / voxel_size).astype(np.int32)

    # View the (N, 3) int32 array as a single column of compound dtype - this
    # gives each voxel a unique scalar identity for np.unique.
    view_dtype = np.dtype([("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    voxel_view = coords.view(view_dtype).reshape(-1)

    # `inverse` gives, for each input row, which unique-voxel it belongs to.
    _, inverse = np.unique(voxel_view, return_inverse=True)
    n_voxels = int(inverse.max()) + 1

    # Mean of points per voxel via add.at + counts
    sums = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(sums, inverse, points.astype(np.float64))
    counts = np.bincount(inverse, minlength=n_voxels).astype(np.float64)
    out_pts = (sums / counts[:, None]).astype(points.dtype)

    if colors is None:
        return out_pts, None

    csums = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(csums, inverse, colors.astype(np.float64))
    out_col = (csums / counts[:, None]).astype(colors.dtype)
    return out_pts, out_col


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------

def run(dataset_name: str, cfg: Config, project_root: Path,
        max_images: int | None = None) -> None:
    print(f"\n=== Track C on {dataset_name} ===")
    image_paths, cams, K = load_dataset(dataset_name, project_root)

    if max_images is not None:
        image_paths = image_paths[:max_images]
        cams = cams[:max_images]
        print(f"  Limiting to first {max_images} images (smoke test).")

    # Per-camera sanity check (cheap)
    print("Validating coordinate-system conversion ...")
    for c in cams:
        sanity_check_pose(c, K, verbose=False)
    print(f"  All {len(cams)} cameras validated.")

    # Build projection matrices once
    Rts: list[tuple[np.ndarray, np.ndarray]] = [pose_to_opencv(c) for c in cams]
    Ps: list[np.ndarray] = [build_projection_matrix(K, R, t) for R, t in Rts]

    # Feature detection
    print("Detecting SIFT features ...")
    t0 = time.time()
    keypoints, descriptors, images_bgr = detect_features(image_paths, cfg)
    print(f"  Feature detection took {time.time() - t0:.1f} s.")

    # Pre-extract pixel arrays for each image's keypoints (for indexed lookups)
    kp_xy: list[np.ndarray] = [
        np.array([kp.pt for kp in kps], dtype=np.float64) if kps
        else np.zeros((0, 2), dtype=np.float64)
        for kps in keypoints
    ]

    N = len(image_paths)
    pairs_to_try: list[tuple[int, int]] = []
    for i in range(N):
        for j in range(i + 1, min(N, i + cfg.max_pair_gap + 1)):
            pairs_to_try.append((i, j))
    print(f"Considering {len(pairs_to_try)} image pairs "
          f"(gap <= {cfg.max_pair_gap}).")

    # Tracks for multi-view promotion: maps a "track id" -> dict of
    # {image_index: keypoint_index}.  We grow tracks transitively across
    # pair matches using union-find over (image, keypoint) nodes.
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(x: tuple[int, int]) -> tuple[int, int]:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Per-pair triangulated batches (we keep these for the 2-view fallback)
    pair_points: list[np.ndarray] = []
    pair_colors: list[np.ndarray] = []
    pair_inlier_records: list[tuple[int, int, np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]] = []

    print("Matching + triangulating each pair ...")
    n_total_inliers = 0
    for (i, j) in tqdm(pairs_to_try, desc="pairs", unit="pair"):
        matches = match_pair(descriptors[i], descriptors[j], cfg.ratio_test)
        if len(matches) < cfg.min_matches_per_pair:
            continue

        idx_a = np.array([m.queryIdx for m in matches], dtype=np.int64)
        idx_b = np.array([m.trainIdx for m in matches], dtype=np.int64)
        pts_a = kp_xy[i][idx_a]
        pts_b = kp_xy[j][idx_b]

        # F-matrix RANSAC outlier rejection
        pts_a_in, pts_b_in, mask = filter_with_fundamental(pts_a, pts_b)
        if len(pts_a_in) < cfg.min_matches_per_pair // 2:
            continue
        idx_a_in = idx_a[mask]
        idx_b_in = idx_b[mask]

        # Triangulate
        X = triangulate_pair(Ps[i], Ps[j], pts_a_in, pts_b_in)

        # Filter
        R_a, t_a = Rts[i]
        R_b, t_b = Rts[j]
        keep = filter_triangulated(X, pts_a_in, pts_b_in,
                                   Ps[i], Ps[j], R_a, t_a, R_b, t_b, cfg)
        X_keep = X[keep]
        if len(X_keep) == 0:
            continue
        idx_a_keep = idx_a_in[keep]
        idx_b_keep = idx_b_in[keep]
        pts_a_keep = pts_a_in[keep]
        pts_b_keep = pts_b_in[keep]

        # Record for the pair-only fallback
        # Sample colors from image i (BGR -> RGB)
        h_i, w_i = images_bgr[i].shape[:2]
        col_uv = np.clip(np.round(pts_a_keep).astype(np.int32), 0,
                         [w_i - 1, h_i - 1])
        cols_bgr = images_bgr[i][col_uv[:, 1], col_uv[:, 0]]
        cols_rgb = cols_bgr[:, ::-1].copy()
        pair_points.append(X_keep)
        pair_colors.append(cols_rgb)
        pair_inlier_records.append((i, j, idx_a_keep, idx_b_keep,
                                    pts_a_keep, pts_b_keep))

        # Union-find for tracks
        for ka, kb in zip(idx_a_keep, idx_b_keep):
            union((i, int(ka)), (j, int(kb)))

        n_total_inliers += len(X_keep)

    print(f"  Total inlier 2-view triangulations: {n_total_inliers:,}")

    # ------------------------------------------------------------------
    # Multi-view promotion
    # ------------------------------------------------------------------
    final_points: list[np.ndarray] = []
    final_colors: list[np.ndarray] = []
    used_nodes: set[tuple[int, int]] = set()

    if cfg.enable_multiview and pair_inlier_records:
        print("Building multi-view tracks ...")
        # Group nodes by their union-find root
        clusters: dict[tuple[int, int],
                       dict[int, int]] = defaultdict(dict)  # root -> {img: kp}
        # Collect all nodes that appear in any pair
        all_nodes: set[tuple[int, int]] = set()
        for (i, j, ia, ib, _, _) in pair_inlier_records:
            for k in ia:
                all_nodes.add((i, int(k)))
            for k in ib:
                all_nodes.add((j, int(k)))

        for node in all_nodes:
            root = find(node)
            img_idx, kp_idx = node
            # If the same image already appears in this cluster, keep the
            # first one we saw (avoid contradictory observations from the
            # same image).
            if img_idx not in clusters[root]:
                clusters[root][img_idx] = kp_idx

        # Triangulate each long-enough track with DLT
        n_mv_kept = 0
        for root, obs in tqdm(clusters.items(), desc="multi-view",
                              unit="track"):
            if len(obs) < cfg.min_track_length:
                continue
            view_indices = sorted(obs.keys())
            Ps_track = [Ps[v] for v in view_indices]
            pts_track = [kp_xy[v][obs[v]] for v in view_indices]
            X = linear_triangulate_multiview(Ps_track, pts_track)

            # Check reprojection error across ALL views
            errs = []
            depths = []
            for v, pix in zip(view_indices, pts_track):
                p = project_point(Ps[v], X)
                if not np.isfinite(p).all():
                    errs.append(np.inf)
                    break
                errs.append(float(np.linalg.norm(p - pix)))
                R_v, t_v = Rts[v]
                depths.append(float((R_v @ X + t_v)[2]))
            if not errs or max(errs) > cfg.multiview_reproj_px_threshold:
                continue
            if min(depths) < cfg.min_depth or max(depths) > cfg.max_depth:
                continue

            # Color from the first observation
            v0 = view_indices[0]
            uv = pts_track[0]
            h_v, w_v = images_bgr[v0].shape[:2]
            uvi = np.clip(np.round(uv).astype(np.int32), [0, 0],
                          [w_v - 1, h_v - 1])
            col_bgr = images_bgr[v0][uvi[1], uvi[0]]
            col_rgb = col_bgr[::-1].copy()

            final_points.append(X.reshape(1, 3))
            final_colors.append(col_rgb.reshape(1, 3))
            n_mv_kept += 1

            # Mark the nodes so we don't double-count from pair_points
            for v in view_indices:
                used_nodes.add((v, int(obs[v])))

        print(f"  Multi-view tracks kept: {n_mv_kept:,}")

    # Fallback: include 2-view points whose features were NOT promoted to MV
    print("Merging in remaining 2-view points ...")
    n_pair_kept = 0
    for (X_batch, col_batch, (i, j, ia, ib, pa, pb)) in zip(
            pair_points, pair_colors, pair_inlier_records):
        if not cfg.enable_multiview:
            # Take everything
            final_points.append(X_batch)
            final_colors.append(col_batch)
            n_pair_kept += len(X_batch)
            continue
        # Otherwise drop any point whose feature already appears in a MV track
        keep_mask = np.array([(i, int(ka)) not in used_nodes
                              and (j, int(kb)) not in used_nodes
                              for ka, kb in zip(ia, ib)], dtype=bool)
        if keep_mask.any():
            final_points.append(X_batch[keep_mask])
            final_colors.append(col_batch[keep_mask])
            n_pair_kept += int(keep_mask.sum())
    print(f"  2-view points kept: {n_pair_kept:,}")

    if not final_points:
        raise RuntimeError("No 3D points produced.  Inspect filter thresholds.")
    points = np.concatenate(final_points, axis=0)
    colors = np.concatenate(final_colors, axis=0).astype(np.uint8)
    print(f"Total points pre-dedup: {len(points):,}")

    # Voxel deduplication
    if cfg.enable_dedup and cfg.voxel_size > 0:
        print(f"Voxel-downsampling at {cfg.voxel_size} world units ...")
        points, colors = voxel_downsample(points, colors, cfg.voxel_size)
        print(f"  -> {len(points):,} points after dedup.")

    # Save outputs
    out_root = project_root / cfg.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    ply_path = out_root / f"{dataset_name.lower()}_track_c.ply"
    txt_path = out_root / f"{dataset_name.lower()}_track_c.txt"
    save_ply(ply_path, points, colors, binary=True)
    save_xyz_txt(txt_path, points, colors)
    print(f"\nWrote {ply_path}")
    print(f"Wrote {txt_path}")
    print(f"Final point cloud: {len(points):,} points\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=sorted(DATASETS.keys()),
                   help="Which dataset to process.")
    p.add_argument("--project-root", default=".",
                   help="Project root directory (default: cwd).")
    p.add_argument("--max-images", type=int, default=None,
                   help="Use only the first N images (smoke test).")

    # Knobs you may want to tweak at the CLI
    p.add_argument("--max-pair-gap", type=int, default=4)
    p.add_argument("--reproj-threshold", type=float, default=4.0)
    p.add_argument("--ratio-test", type=float, default=0.75)
    p.add_argument("--voxel-size", type=float, default=2.0)
    p.add_argument("--sift-contrast", type=float, default=0.04,
                   help="SIFT contrast threshold. Lower = more features. "
                        "Try 0.01 for low-texture scenes like Box.")
    p.add_argument("--sift-edge", type=float, default=10.0,
                   help="SIFT edge threshold. Higher = more features on edges.")
    p.add_argument("--min-tri-angle", type=float, default=1.5,
                   help="Minimum triangulation angle in degrees. Lower = more "
                        "permissive but noisier points.")
    p.add_argument("--no-multiview", action="store_true")
    p.add_argument("--no-dedup", action="store_true")
    args = p.parse_args()

    cfg = Config(
        sift_contrast_threshold=args.sift_contrast,
        sift_edge_threshold=args.sift_edge,
        min_triangulation_angle_deg=args.min_tri_angle,
        max_pair_gap=args.max_pair_gap,
        reproj_px_threshold=args.reproj_threshold,
        ratio_test=args.ratio_test,
        voxel_size=args.voxel_size,
        enable_multiview=not args.no_multiview,
        enable_dedup=not args.no_dedup,
    )

    project_root = Path(args.project_root).resolve()
    run(args.dataset, cfg, project_root, max_images=args.max_images)


if __name__ == "__main__":
    main()
