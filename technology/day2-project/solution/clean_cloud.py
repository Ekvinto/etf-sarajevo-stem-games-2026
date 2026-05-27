"""
clean_cloud.py
--------------
Post-process a point cloud (PLY) to remove outliers and (optionally) the
dominant ground plane.

Filters available (apply in this order, in the same run):
    1. Statistical Outlier Removal (SOR) - the standard cleanup.
       For each point, compute the mean distance to its k nearest neighbors.
       Globally, this distribution has a mean mu and std sigma.  Discard
       points where mean-dist > mu + std_ratio * sigma.
    2. Radius Outlier Removal (ROR) - kept as an option for very noisy data.
       Discard points with fewer than `min_neighbors` neighbors within
       `radius` world-units.
    3. Plane removal (RANSAC) - find the largest planar surface and remove
       its inliers.  Useful when the floor / ground / wall genuinely is
       noise for your purposes.  Off by default.

Usage:
    # Standard cleanup (recommended starting point)
    python clean_cloud.py output/entrance_track_c.ply

    # Aggressive cleanup
    python clean_cloud.py output/entrance_track_c.ply --k 30 --std-ratio 1.5

    # Also remove the dominant ground plane
    python clean_cloud.py output/box_track_c.ply --remove-plane

    # Custom output path
    python clean_cloud.py output/box_track_c.ply -o output/box_cleaned.ply
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import save_ply, save_xyz_txt   # noqa: E402
from track_a import read_ply_vertices       # noqa: E402


# ----------------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------------

def statistical_outlier_removal(points: np.ndarray, k: int = 20,
                                std_ratio: float = 2.0
                                ) -> np.ndarray:
    """Return a boolean keep-mask.

    For each point, finds the mean distance to its k nearest neighbors.
    Globally this gives a distribution with mean mu and std sigma.
    Points are kept iff mean_dist <= mu + std_ratio * sigma.
    """
    n = len(points)
    if n < k + 1:
        # Too few points to do statistics, keep everything.
        return np.ones(n, dtype=bool)

    print(f"  SOR: building KD-tree on {n:,} points ...")
    tree = cKDTree(points)
    # k+1 because the closest neighbor of each point is itself (distance 0)
    print(f"  SOR: querying {k+1} nearest neighbors ...")
    dists, _ = tree.query(points, k=k + 1, workers=-1)
    # Skip the self-distance at column 0
    mean_dists = dists[:, 1:].mean(axis=1)

    mu = float(mean_dists.mean())
    sigma = float(mean_dists.std())
    threshold = mu + std_ratio * sigma
    keep = mean_dists <= threshold
    print(f"  SOR: mean k-NN dist = {mu:.3f}, std = {sigma:.3f}, "
          f"threshold = {threshold:.3f}")
    print(f"  SOR: kept {int(keep.sum()):,} / {n:,} "
          f"({100.0*keep.sum()/n:.1f}%)")
    return keep


def radius_outlier_removal(points: np.ndarray, radius: float,
                           min_neighbors: int) -> np.ndarray:
    """Return a boolean keep-mask.

    Discards any point with fewer than `min_neighbors` neighbors within
    `radius`.  (Self is included in the count, so a min_neighbors of 5
    means at least 4 OTHER neighbors must be within `radius`.)
    """
    n = len(points)
    print(f"  ROR: querying neighbors within r={radius} ...")
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, r=radius, workers=-1,
                                   return_length=True)
    counts = np.asarray(counts, dtype=np.int64)
    keep = counts >= min_neighbors
    print(f"  ROR: kept {int(keep.sum()):,} / {n:,} "
          f"({100.0*keep.sum()/n:.1f}%)")
    return keep


def fit_plane_ransac(points: np.ndarray,
                     distance_threshold: float,
                     iterations: int = 2000,
                     seed: int = 0,
                     ) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC-fit the dominant plane to a point cloud.

    Returns:
        normal:    (3,) unit normal of the plane
        d:         scalar offset, so plane is { x | normal . x + d == 0 }
        inliers:   boolean mask of points within `distance_threshold` of plane
    """
    n = len(points)
    if n < 3:
        raise ValueError("Need at least 3 points to fit a plane")

    rng = np.random.default_rng(seed)
    best_count = -1
    best_normal = np.zeros(3)
    best_d = 0.0

    pts = points.astype(np.float64)
    for _ in range(iterations):
        idx = rng.choice(n, 3, replace=False)
        p1, p2, p3 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        nrm = np.linalg.norm(normal)
        if nrm < 1e-9:
            continue
        normal = normal / nrm
        d = -float(normal @ p1)

        dists = np.abs(pts @ normal + d)
        count = int((dists < distance_threshold).sum())
        if count > best_count:
            best_count = count
            best_normal = normal
            best_d = d

    inliers = np.abs(pts @ best_normal + best_d) < distance_threshold
    return best_normal, best_d, inliers


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Path to input .ply (e.g. output/box_track_c.ply)")
    p.add_argument("-o", "--output", default=None,
                   help="Output .ply path.  Default: <input>_cleaned.ply alongside input.")

    # SOR
    p.add_argument("--no-sor", action="store_true",
                   help="Skip statistical outlier removal.")
    p.add_argument("--k", type=int, default=20,
                   help="Number of neighbors for SOR (default 20).")
    p.add_argument("--std-ratio", type=float, default=2.0,
                   help="Std-dev multiplier for SOR threshold.  Lower = more "
                        "aggressive (default 2.0).")

    # ROR
    p.add_argument("--radius", type=float, default=0.0,
                   help="Enable radius outlier removal with this radius "
                        "(in world units).  0 = disabled (default).")
    p.add_argument("--min-neighbors", type=int, default=5,
                   help="Minimum neighbors within --radius for ROR (default 5).")

    # Plane removal
    p.add_argument("--remove-plane", action="store_true",
                   help="RANSAC-fit and remove the largest planar surface "
                        "(e.g. floor).  Off by default - the floor is usually "
                        "real scene geometry, not noise.")
    p.add_argument("--plane-thresh", type=float, default=2.0,
                   help="Distance threshold for RANSAC plane inliers, in "
                        "world units (default 2.0).")
    p.add_argument("--plane-iters", type=int, default=2000,
                   help="RANSAC iterations for plane fit (default 2000).")
    p.add_argument("--plane-min-ratio", type=float, default=0.15,
                   help="Only remove the plane if it represents at least this "
                        "fraction of all points (default 0.15).  Guards against "
                        "removing legitimate structure when there is no real plane.")
    args = p.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        print(f"ERROR: {in_path} not found.")
        sys.exit(1)
    if args.output is None:
        out_path = in_path.parent / (in_path.stem + "_cleaned.ply")
    else:
        out_path = Path(args.output).resolve()

    print(f"Reading {in_path} ...")
    points, colors = read_ply_vertices(in_path)
    print(f"  -> {len(points):,} points "
          f"({'with' if colors is not None else 'no'} colors)")
    n_orig = len(points)

    if n_orig == 0:
        print("Empty cloud, nothing to do.")
        sys.exit(0)

    # ---- SOR
    if not args.no_sor:
        print(f"\n[1] Statistical Outlier Removal (k={args.k}, "
              f"std_ratio={args.std_ratio}) ...")
        t0 = time.time()
        mask = statistical_outlier_removal(points, k=args.k,
                                           std_ratio=args.std_ratio)
        points = points[mask]
        colors = colors[mask] if colors is not None else None
        print(f"  SOR took {time.time()-t0:.1f}s")
    else:
        print("\n[1] SOR skipped (--no-sor).")

    # ---- ROR
    if args.radius > 0:
        print(f"\n[2] Radius Outlier Removal (r={args.radius}, "
              f"min_neighbors={args.min_neighbors}) ...")
        t0 = time.time()
        mask = radius_outlier_removal(points, radius=args.radius,
                                      min_neighbors=args.min_neighbors)
        points = points[mask]
        colors = colors[mask] if colors is not None else None
        print(f"  ROR took {time.time()-t0:.1f}s")
    else:
        print("\n[2] ROR skipped (--radius 0).")

    # ---- Plane removal
    if args.remove_plane:
        print(f"\n[3] Plane removal (RANSAC, thresh={args.plane_thresh}, "
              f"iters={args.plane_iters}) ...")
        t0 = time.time()
        normal, d, plane_mask = fit_plane_ransac(
            points,
            distance_threshold=args.plane_thresh,
            iterations=args.plane_iters,
        )
        ratio = plane_mask.sum() / max(1, len(points))
        print(f"  Plane normal: {normal.round(3)}, offset: {d:.2f}")
        print(f"  Plane inliers: {int(plane_mask.sum()):,} / {len(points):,} "
              f"({100*ratio:.1f}%)")
        if ratio < args.plane_min_ratio:
            print(f"  Plane covers < {args.plane_min_ratio*100:.0f}% of "
                  f"points; NOT removing (looks like the plane was a "
                  f"coincidence, not real structure).")
        else:
            keep = ~plane_mask
            points = points[keep]
            colors = colors[keep] if colors is not None else None
            print(f"  Removed plane.  Remaining: {len(points):,}")
        print(f"  Plane removal took {time.time()-t0:.1f}s")
    else:
        print("\n[3] Plane removal skipped (no --remove-plane).")

    # ---- Save
    if colors is not None:
        save_ply(out_path, points, colors.astype(np.uint8), binary=True)
        save_xyz_txt(out_path.with_suffix(".txt"), points,
                     colors.astype(np.uint8))
    else:
        save_ply(out_path, points, None, binary=True)
        save_xyz_txt(out_path.with_suffix(".txt"), points)

    n_final = len(points)
    print(f"\n=== Done ===")
    print(f"  Input : {n_orig:,} points")
    print(f"  Output: {n_final:,} points  ({100*n_final/n_orig:.1f}% kept)")
    print(f"  PLY -> {out_path}")
    print(f"  TXT -> {out_path.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
