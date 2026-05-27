"""
visualize.py
------------
Render or interactively view a PLY point cloud.

Two modes:
    --backend open3d     interactive 3D viewer (preferred)
    --backend mpl        matplotlib scatter plot (fallback when Open3D isn't
                         installable on your Python version)
    --backend stats      print only summary stats (always works)

Usage:
    python src/visualize.py output/box_track_c.ply
    python src/visualize.py output/box_track_c.ply --backend mpl --max-points 50000
    python src/visualize.py output/box_track_c.ply --backend stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_a import read_ply_vertices    # tiny PLY reader lives there       # noqa: E402


def show_stats(points: np.ndarray, colors: np.ndarray | None) -> None:
    if len(points) == 0:
        print("Empty point cloud.")
        return
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    pmean = points.mean(axis=0)
    extent = pmax - pmin
    print(f"# points : {len(points):,}")
    print(f"min      : ({pmin[0]:.2f}, {pmin[1]:.2f}, {pmin[2]:.2f})")
    print(f"max      : ({pmax[0]:.2f}, {pmax[1]:.2f}, {pmax[2]:.2f})")
    print(f"mean     : ({pmean[0]:.2f}, {pmean[1]:.2f}, {pmean[2]:.2f})")
    print(f"extent   : ({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f})")
    if colors is not None:
        print(f"colors   : present (uint8)")
    else:
        print(f"colors   : none")


def show_open3d(points: np.ndarray, colors: np.ndarray | None) -> None:
    import open3d as o3d  # imported lazily so the script still loads if not present
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(
            colors.astype(np.float64) / 255.0)
    print("Opening Open3D viewer.  Close the window to exit.")
    o3d.visualization.draw_geometries([pcd])


def show_mpl(points: np.ndarray, colors: np.ndarray | None,
             max_points: int) -> None:
    import matplotlib.pyplot as plt  # lazy

    if len(points) > max_points:
        idx = np.random.default_rng(0).choice(len(points), max_points,
                                              replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    c = colors / 255.0 if colors is not None else "tab:blue"
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=1.5, c=c, marker=".")
    # equal aspect (matplotlib hack)
    extents = points.max(axis=0) - points.min(axis=0)
    max_extent = float(extents.max()) or 1.0
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - max_extent / 2, mid[0] + max_extent / 2)
    ax.set_ylim(mid[1] - max_extent / 2, mid[1] + max_extent / 2)
    ax.set_zlim(mid[2] - max_extent / 2, mid[2] + max_extent / 2)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ply", help="Path to a .ply file.")
    p.add_argument("--backend", choices=["open3d", "mpl", "stats"],
                   default="open3d")
    p.add_argument("--max-points", type=int, default=80_000,
                   help="Subsample for matplotlib (default 80k).")
    args = p.parse_args()

    path = Path(args.ply)
    if not path.exists():
        print(f"ERROR: not found: {path}")
        sys.exit(1)

    points, colors = read_ply_vertices(path)
    print(f"Loaded {path.name}:")
    show_stats(points, colors)

    if args.backend == "stats":
        return
    if args.backend == "open3d":
        try:
            show_open3d(points, colors)
        except ImportError:
            print("\nOpen3D is not installed; falling back to matplotlib.\n"
                  "Install with:  pip install open3d   "
                  "(or use --backend mpl explicitly)\n")
            show_mpl(points, colors, args.max_points)
        return
    if args.backend == "mpl":
        show_mpl(points, colors, args.max_points)


if __name__ == "__main__":
    main()
