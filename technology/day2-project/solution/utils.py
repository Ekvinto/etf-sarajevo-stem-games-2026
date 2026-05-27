"""
utils.py
--------
Foundation utilities for the STEM Games Day 2 multi-view reconstruction task.

Contains:
  * CameraPose dataclass  - holds an Unreal-style camera (position + Forward/Right/Up).
  * parse_cameras_file()  - robust parser for boxInput.txt / entranceInput.txt.
  * load_K()              - reads the K.txt matrix files.
  * pose_to_opencv()      - converts Unreal (X-fwd, Y-right, Z-up, left-handed)
                            into OpenCV (X-right, Y-down, Z-fwd, right-handed).
  * build_projection_matrix() - K @ [R | t].
  * project_point()       - sanity-check helper.
  * save_ply() / save_xyz_txt() - submission writers.
  * sanity_check_pose()   - validates the coordinate-conversion math.

Run `python utils.py path/to/boxInput.txt` to see a quick self-test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------

@dataclass
class CameraPose:
    """An Unreal-style camera pose, exactly as given in the input txt files.

    Position is in world units. Forward/Right/Up are unit vectors in world
    coordinates describing how the camera is oriented.
    """
    index: int                # 1-based index, matches the txt enumeration
    position: np.ndarray      # shape (3,)
    forward: np.ndarray       # shape (3,) - unit vector
    right:   np.ndarray       # shape (3,) - unit vector
    up:      np.ndarray       # shape (3,) - unit vector

    def __post_init__(self):
        # Defensive: ensure float arrays of length 3
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.forward  = np.asarray(self.forward,  dtype=np.float64).reshape(3)
        self.right    = np.asarray(self.right,    dtype=np.float64).reshape(3)
        self.up       = np.asarray(self.up,       dtype=np.float64).reshape(3)


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------

# Match  KEY  Xnnn  Ynnn  Znnn   where:
#   * KEY is one of CamPosition/CamForward/CamRight/CamUp
#   * the colon after KEY is OPTIONAL  (Box entry 5 omits it)
#   * any whitespace (spaces, tabs) between fields
#   * numbers are floats, possibly negative
_FLOAT = r"-?\d+(?:\.\d+)?"
_KV_RE = re.compile(
    r"(?P<key>CamPosition|CamForward|CamRight|CamUp)\s*:?"  # KEY + optional colon
    r"\s*X\s*=\s*(?P<x>" + _FLOAT + r")"
    r"\s+Y\s*=\s*(?P<y>" + _FLOAT + r")"
    r"\s+Z\s*=\s*(?P<z>" + _FLOAT + r")",
    re.IGNORECASE,
)

# Match the numeric header like   "1)" or "12)"
_HEADER_RE = re.compile(r"^\s*(\d+)\s*\)\s*$")


def parse_cameras_file(path: str | Path) -> list[CameraPose]:
    """Parse a boxInput.txt / entranceInput.txt style file.

    Robust to:
      * variable whitespace (tabs vs spaces)
      * missing colon after the KEY (Box entry 5 has 'CamRight  X=...' with no ':')
      * extra blank lines inside an entry (Entrance entry 3)
      * leading lines like 'camera field of view is always 90 degrees'

    Returns a list of CameraPose objects in file order.
    """
    text = Path(path).read_text(encoding="utf-8")

    # Strategy: walk line by line.  When we see a "N)" header, start a new
    # block; collect everything until the next header (or EOF).  Then regex-
    # extract the four expected keys from that block.
    blocks: list[tuple[int, str]] = []
    current_idx: int | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        m = _HEADER_RE.match(raw_line)
        if m:
            if current_idx is not None:
                blocks.append((current_idx, "\n".join(current_lines)))
            current_idx = int(m.group(1))
            current_lines = []
        else:
            if current_idx is not None:
                current_lines.append(raw_line)
    if current_idx is not None:
        blocks.append((current_idx, "\n".join(current_lines)))

    cameras: list[CameraPose] = []
    _CANONICAL = {
        "camposition": "CamPosition",
        "camforward":  "CamForward",
        "camright":    "CamRight",
        "camup":       "CamUp",
    }
    for idx, body in blocks:
        kv: dict[str, np.ndarray] = {}
        for m in _KV_RE.finditer(body):
            key_canon = _CANONICAL[m.group("key").lower()]
            vec = np.array([float(m.group("x")),
                            float(m.group("y")),
                            float(m.group("z"))], dtype=np.float64)
            kv[key_canon] = vec

        required = {"CamPosition", "CamForward", "CamRight", "CamUp"}
        missing = required - set(kv.keys())
        if missing:
            raise ValueError(
                f"Camera block {idx} is missing keys {sorted(missing)}.\n"
                f"Block contents were:\n{body}"
            )

        cameras.append(CameraPose(
            index=idx,
            position=kv["CamPosition"],
            forward=kv["CamForward"],
            right=kv["CamRight"],
            up=kv["CamUp"],
        ))

    if not cameras:
        raise ValueError(f"No camera blocks were found in {path}.")

    # Sanity: indices should be 1..N and unique
    seen = {c.index for c in cameras}
    if len(seen) != len(cameras):
        raise ValueError(f"Duplicate camera indices in {path}: {[c.index for c in cameras]}")

    return cameras


def load_K(path: str | Path) -> np.ndarray:
    """Load a 3x3 intrinsic matrix from a K.txt file.

    Accepts both the bracketed format we were given:
        K = [960.00 0 960.00;
             0 960.00 540.00;
             0 0 1]
    and a plain whitespace-separated 3x3.
    """
    text = Path(path).read_text(encoding="utf-8")

    # Strip 'K = [' prefix and trailing ']' if present
    text = text.replace("K", " ").replace("=", " ").replace("[", " ").replace("]", " ")
    # Replace semicolons with newlines so each row is a line
    text = text.replace(";", "\n")

    rows: list[list[float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        nums = re.findall(_FLOAT, line)
        if not nums:
            continue
        rows.append([float(x) for x in nums])

    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        raise ValueError(f"Could not parse a 3x3 matrix from {path}.  Got rows: {rows}")

    return np.array(rows, dtype=np.float64)


# ----------------------------------------------------------------------------
# Coordinate-system conversion: Unreal -> OpenCV
# ----------------------------------------------------------------------------
#
# Unreal Engine (left-handed):
#     X = Forward, Y = Right, Z = Up
# The given CamForward/CamRight/CamUp vectors are expressed in WORLD coords
# and describe the camera's local axes.
#
# OpenCV camera (right-handed):
#     X = Right (in image), Y = Down (in image), Z = Forward (into scene)
#
# Mapping from Unreal-camera-local to OpenCV-camera-local:
#     OpenCV_X = Unreal_Right
#     OpenCV_Y = -Unreal_Up      (image-down is negative of world-up)
#     OpenCV_Z = Unreal_Forward
#
# Therefore the world-to-camera rotation R (OpenCV) has the camera axes,
# expressed in world coords, as its ROWS:
#     R = [  CamRight  ]
#         [ -CamUp     ]
#         [  CamForward]
# and translation:
#     t = -R @ CamPosition
#
# Verification: for a point d*CamForward in front of the camera,
#     P_world - CamPosition = d * CamForward
#     R @ (d * CamForward) = ( Right . CamForward,
#                              -Up   . CamForward,
#                               Forward . CamForward ) * d
#                          = (0, 0, d)
# i.e. directly in front along OpenCV's +Z.  Correct.

def pose_to_opencv(cam: CameraPose) -> tuple[np.ndarray, np.ndarray]:
    """Convert an Unreal-style CameraPose to an OpenCV (R, t) world-to-camera pair.

    Returns:
        R: (3, 3) rotation matrix
        t: (3,)   translation vector
    """
    R = np.stack([cam.right, -cam.up, cam.forward], axis=0)   # (3, 3)
    # Renormalise rows to guard against floating-point drift in the inputs
    # (the txt files round to 3 decimals)
    for i in range(3):
        n = np.linalg.norm(R[i])
        if n < 1e-9:
            raise ValueError(f"Camera {cam.index} has a degenerate basis vector.")
        R[i] = R[i] / n
    t = -R @ cam.position
    return R, t


def build_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a 3x4 projection matrix  P = K @ [R | t]."""
    Rt = np.zeros((3, 4), dtype=np.float64)
    Rt[:, :3] = R
    Rt[:, 3]  = t
    return K @ Rt


# ----------------------------------------------------------------------------
# Projection helpers
# ----------------------------------------------------------------------------

def project_point(P: np.ndarray, X_world: np.ndarray) -> np.ndarray:
    """Project a (3,) or (N,3) world point through a 3x4 P matrix.

    Returns pixel coordinates of shape (2,) or (N, 2).  Points behind the
    camera (z <= 0 in camera frame) come back with NaNs.
    """
    X = np.asarray(X_world, dtype=np.float64)
    single = X.ndim == 1
    if single:
        X = X[None, :]
    X_h = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)   # (N,4)
    x_h = X_h @ P.T                                                # (N,3)
    z = x_h[:, 2]
    valid = z > 1e-9
    pixels = np.full((X.shape[0], 2), np.nan, dtype=np.float64)
    pixels[valid, 0] = x_h[valid, 0] / z[valid]
    pixels[valid, 1] = x_h[valid, 1] / z[valid]
    return pixels[0] if single else pixels


def reprojection_error(P: np.ndarray, X_world: np.ndarray,
                       pixel_obs: np.ndarray) -> np.ndarray:
    """L2 pixel error for each point.  Returns shape (N,)."""
    proj = project_point(P, X_world)
    err = np.linalg.norm(proj - pixel_obs, axis=-1)
    return err


# ----------------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------------

def save_ply(path: str | Path,
             points: np.ndarray,
             colors: np.ndarray | None = None,
             binary: bool = True) -> None:
    """Write a PLY point cloud.

    Args:
        path:    output filename
        points:  (N, 3) float array of XYZ
        colors:  optional (N, 3) uint8 array of RGB
        binary:  True for binary_little_endian, False for ascii
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3); got {points.shape}")
    N = points.shape[0]

    has_color = colors is not None
    if has_color:
        colors = np.asarray(colors, dtype=np.uint8)
        if colors.shape != (N, 3):
            raise ValueError(f"colors must be (N,3); got {colors.shape}")

    header_lines = [
        "ply",
        f"format {'binary_little_endian' if binary else 'ascii'} 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header_lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header_lines.append("end_header")
    header = ("\n".join(header_lines) + "\n").encode("ascii")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if binary:
        with open(path, "wb") as f:
            f.write(header)
            if has_color:
                # interleave xyz (float32) and rgb (uint8) into one structured array
                dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                               ("r", "u1"), ("g", "u1"), ("b", "u1")])
                arr = np.empty(N, dtype=dt)
                arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
                arr["r"], arr["g"], arr["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
                f.write(arr.tobytes())
            else:
                f.write(points.astype("<f4").tobytes())
    else:
        with open(path, "w", encoding="ascii") as f:
            f.write(header.decode("ascii"))
            if has_color:
                for p, c in zip(points, colors):
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                            f"{int(c[0])} {int(c[1])} {int(c[2])}\n")
            else:
                for p in points:
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def save_xyz_txt(path: str | Path, points: np.ndarray,
                 colors: np.ndarray | None = None) -> None:
    """Write the submission text file.

    Format: one point per line, "X Y Z" (optionally "X Y Z R G B").
    """
    points = np.asarray(points, dtype=np.float64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if colors is None:
        np.savetxt(path, points, fmt="%.4f")
    else:
        colors = np.asarray(colors, dtype=np.int32)
        combined = np.concatenate([points, colors], axis=1)
        # 3 float columns, 3 int columns
        np.savetxt(path, combined,
                   fmt=["%.4f", "%.4f", "%.4f", "%d", "%d", "%d"])


# ----------------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------------

def sanity_check_pose(cam: CameraPose, K: np.ndarray,
                      img_size_wh: tuple[int, int] = (1920, 1080),
                      verbose: bool = True) -> dict:
    """Project a point known to be in front of the camera and verify it lands
    at the image centre.  Also project two off-axis points and check left/right
    + up/down behaviour.

    Returns a dict of diagnostic info.  Raises if anything is clearly wrong.
    """
    R, t = pose_to_opencv(cam)
    P    = build_projection_matrix(K, R, t)
    w, h = img_size_wh

    # 1) Point 100 units in front of the camera should land near image centre.
    p_front  = cam.position + 100.0 * cam.forward
    px_front = project_point(P, p_front)

    # 2) Point in front + a bit to the right should land right of centre.
    p_right  = cam.position + 100.0 * cam.forward + 20.0 * cam.right
    px_right = project_point(P, p_right)

    # 3) Point in front + a bit up should land ABOVE centre (smaller v).
    p_up     = cam.position + 100.0 * cam.forward + 20.0 * cam.up
    px_up    = project_point(P, p_up)

    centre = np.array([K[0, 2], K[1, 2]])
    info = {
        "camera_index": cam.index,
        "px_front": px_front,
        "px_right": px_right,
        "px_up":    px_up,
        "centre":   centre,
    }

    # Tolerance is generous: the txt files only have 3-decimal precision in
    # the rotation vectors, which introduces ~1-2 px of noise after row
    # renormalisation.  This is below the SIFT-matching uncertainty anyway.
    err_front = np.linalg.norm(px_front - centre)
    if err_front > 5.0:
        raise AssertionError(
            f"Sanity check failed for camera {cam.index}: a point on the "
            f"forward ray should project near {centre}, got {px_front}."
        )
    if not (px_right[0] > centre[0] + 50.0):
        raise AssertionError(
            f"Sanity check failed for camera {cam.index}: a point on "
            f"+CamRight should project to the right of centre."
        )
    if not (px_up[1] < centre[1] - 50.0):
        raise AssertionError(
            f"Sanity check failed for camera {cam.index}: a point on "
            f"+CamUp should project above centre (smaller v)."
        )

    if verbose:
        print(f"  cam {cam.index:2d}: forward->{px_front.round(2)}  "
              f"+right->{px_right.round(2)}  +up->{px_up.round(2)}  "
              f"(centre {centre.round(2)})")
    return info


# ----------------------------------------------------------------------------
# Image enumeration helpers
# ----------------------------------------------------------------------------

def list_dataset_images(dataset_dir: str | Path, prefix: str) -> list[Path]:
    """Return image paths sorted by their numeric suffix.

    e.g. for prefix='box', returns box1.png, box2.png, ... box12.png in order
    regardless of filesystem listing order.
    """
    dataset_dir = Path(dataset_dir)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)\.(?:png|jpg|jpeg|PNG|JPG|JPEG)$")
    found: list[tuple[int, Path]] = []
    for p in dataset_dir.iterdir():
        m = pat.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda t: t[0])
    return [p for _, p in found]


# ----------------------------------------------------------------------------
# Self-test (run `python utils.py path/to/boxInput.txt path/to/K.txt`)
# ----------------------------------------------------------------------------

def _self_test(cameras_path: str, k_path: str) -> None:
    print(f"Parsing cameras from {cameras_path} ...")
    cams = parse_cameras_file(cameras_path)
    print(f"  -> {len(cams)} cameras parsed.")

    print(f"Loading K from {k_path} ...")
    K = load_K(k_path)
    print(f"  K =\n{K}")

    print("Running per-camera sanity check ...")
    for c in cams:
        sanity_check_pose(c, K)
    print("All cameras passed the projection sanity check.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("usage: python utils.py path/to/boxInput.txt path/to/K.txt")
        sys.exit(1)
    _self_test(sys.argv[1], sys.argv[2])
