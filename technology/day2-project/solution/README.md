# STEM Games Day 2 — Multi-View 3D Reconstruction

Project layout (flat, everything at project root):

```
STEM-GAMES-TECHNOLOGY/
├── data/
│   ├── Box/          # 12 PNGs + boxInput.txt
│   ├── Entrance/     # 12 PNGs + entranceInput.txt
│   ├── Statue/       # 18 PNGs + K.txt
│   └── Fountain/     # 11 JPGs + K.txt
├── output/                  # created automatically when you run scripts
├── utils.py                 # parsing, coord conversion, IO
├── track_c.py               # OpenCV sparse triangulation (Box, Entrance)
├── track_a.py               # COLMAP driver (all 4 datasets)
├── visualize.py             # quick PLY viewer
├── track_b_colab.ipynb      # MASt3R notebook for Google Colab
├── requirements.txt
└── README.md  (this file)
```

Note: Box and Entrance don't have their own `K.txt`. The code uses a built-in default `K = [[960,0,960],[0,960,540],[0,0,1]]` for them, which matches the given calibration.

---

## 1. Setup

You should already have:
- Python 3.13 venv inside the PyCharm project (check the bottom-right corner of PyCharm).
- COLMAP installed and on PATH — verify by running `colmap -h` in a terminal.
- The input image sets copied into `data/` (see section 1.1 below).

### 1.1 Get the input data

The `data/` folder is **not stored in the repository** — the image sets are
kept once, under `../task-description/TestImages/`. Before running anything,
copy the four image-set folders into `solution/data/`.

From inside the `solution/` folder:

```bat
:: Windows
xcopy /E /I "..\task-description\TestImages" "data"
```

```bash
# macOS / Linux
mkdir -p data && cp -r ../task-description/TestImages/* data/
```

After this, `data/` should contain `Box/`, `Entrance/`, `Statue/`, and
`Fountain/`, exactly as in the layout above.

Install Python dependencies. In the PyCharm terminal (which should have your venv active):

```bat
pip install -r requirements.txt
```

If `open3d` fails on Python 3.13, install everything else and use the matplotlib fallback for visualization:

```bat
pip install numpy scipy opencv-contrib-python Pillow tqdm matplotlib
```

After install, the red "No module named 'numpy'" errors PyCharm shows in the editor will disappear.

---

## 2. Run order

Run each script from inside the `solution/` folder (`technology/day2-project/solution/` in this repository).

### 2.1 Sanity-check the foundation

```bat
python utils.py data/Box/boxInput.txt data/Box/K.txt
```

`data/Box/K.txt` doesn't exist — that's intentional, the code falls back to its default K. But the utility's CLI requires a K path; pass `data/Statue/K.txt` instead since the calibration is identical:

```bat
python utils.py data/Box/boxInput.txt data/Statue/K.txt
python utils.py data/Entrance/entranceInput.txt data/Statue/K.txt
```

You should see all cameras print pass lines ending in: **`All cameras passed the projection sanity check.`** If this fails the rest won't work — stop and investigate.

### 2.2 Track C: OpenCV triangulation on Box & Entrance

Smoke test first with a small image subset:

```bat
python track_c.py Box --max-images 4
```

If that finishes in ~30 s and prints `Final point cloud: ... points`, you're good. Then the full run:

```bat
python track_c.py Box
python track_c.py Entrance
```

Each takes a couple of minutes. Outputs:

```
output/box_track_c.ply
output/box_track_c.txt
output/entrance_track_c.ply
output/entrance_track_c.txt
```

### 2.3 Track A: COLMAP on all four datasets

Sparse only (fast):

```bat
python track_a.py all
```

Per-dataset time on CPU (rough): Box/Entrance ~5 min each, Statue ~10–15 min, Fountain ~15–30 min (high res).

Outputs:
```
output/box_track_a_sparse.ply
output/entrance_track_a_sparse.ply
output/statue_track_a_sparse.ply
output/fountain_track_a_sparse.ply
```

Optional dense (slow on CPU — hours, run overnight):

```bat
python track_a.py Box --dense
```

### 2.4 Track B: MASt3R via Google Colab (only for Statue & Fountain)

Open the `track_b_colab_*.ipynb` notebook in Google Colab and pick a GPU runtime (Runtime → Change runtime type). The team's competition runs used an **NVIDIA A100** GPU on Colab; a T4 also works but is slower. Run all cells. The notebook will:
1. Clone the MASt3R repo and download weights.
2. Ask you to upload a zip of the dataset (`Statue.zip` or `Fountain.zip`).
3. Run inference and offer the resulting `.ply` for download.

Save downloaded files as `output/statue_track_b.ply` and `output/fountain_track_b.ply`.

### 2.5 Visualize

```bat
python visualize.py output/box_track_c.ply
python visualize.py output/entrance_track_c.ply --backend mpl
python visualize.py output/statue_track_a_sparse.ply --backend stats
```

Or simply double-click the `.ply` files to open them in MeshLab.

---

## 3. Final submission selection

For each dataset, pick the best-looking cloud:

| Dataset  | Best option (usually)                                          |
|----------|----------------------------------------------------------------|
| Box      | `box_track_c.ply` (uses known poses; should be very clean)     |
| Entrance | `entrance_track_c.ply`                                         |
| Statue   | `statue_track_a_sparse.ply` (or `_track_b.ply` if Colab worked)|
| Fountain | `fountain_track_a_sparse.ply` (or `_track_b.ply`)              |

Rename or copy the chosen file as `<dataset>.txt` for submission.

---

## 4. Troubleshooting

**"COLMAP not found on PATH"** — Either close and reopen your terminal after editing PATH, or pass `--colmap-exe "C:/Tools/COLMAP/colmap.bat"` explicitly.

**"No images matching pattern"** — Check that images are named `box1.png`, `box2.png`, etc., and live directly inside `data/Box/`.

**"Sanity check failed for camera N"** — A rotation vector in the input txt file may be malformed. Look at the camera index and inspect that block in the input file.

**Track C produces very few points** — Loosen thresholds:
```bat
python track_c.py Box --reproj-threshold 8 --ratio-test 0.85 --voxel-size 1
```

**COLMAP mapper produces no model** — The dataset may have insufficient overlap. Inspect `output/colmap_<name>/logs/mapper.log`. Try fewer images or `--Mapper.init_min_num_inliers 30`.

**Open3D doesn't install** — Use `--backend mpl` in `visualize.py`, or just open `.ply` files in MeshLab.
