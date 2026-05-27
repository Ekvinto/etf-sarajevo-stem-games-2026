# Technology — Day 2: Photogrammetry Project

## The problem

Given four sets of photographs (each set showing one scene from slightly
different positions and angles), estimate the 3D coordinates — `(X, Y, Z)` —
of as many pixels as possible. In industry this is called a **point cloud**.

- **Box** and **Entrance** — camera positions and orientations are *known*
  (provided as forward / right / up vectors). These cases are easier.
- **Statue** and **Fountain** — camera poses are *unknown*, making the problem
  considerably harder. (Fountain images are 2048×3072; the rest are 1920×1080.)

The expected submission is, for each image set, a text file listing the
estimated 3D points — ideally with a visualization.

## Result

This project scored **400 / 600**.

## Contents

| Folder | What's inside |
|--------|---------------|
| `task-description/` | The official Day 2 project brief. |
| `solution/` | The team's code and generated output (estimated point clouds, visualizations). |
