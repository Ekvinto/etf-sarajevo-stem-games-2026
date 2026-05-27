# ETF Sarajevo — STEM Games 2026

A complete archive of the **Faculty of Electrical Engineering, University of
Sarajevo (ETF)** team's work at **STEM Games 2026** — problem statements,
source code, solutions, reports, and presentations from the competition arenas.

The purpose of this repository is to serve as a **study and training resource**
for future ETF Sarajevo teams. Instead of starting from zero each year, next
year's competitors can use this archive to understand the kinds of problems
STEM Games poses and how one team approached them.

## Arenas & results

| Arena | Result | Contents |
|-------|--------|----------|
| Mathematics | 1st place | Bot-detection project, report, presentation |
| Technology  | 6th place | Day 1 competitive programming + Day 2 photogrammetry project |
| Engineering | participated | 20+ short interactive engineering tasks |
| Science     | — | Task description and reference papers (provided by the University of Osijek) |

For the Technology arena, the Day 2 project scored 400/600; the Day 1
algorithmic round contributed a smaller share of the total.

## Repository structure

```
etf-sarajevo-stem-games-2026/
├── mathematics/                       Mathematics arena — 1st place
│   ├── task-description/              official assignment
│   ├── projects/
│   │   ├── romance-scam-detection-abandoned/  early prototype (abandoned — no dataset)
│   │   └── ragebait-comment-detection/        final submitted project
│   ├── report/                        written report(s) + LaTeX sources
│   └── presentation/                  Day 3 presentation slides
│
├── technology/                        Technology arena — 6th place
│   ├── day1-leetcode/                 Day 1 — competitive programming round
│   │   ├── problem-statements/        problem statements (PDF)
│   │   ├── solutions-competition/     solutions written during the contest
│   │   ├── solutions-ai-postcomp/     AI-assisted solutions added afterwards
│   │   ├── testcases/                 grading test clusters (inputs/outputs)
│   │   └── scripts/                   helper script to fetch test data
│   └── day2-project/                  Day 2 — photogrammetry / 3D point cloud
│       ├── task-description/
│       └── solution/                  team code + generated output
│
├── engineering/                       Engineering arena
│   ├── task-description/
│   ├── tasks/                         individual task screenshots (20+)
│   └── solutions-ai-postcomp/         AI-assisted solutions added afterwards
│
└── science/                           Science arena
    ├── task-description/              official assignment
    └── references/                    supporting research papers
```

## The arenas

### Mathematics — *"Bots and how to catch them"*

Teams chose a type of internet bot and designed a method to detect it. Our
team explored two directions: an early **romance-scam detector** (abandoned
because no suitable labelled dataset was available) and the **final
submission**, a detector for AI-generated engagement-baiting ("ragebait")
comments. The report covers the work; the Day 3 presentation is in
`presentation/`.

### Technology

Two parts:

- **Day 1 — competitive programming.** Four problems: Undistortion, Largest
  Triangle, Gaussian Blur, and Find the Square. Statements are in
  `day1-leetcode/problem-statements/`.
- **Day 2 — photogrammetry project.** Given several photographs of a scene
  taken from slightly different positions, estimate the 3D coordinates of as
  many pixels as possible (a *point cloud*). Two image sets came with known
  camera poses, two without. See `day2-project/`.

### Engineering

A series of 20+ short, interactive tasks spanning multiple engineering
disciplines — control theory, signal processing, structural mechanics,
thermodynamics, and more (PID tuning, Kalman filtering, Fourier analysis,
Mohr's circle, beam deflection, etc.). Task screenshots are in
`engineering/tasks/`.

### Science

The Science arena task description and accompanying reference papers were
provided by the **Department of Physics, University of Osijek** (*Odsjek za
Fiziku, Univerzitet u Osijeku*). See `science/` for details and a full
acknowledgement.

## A note on AI-assisted solutions

Some solutions in this repository were produced **after the competition** with
the help of AI tools, purely as reference and learning material. These live in
folders named **`solutions-ai-postcomp/`** and were **not** part of the team's
competition submissions.

Work actually done by the team under competition conditions is kept separately
— for example in `solutions-competition/` — and labelled as such. This
separation is deliberate: the archive should make it obvious which work was
done during the competition and which was added later for study.

## Using this archive (for future teams)

- Read each arena's `task-description/` to see the kinds of problems to expect.
- Study the competition solutions and reports to see what a submission looks
  like under time pressure.
- Use the `solutions-ai-postcomp/` material as a worked reference — but try the
  problems yourself first.

## Contributing

This repository documents the 2026 competition. Teammates and future students
are welcome to add missing materials, improve solutions, or add explanatory
notes. If you are preparing for a later edition of STEM Games, treat this as
reference material and consider starting a fresh
`etf-sarajevo-stem-games-YYYY` repository for your own year.
