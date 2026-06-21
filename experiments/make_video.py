"""Render an animated MP4 of drone flights overlaid on the compliance landscape.

Run from the experiments/ directory:
    python make_video.py --speed 0.5
    python make_video.py --all-speeds --output coco_vs_promis_all.mp4
"""

import argparse
import ast
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image
from promis.geo import CartesianCollection

# ── Constants ──────────────────────────────────────────────────────────────────
XMIN, XMAX = -1.835, 1.835
YMIN, YMAX = -1.45, 1.45
CRASH_Z = 0.2  # metres — matches crash detection in coco.ipynb cell 32

# Plot area expanded 15 % beyond the image on each side, filled with dimgray
_XPAD = (XMAX - XMIN) * 0.15 / 2
_YPAD = (YMAX - YMIN) * 0.15 / 2
PLOT_XMIN, PLOT_XMAX = XMIN - _XPAD, XMAX + _XPAD
PLOT_YMIN, PLOT_YMAX = YMIN - _YPAD, YMAX + _YPAD

GROUPS = [
    {"name": "CoCo (Logic + Doubt)", "tag": "coco-fixed", "color": "#2ecc71"},
    {"name": "Baseline",           "tag": "promis",     "color": "#e74c3c"},
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_flights(pattern: str) -> list[dict]:
    flights = []
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path)
        crash_mask = df["stateEstimateZ"] < CRASH_Z
        crash_idx = int(crash_mask.idxmax()) if crash_mask.any() else None
        if crash_idx is not None and crash_idx <= 10:  # skip takeoff noise
            crash_idx = None
        flights.append({"df": df, "crash_idx": crash_idx})
    return flights


def load_speed_data(speed: float, landscape_file: str | None = None) -> dict:
    groups = []
    for g in GROUPS:
        pattern = f"data/rosbags/{g['tag']}-{speed}/*.csv"
        flights = load_flights(pattern)
        if not flights:
            raise FileNotFoundError(f"No CSVs matched: {pattern}")
        groups.append({**g, "flights": flights})

    if landscape_file is None:
        landscape_file = f"data/drone-exp/original_landscape-speed-{speed:.2f}.pkl"
    landscape = CartesianCollection.load(landscape_file)
    interpolator = landscape.get_interpolator("hybrid")
    xs = np.linspace(XMIN * 1000, XMAX * 1000, 300)
    ys = np.linspace(YMIN * 1000, YMAX * 1000, 240)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    compliance_grid = interpolator(pts).reshape(240, 300)

    # Pad to full plot area with 1.0 (fully compliant → blue) so the margin
    # blends identically to a compliance = 1 region inside the landscape
    _pad_cols = round(300 * _XPAD / (XMAX - XMIN))
    _pad_rows = round(240 * _YPAD / (YMAX - YMIN))
    compliance_grid = np.pad(
        compliance_grid,
        ((_pad_rows, _pad_rows), (_pad_cols, _pad_cols)),
        constant_values=1.0,
    )

    data_hz = 50.0
    sample_df = groups[0]["flights"][0]["df"]
    if "time" in sample_df.columns:
        data_hz = len(sample_df) / (sample_df["time"].iloc[-1] - sample_df["time"].iloc[0])

    return {"speed": speed, "groups": groups, "grid": compliance_grid, "data_hz": data_hz}


# ── Figure helpers ─────────────────────────────────────────────────────────────

# Colors for the three speeds in the website card video (cool → warm = slow → fast)
SPEED_COLORS = {0.2: "#74b9ff", 0.5: "#fdcb6e", 1.0: "#e17055"}

# Neutral dark background and semantic fill colors for the bounding-box scene
WEBSITE_BG = "#1e1e1e"
BBOX_FILL = {"red": "#e74c3c", "blue": "#3498db", "yellow": "#f1c40f", "green": "#2ecc71"}

# linearized_image.png pixel dimensions — used for px → metre conversion
_IMG_W, _IMG_H = 1228, 1000


def load_bounding_boxes() -> list[dict]:
    """Return deduplicated oriented polygons in metre coordinates."""
    df = pd.read_csv("vision/bounding_boxes.csv")
    df = df.drop_duplicates(subset="poly_oriented")
    boxes = []
    for _, row in df.iterrows():
        poly_px = ast.literal_eval(row["poly_oriented"])
        poly_m = [
            (XMIN + (px / _IMG_W) * (XMAX - XMIN),
             YMAX - (py / _IMG_H) * (YMAX - YMIN))
            for px, py in poly_px
        ]
        boxes.append({"poly": np.array(poly_m), "color": row["color"]})
    return boxes


def setup_background(ax, boxes: list[dict]) -> None:
    """Neutral background with semantic bounding-box polygons, no aerial image."""
    ax.set_facecolor(WEBSITE_BG)
    for box in boxes:
        ax.add_patch(MplPolygon(
            box["poly"], closed=True,
            facecolor=BBOX_FILL.get(box["color"], "#888888"),
            edgecolor="none", alpha=0.75, zorder=1,
        ))
    ax.set_xlim(PLOT_XMIN, PLOT_XMAX)
    ax.set_ylim(PLOT_YMIN, PLOT_YMAX)
    ax.set_xlabel("x / m", fontsize=15)
    ax.tick_params(labelsize=13)


def setup_panel(ax, speed: float, compliance_grid: np.ndarray, scene_image) -> None:
    ax.set_facecolor("dimgray")
    ax.imshow(
        scene_image.convert("L"),
        cmap="gray",
        alpha=0.55,
        extent=(XMIN, XMAX, YMIN, YMAX),
        origin="upper",
        rasterized=True,
        zorder=1,
    )
    ax.imshow(
        compliance_grid,
        extent=(PLOT_XMIN, PLOT_XMAX, PLOT_YMIN, PLOT_YMAX),
        origin="lower",
        cmap="coolwarm_r",
        vmin=0,
        vmax=1,
        alpha=0.45,
        zorder=2,
    )
    ax.set_xlim(PLOT_XMIN, PLOT_XMAX)
    ax.set_ylim(PLOT_YMIN, PLOT_YMAX)
    ax.set_xlabel("x / m", fontsize=15)
    ax.tick_params(labelsize=13)
    # ax.set_title(f"v = {speed} m/s", fontsize=12)


def build_artists(ax, groups: list[dict]) -> list[dict]:
    artists = []
    for g in groups:
        for flight in g["flights"]:
            df = flight["df"]
            crash_idx = flight["crash_idx"]
            xs = df["stateEstimateX"].values
            ys = df["stateEstimateY"].values

            (trail_line,) = ax.plot([], [], color=g["color"], alpha=0.35, lw=1.5, zorder=4)
            (dot,) = ax.plot(
                [], [], "o",
                color=g["color"],
                markersize=9,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=6,
            )

            crash_marker = None
            if crash_idx is not None:
                (crash_marker,) = ax.plot(
                    xs[crash_idx], ys[crash_idx],
                    "x",
                    color="firebrick",
                    markersize=18,
                    markeredgewidth=3,
                    zorder=7,
                    visible=False,
                )

            artists.append(
                {
                    "xs": xs,
                    "ys": ys,
                    "df_len": len(df),
                    "crash_idx": crash_idx,
                    "trail": trail_line,
                    "dot": dot,
                    "crash_marker": crash_marker,
                    "done": False,
                }
            )
    return artists


# ── Animation core ─────────────────────────────────────────────────────────────

def make_update_fn(all_artists, n_frames, max_rows):
    def update(frame):
        # Map [0, n_frames-1] → [0, max_rows-1] so the last frame always
        # reaches the final data row (frame/n_frames would fall short by one step).
        t = frame / (n_frames - 1) if n_frames > 1 else 1.0
        for a in all_artists:
            row = min(round(t * (max_rows - 1)), a["df_len"] - 1)
            t0 = max(0, row - TRAIL_FRAMES)
            a["trail"].set_data(a["xs"][t0 : row + 1], a["ys"][t0 : row + 1])

            if not a["done"]:
                a["dot"].set_data([a["xs"][row]], [a["ys"][row]])

            if a["crash_marker"] is not None and not a["done"] and row >= a["crash_idx"]:
                a["crash_marker"].set_visible(True)
                a["dot"].set_markersize(0)
                a["done"] = True

    return update


TRAIL_FRAMES = 45


# ── Single-speed video ─────────────────────────────────────────────────────────

def make_video(speed: float, output: str, fps: int, duration: float) -> None:
    data = load_speed_data(speed)
    scene_image = Image.open("vision/linearized_image.png")

    n_frames = int(duration * fps)
    max_rows = max(len(f["df"]) for g in data["groups"] for f in g["flights"])

    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.patch.set_facecolor("white")
    setup_panel(ax, speed, data["grid"], scene_image)
    ax.set_ylabel("y / m", fontsize=15)

    legend_handles = [
        Line2D([0], [0], color=g["color"], lw=2, marker="o", markersize=6, label=g["name"])
        for g in data["groups"]
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=13)

    # sm = plt.cm.ScalarMappable(cmap="coolwarm_r", norm=Normalize(0, 1))
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    # cbar.set_label("Compliance")

    fig.suptitle(f"Constitutional Controller  |  v = {speed} m/s", fontsize=13)
    ax.set_title("")

    all_artists = build_artists(ax, data["groups"])
    update = make_update_fn(all_artists, n_frames, max_rows)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
    _save(anim, output, fps)
    plt.close()


# ── Combined multi-speed video ─────────────────────────────────────────────────

def make_combined_video(speeds: list[float], output: str, fps: int, duration: float) -> None:
    all_data = [load_speed_data(s) for s in speeds]
    scene_image = Image.open("vision/linearized_image.png")

    n_frames = int(duration * fps)
    max_rows = max(len(f["df"]) for d in all_data for g in d["groups"] for f in g["flights"])

    n_cols = len(speeds)
    fig, axes = plt.subplots(1, n_cols, figsize=(6.5 * n_cols + 1.2, 6.5), sharey=True)
    fig.patch.set_facecolor("white")

    all_artists = []
    for ax, data in zip(axes, all_data):
        setup_panel(ax, data["speed"], data["grid"], scene_image)
        all_artists.extend(build_artists(ax, data["groups"]))

    axes[0].set_ylabel("y / m", fontsize=15)

    # Shared legend below the panels
    legend_handles = [
        Line2D([0], [0], color=g["color"], lw=2, marker="o", markersize=6, label=g["name"])
        for g in GROUPS
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(GROUPS),
        bbox_to_anchor=(0.5, 0.9),
        fontsize=13,
    )

    fig.tight_layout()

    update = make_update_fn(all_artists, n_frames, max_rows)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
    _save(anim, output, fps)
    plt.close()


# ── Website card video (16:9, CoCo-only, all speeds overlaid) ─────────────────

def make_website_video(speeds: list[float], output: str, fps: int, duration: float) -> None:
    boxes = load_bounding_boxes()
    n_frames = int(duration * fps)

    # CoCo-only groups, one per speed with a distinct colour
    speed_groups = []
    for speed in speeds:
        pattern = f"data/rosbags/coco-fixed-{speed}/*.csv"
        flights = load_flights(pattern)
        if not flights:
            raise FileNotFoundError(f"No CSVs matched: {pattern}")
        speed_groups.append({"name": f"v = {speed} m/s", "color": SPEED_COLORS[speed], "flights": flights})

    max_rows = max(len(f["df"]) for g in speed_groups for f in g["flights"])

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_facecolor(WEBSITE_BG)

    setup_background(ax, boxes)
    ax.axis("off")

    all_artists = build_artists(ax, speed_groups)
    update = make_update_fn(all_artists, n_frames, max_rows)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
    _save(anim, output, fps)
    plt.close()


# ── Save helper ────────────────────────────────────────────────────────────────

def _save(anim: animation.FuncAnimation, output: str, fps: int) -> None:
    writer = animation.FFMpegWriter(
        fps=fps,
        bitrate=4000,
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
    )
    print(f"Rendering → {output}")
    anim.save(output, writer=writer, dpi=150)
    print(f"Saved: {output}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Render CoCo flight data as video.")
    parser.add_argument("--speed", type=float, default=0.5, choices=[0.2, 0.5, 1.0])
    parser.add_argument("--all-speeds", action="store_true", help="Combine all speeds side-by-side")
    parser.add_argument("--overview", action="store_true", help="16:9 card video: CoCo-only, all speeds overlaid")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=20.0, help="Video length in seconds")
    args = parser.parse_args()

    if args.overview:
        output = args.output or "coco_overview.mp4"
        make_website_video([0.2, 0.5, 1.0], output, args.fps, args.duration)
    elif args.all_speeds:
        output = args.output or "comparison.mp4"
        make_combined_video([0.2, 0.5, 1.0], output, args.fps, args.duration)
    else:
        output = args.output or f"comparison_{args.speed:.1f}ms.mp4"
        make_video(args.speed, output, args.fps, args.duration)


if __name__ == "__main__":
    main()
