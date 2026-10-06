#!/usr/bin/env python3
"""Generate presentation-ready ablation study plots from data/processed/ablation."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    data_path = Path("data/processed/ablation/ablation.json")
    if not data_path.exists():
        raise FileNotFoundError(f"Ablation data not found at {data_path}")

    with open(data_path, "r") as f:
        data = json.load(f)

    # Extract metrics (taking first seed index 0)
    arms = ["TEXT ONLY", "IMAGE ONLY", "FUSED"]
    arm_keys = ["text_only", "image_only", "fused"]
    
    macro_f1s = [data[k][0]["macro_f1"] for k in arm_keys]
    precisions = [data[k][0]["precision"] for k in arm_keys]
    recalls = [data[k][0]["recall"] for k in arm_keys]
    pr_aucs = [data[k][0]["pr_auc"] for k in arm_keys]
    roc_aucs = [data[k][0]["roc_auc"] for k in arm_keys]

    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Slide-Themed Minimalist Blueprint Chart (matches user slide mockup)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=300)
    fig.patch.set_facecolor("#F7F7F4")
    ax.set_facecolor("#F7F7F4")

    # Subtle graph grid styling matching the blueprint grid
    ax.grid(True, which="both", color="#E1E1DC", linestyle="-", linewidth=0.75, zorder=0)
    ax.set_axisbelow(True)

    x_positions = np.arange(len(arms))
    bar_width = 0.45

    # High-contrast technical colors
    colors = ["#2B4C7E", "#C05646", "#2A7B62"]
    edge_colors = ["#1B3356", "#873225", "#1B5241"]

    bars = ax.bar(
        x_positions,
        macro_f1s,
        width=bar_width,
        color=colors,
        edgecolor=edge_colors,
        linewidth=1.4,
        zorder=3
    )

    # Add numeric labels on top of bars
    for bar, val in zip(bars, macro_f1s):
        y_pos = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos + 0.03,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontfamily="monospace",
            fontsize=13,
            fontweight="bold",
            color="#1E1E1E",
            zorder=4
        )

    # Formatting axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(arms, fontfamily="monospace", fontsize=11, fontweight="bold", color="#1E1E1E")
    ax.set_ylabel("Macro-F1 Score", fontfamily="monospace", fontsize=12, fontweight="bold", color="#1E1E1E", labelpad=10)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=10, colors="#222222")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#222222")
        ax.spines[spine].set_linewidth(1.5)

    plt.tight_layout()
    chart1_path = out_dir / "ablation_macro_f1_slide_theme.png"
    plt.savefig(chart1_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 1b. Transparent Version (drops directly into the slide box)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.grid(axis="y", color="#CBD5E1", linestyle="--", linewidth=0.7, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    bars = ax.bar(
        x_positions,
        macro_f1s,
        width=0.45,
        color=colors,
        edgecolor=edge_colors,
        linewidth=1.4,
        zorder=3
    )

    for bar, val in zip(bars, macro_f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.03,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontfamily="monospace",
            fontsize=13,
            fontweight="bold",
            color="#1E1E1E",
            zorder=4
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(arms, fontfamily="monospace", fontsize=11, fontweight="bold", color="#1E1E1E")
    ax.set_ylabel("Macro-F1 Score", fontfamily="monospace", fontsize=12, fontweight="bold", color="#1E1E1E", labelpad=10)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=10, colors="#222222")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#222222")
        ax.spines[spine].set_linewidth(1.5)

    plt.tight_layout()
    chart1b_path = out_dir / "ablation_macro_f1_transparent.png"
    plt.savefig(chart1b_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 2. Modern Presentation Ready (Transparent / White Background)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#EAEAEA", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    modern_colors = ["#3B82F6", "#EF4444", "#10B981"]
    bars = ax.bar(
        x_positions,
        macro_f1s,
        width=0.42,
        color=modern_colors,
        edgecolor="none",
        alpha=0.9,
        zorder=3
    )

    for bar, val in zip(bars, macro_f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="semibold",
            color="#1F2937",
            zorder=4
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(arms, fontsize=11, fontweight="bold", color="#1F2937")
    ax.set_ylabel("Macro-F1 Score", fontsize=12, fontweight="bold", color="#1F2937", labelpad=10)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#CBD5E1")
        ax.spines[spine].set_linewidth(1.2)

    ax.set_title("Modality Ablation Performance (Macro-F1)", fontsize=13, fontweight="bold", pad=15, color="#0F172A")
    plt.tight_layout()
    chart2_path = out_dir / "ablation_macro_f1_clean.png"
    plt.savefig(chart2_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 3. Comprehensive Multi-Metric Comparison (All slide metrics)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    fig.patch.set_facecolor("#F9F9F6")
    ax.set_facecolor("#F9F9F6")
    ax.grid(axis="y", color="#E2E2DC", linestyle="-", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    metrics_names = ["Macro-F1", "Precision", "Recall", "PR-AUC", "ROC-AUC"]
    x = np.arange(len(metrics_names))
    width = 0.24

    r1 = ax.bar(x - width, [macro_f1s[0], precisions[0], recalls[0], pr_aucs[0], roc_aucs[0]],
                width, label="Text Only", color="#2B4C7E", zorder=3)
    r2 = ax.bar(x, [macro_f1s[1], precisions[1], recalls[1], pr_aucs[1], roc_aucs[1]],
                width, label="Image Only", color="#C05646", zorder=3)
    r3 = ax.bar(x + width, [macro_f1s[2], precisions[2], recalls[2], pr_aucs[2], roc_aucs[2]],
                width, label="Fused (Cross-Attn)", color="#2A7B62", zorder=3)

    ax.set_ylabel("Score", fontfamily="monospace", fontsize=11, fontweight="bold", color="#1E1E1E")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontfamily="monospace", fontsize=11, fontweight="bold", color="#1E1E1E")
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=True, facecolor="#F9F9F6", edgecolor="#CCCCCC", fontsize=10, loc="upper right")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#222222")
        ax.spines[spine].set_linewidth(1.2)

    # Values for Macro-F1 specifically highlighted
    for r in [r1, r2, r3]:
        rect = r[0]  # Macro-F1 bar
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontfamily="monospace", fontsize=9, fontweight="bold")

    ax.set_title("Full Evaluation Metrics by Modality", fontfamily="monospace", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    chart3_path = out_dir / "ablation_multi_metric.png"
    plt.savefig(chart3_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    print("Generated:")
    print(f" - {chart1_path}")
    print(f" - {chart2_path}")
    print(f" - {chart3_path}")

if __name__ == "__main__":
    generate_plots()
