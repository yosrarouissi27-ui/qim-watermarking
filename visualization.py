"""
visualization.py — Affichage des images et des résultats

Fonctions
---------
    show_watermarking_results(original, watermarked, diff, attacked_images)
    plot_ber_comparison(results_dict, original_bits)
    show_watermark_bits(original_bits, extracted_bits, attack_name)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from metrics import ber, psnr


# ─── Palette ──────────────────────────────────────────────────────────────────
PURPLE = '#6c3fc5'
LIGHT_PURPLE = '#c9b5f0'
DARK = '#1a1a2e'
GRAY = '#555555'


def show_watermarking_results(
    original: np.ndarray,
    watermarked: np.ndarray,
    attacked_images: dict[str, np.ndarray] | None = None,
    save_path: str | None = None
) -> None:
    """
    Affiche l'image originale, tatouée, la différence amplifiée,
    et (optionnellement) les images attaquées.
    """
    diff = np.abs(watermarked.astype(np.int16) - original.astype(np.int16)).astype(np.uint8)
    diff_amplified = np.clip(diff * 10, 0, 255).astype(np.uint8)

    n_attacked = len(attacked_images) if attacked_images else 0
    n_cols = 3 + n_attacked
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5))
    fig.patch.set_facecolor(DARK)

    def show(ax, img, title, cmap='gray'):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, color='white', fontsize=10, pad=8)
        ax.axis('off')

    p = psnr(original, watermarked)
    show(axes[0], original,        "Image originale")
    show(axes[1], watermarked,     f"Image tatouée\nPSNR = {p:.1f} dB")
    show(axes[2], diff_amplified,  "Différence (×10)")

    if attacked_images:
        for idx, (name, img) in enumerate(attacked_images.items()):
            show(axes[3 + idx], img, f"Attaque :\n{name}")

    plt.suptitle("Système de Tatouage QIM — Visualisation", color='white',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
        print(f"  [✓] Image sauvegardée : {save_path}")
    plt.show()


def plot_ber_comparison(
    results: dict[str, list],
    original_bits: list | np.ndarray,
    save_path: str | None = None
) -> None:
    """
    Trace un diagramme en barres comparant le BER de chaque attaque.
    """
    attack_names = list(results.keys())
    ber_values = [ber(original_bits, bits) for bits in results.values()]

    fig, ax = plt.subplots(figsize=(max(6, len(attack_names) * 1.5), 4.5))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor('#0d0d1a')

    colors = [PURPLE if b < 0.15 else ('#f0a500' if b < 0.40 else '#e03030')
              for b in ber_values]
    bars = ax.bar(attack_names, ber_values, color=colors, edgecolor='white',
                  linewidth=0.5, width=0.6)

    # Ligne de référence
    ax.axhline(y=0.15, color=LIGHT_PURPLE, linestyle='--', linewidth=1,
               label='Seuil robustesse (BER=0.15)')
    ax.axhline(y=0.40, color='#f0a500', linestyle=':', linewidth=1,
               label='Seuil critique (BER=0.40)')

    for bar, val in zip(bars, ber_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=9)

    ax.set_ylim(0, max(max(ber_values) * 1.25, 0.55))
    ax.set_ylabel('BER (Bit Error Rate)', color='white')
    ax.set_title('Comparaison BER par type d\'attaque', color='white',
                 fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor(GRAY)
    ax.legend(facecolor='#0d0d1a', edgecolor=GRAY, labelcolor='white', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
        print(f"  [✓] Graphique sauvegardé : {save_path}")
    plt.show()


def show_watermark_bits(
    original_bits: list | np.ndarray,
    extracted_bits: list | np.ndarray,
    attack_name: str = "Sans attaque",
    save_path: str | None = None
) -> None:
    """
    Visualise les bits originaux vs extraits sous forme de barres binaires.
    """
    orig = np.array(original_bits)
    extr = np.array(extracted_bits)
    n = len(orig)

    errors = orig != extr
    b = ber(orig, extr)

    fig, axes = plt.subplots(2, 1, figsize=(min(n * 0.25 + 2, 16), 3.5))
    fig.patch.set_facecolor(DARK)

    for ax, bits, label in zip(axes, [orig, extr], ['Bits originaux', f'Bits extraits ({attack_name})']):
        ax.set_facecolor('#0d0d1a')
        for i, bit in enumerate(bits):
            color = '#e03030' if (label != 'Bits originaux' and errors[i]) else \
                    (PURPLE if bit == 1 else LIGHT_PURPLE)
            ax.bar(i, 1, color=color, width=0.85, edgecolor='none')
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel(label, color='white', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(GRAY)

    fig.suptitle(f'Watermark — BER = {b:.4f} | {int(np.sum(errors))}/{n} erreurs',
                 color='white', fontsize=11, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
        print(f"  [✓] Bits sauvegardés : {save_path}")
    plt.show()
