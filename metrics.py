"""
metrics.py — Métriques d'évaluation du système de tatouage

Métriques de qualité visuelle
------------------------------
    psnr(original, watermarked)  → PSNR en dB  (Peak Signal-to-Noise Ratio)
    ssim(original, watermarked)  → SSIM ∈ [0,1] (Structural Similarity Index)

Métriques de robustesse
------------------------
    ber(original_bits, extracted_bits) → BER ∈ [0,1] (Bit Error Rate)
    nc(original_bits, extracted_bits)  → NC  ∈ [0,1]  (Normalized Correlation)

Fonctions utilitaires
---------------------
    print_report(...)   → affiche un tableau récapitulatif
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ─── Qualité visuelle ─────────────────────────────────────────────────────────

def psnr(original: np.ndarray, watermarked: np.ndarray) -> float:
    """
    Calcule le PSNR entre l'image originale et l'image tatouée.
    Un PSNR > 35 dB indique une distorsion imperceptible.

    Returns
    -------
    psnr_db : float
    """
    return peak_signal_noise_ratio(original, watermarked, data_range=255)


def ssim(original: np.ndarray, watermarked: np.ndarray) -> float:
    """
    Calcule le SSIM entre deux images.
    SSIM = 1 → identiques ; SSIM > 0.95 → très bonne qualité.

    Returns
    -------
    ssim_val : float ∈ [0, 1]
    """
    return structural_similarity(original, watermarked, data_range=255)


# ─── Robustesse ───────────────────────────────────────────────────────────────

def ber(original_bits: np.ndarray | list, extracted_bits: np.ndarray | list) -> float:
    """
    Bit Error Rate : fraction de bits mal extraits.
    BER = 0   → extraction parfaite.
    BER = 0.5 → aléatoire (équivalent à deviner).

    Returns
    -------
    ber_val : float ∈ [0, 1]
    """
    orig = np.array(original_bits, dtype=int)
    extr = np.array(extracted_bits, dtype=int)
    if len(orig) != len(extr):
        raise ValueError("Les deux séquences doivent avoir la même longueur.")
    return float(np.sum(orig != extr) / len(orig))


def nc(original_bits: np.ndarray | list, extracted_bits: np.ndarray | list) -> float:
    """
    Normalized Correlation entre bits originaux et extraits.
    NC = 1 → extraction parfaite ; NC = 0 → aucune corrélation.

    Returns
    -------
    nc_val : float ∈ [0, 1]
    """
    orig = np.array(original_bits, dtype=float)
    extr = np.array(extracted_bits, dtype=float)
    # Convertir {0,1} → {-1, +1} pour la corrélation normalisée
    w1 = 2 * orig - 1
    w2 = 2 * extr - 1
    denom = np.sqrt(np.sum(w1 ** 2) * np.sum(w2 ** 2))
    if denom == 0:
        return 0.0
    return float(np.dot(w1, w2) / denom)


# ─── Rapport ──────────────────────────────────────────────────────────────────

def print_report(
    original: np.ndarray,
    watermarked: np.ndarray,
    original_bits: list | np.ndarray,
    results: dict[str, list]
) -> None:
    """
    Affiche un tableau récapitulatif des métriques pour chaque attaque.

    Parameters
    ----------
    original       : image originale
    watermarked    : image tatouée (avant attaque)
    original_bits  : bits du watermark d'origine
    results        : dict {nom_attaque: bits_extraits}
    """
    psnr_wm = psnr(original, watermarked)
    ssim_wm = ssim(original, watermarked)

    sep = "─" * 62
    print(f"\n{'═'*62}")
    print(f"  RAPPORT D'ÉVALUATION — Tatouage QIM")
    print(f"{'═'*62}")
    print(f"  Qualité image tatouée :")
    print(f"    PSNR : {psnr_wm:7.2f} dB  {'✓ imperceptible' if psnr_wm > 35 else '✗ visible'}")
    print(f"    SSIM : {ssim_wm:7.4f}     {'✓ très bonne' if ssim_wm > 0.95 else '✗ dégradée'}")
    print(f"\n  Robustesse après attaques :")
    print(f"  {'Attaque':<25} {'BER':>7}  {'NC':>7}  {'Statut':>12}")
    print(f"  {sep}")

    # Sans attaque
    extr_no_attack = results.get("Sans attaque", original_bits)
    b = ber(original_bits, extr_no_attack)
    n = nc(original_bits, extr_no_attack)
    print(f"  {'Sans attaque':<25} {b:>7.4f}  {n:>7.4f}  {'✓ OK' if b < 0.01 else '✗ KO':>12}")

    for attack_name, extracted in results.items():
        if attack_name == "Sans attaque":
            continue
        b = ber(original_bits, extracted)
        n = nc(original_bits, extracted)
        status = "✓ robuste" if b < 0.15 else ("⚠ dégradé" if b < 0.40 else "✗ cassé")
        print(f"  {attack_name:<25} {b:>7.4f}  {n:>7.4f}  {status:>12}")

    print(f"{'═'*62}\n")
