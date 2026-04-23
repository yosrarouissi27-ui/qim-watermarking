"""
watermark.py — Insertion et extraction du watermark via QIM (Quantization Index Modulation)

Principe QIM (formulation standard)
-------------------------------------
  Paramètre : Δ (delta) — demi-pas de quantification
  Période complète : 2Δ

  Encodage :
    bit=0 → x_enc = 2Δ · round(x / 2Δ)           (multiple pair de Δ)
    bit=1 → x_enc = 2Δ · round(x / 2Δ) + Δ       (multiple impair de Δ)
    Formule unifiée : x_enc = 2Δ · round(x / 2Δ) + bit · Δ

  Décodage :
    b = round(x_enc / Δ) mod 2

  Cette formulation garantit que _qim_decode(_qim_encode(x, b, Δ), Δ) == b
  pour tout x réel, sans ambiguïté aux demi-entiers (pas de banker's rounding).
"""

import numpy as np
from dct_utils import apply_dct_blocks, apply_idct_blocks, select_coeff_indices, padded_shape
from config import DELTA, SECRET_KEY, WATERMARK_BITS


# ─── Génération du watermark ──────────────────────────────────────────────────

def generate_watermark(n_bits: int = WATERMARK_BITS) -> np.ndarray:
    """
    Génère un watermark binaire pseudo-aléatoire reproductible.

    Parameters
    ----------
    n_bits : int

    Returns
    -------
    bits : np.ndarray  shape (n_bits,), dtype int64 — valeurs 0 ou 1
    """
    rng = np.random.default_rng(SECRET_KEY + 1)
    return rng.integers(0, 2, size=n_bits)


# ─── Quantification QIM ───────────────────────────────────────────────────────

def _qim_encode(x: float, bit: int, delta: float) -> float:
    """
    Encode le bit dans le coefficient DCT x par QIM standard.

    bit=0 → grille paire   {0, ±2Δ, ±4Δ, …}
    bit=1 → grille impaire {±Δ, ±3Δ, ±5Δ, …}
    """
    return 2.0 * delta * np.round(x / (2.0 * delta)) + bit * delta


def _qim_decode(x: float, delta: float) -> int:
    """
    Décode un coefficient x en bit {0, 1}.
    Utilise round(x/Δ) mod 2 — robuste car x_enc est un multiple entier de Δ.
    """
    return int(np.round(x / delta)) % 2


# ─── Insertion ────────────────────────────────────────────────────────────────

def embed(image: np.ndarray, watermark_bits: np.ndarray, delta: float = DELTA) -> np.ndarray:
    """
    Insère le watermark dans l'image via QIM dans le domaine DCT.

    Parameters
    ----------
    image          : np.ndarray  shape (H, W), dtype uint8
    watermark_bits : np.ndarray  shape (n_bits,), valeurs 0/1
    delta          : float       — demi-pas de quantification QIM

    Returns
    -------
    watermarked : np.ndarray  shape (H, W), dtype uint8
    """
    n_bits = len(watermark_bits)
    dct_image, orig_shape = apply_dct_blocks(image)
    ps = padded_shape(orig_shape)
    positions = select_coeff_indices(n_bits, ps)

    for bit, (bi, bj, r, c) in zip(watermark_bits, positions):
        x = float(dct_image[bi + r, bj + c])
        dct_image[bi + r, bj + c] = _qim_encode(x, int(bit), delta)

    return apply_idct_blocks(dct_image, orig_shape)


# ─── Extraction ───────────────────────────────────────────────────────────────

def extract(image: np.ndarray, n_bits: int, delta: float = DELTA) -> list:
    """
    Extrait les bits du watermark depuis une image (potentiellement attaquée).

    Parameters
    ----------
    image  : np.ndarray  shape (H, W), dtype uint8
    n_bits : int
    delta  : float       — même demi-pas qu'à l'insertion

    Returns
    -------
    bits : list[int]  — bits extraits (0 ou 1)
    """
    dct_image, orig_shape = apply_dct_blocks(image)
    ps = padded_shape(orig_shape)
    positions = select_coeff_indices(n_bits, ps)

    extracted = []
    for (bi, bj, r, c) in positions:
        x = float(dct_image[bi + r, bj + c])
        extracted.append(_qim_decode(x, delta))

    return extracted
