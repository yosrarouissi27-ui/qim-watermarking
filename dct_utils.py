"""
dct_utils.py — Transformation DCT 2D par blocs 8×8

Fonctions :
    apply_dct_blocks(image)              → (dct_coeffs, orig_shape)
    apply_idct_blocks(dct, orig_shape)   → image uint8
    select_coeff_indices(n_bits, shape)  → list of (bi, bj, r, c)
    padded_shape(image_shape)            → (ph, pw)
"""

import numpy as np
from scipy.fft import dctn, idctn
from config import BLOCK_SIZE, MID_FREQ_INDICES, SECRET_KEY


def padded_shape(image_shape: tuple) -> tuple:
    """Retourne les dimensions après padding multiple de BLOCK_SIZE."""
    h, w = image_shape
    ph = h + (BLOCK_SIZE - h % BLOCK_SIZE) % BLOCK_SIZE
    pw = w + (BLOCK_SIZE - w % BLOCK_SIZE) % BLOCK_SIZE
    return ph, pw


def _pad_image(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    ph, pw = padded_shape((h, w))
    pad_h = ph - h
    pad_w = pw - w
    return np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')


def apply_dct_blocks(image: np.ndarray) -> tuple:
    """
    Applique la DCT 2D bloc par bloc.

    Parameters
    ----------
    image : np.ndarray  shape (H, W), any numeric dtype

    Returns
    -------
    dct_image  : np.ndarray  float64 — coefficients DCT (padded size)
    orig_shape : tuple       — (H_orig, W_orig)
    """
    orig_shape = image.shape[:2]
    padded = _pad_image(image.astype(np.float64))
    ph, pw = padded.shape
    dct_image = np.zeros((ph, pw), dtype=np.float64)

    for i in range(0, ph, BLOCK_SIZE):
        for j in range(0, pw, BLOCK_SIZE):
            block = padded[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE]
            dct_image[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE] = dctn(block, norm='ortho')

    return dct_image, orig_shape


def apply_idct_blocks(dct_image: np.ndarray, orig_shape: tuple) -> np.ndarray:
    """
    Applique l'IDCT 2D bloc par bloc et recadre.

    Returns
    -------
    image : np.ndarray  shape orig_shape, dtype uint8
    """
    ph, pw = dct_image.shape
    image = np.zeros((ph, pw), dtype=np.float64)

    for i in range(0, ph, BLOCK_SIZE):
        for j in range(0, pw, BLOCK_SIZE):
            block = dct_image[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE]
            image[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE] = idctn(block, norm='ortho')

    H, W = orig_shape
    image = image[:H, :W]
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def apply_idct_blocks_float(dct_image: np.ndarray, orig_shape: tuple) -> np.ndarray:
    """
    Même que apply_idct_blocks mais retourne float64 (sans clampage uint8).
    Utilisé en interne pour tester la fidélité du round-trip.
    """
    ph, pw = dct_image.shape
    image = np.zeros((ph, pw), dtype=np.float64)

    for i in range(0, ph, BLOCK_SIZE):
        for j in range(0, pw, BLOCK_SIZE):
            block = dct_image[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE]
            image[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE] = idctn(block, norm='ortho')

    H, W = orig_shape
    return image[:H, :W]


def select_coeff_indices(n_bits: int, image_shape: tuple) -> list:
    """
    Sélectionne pseudo-aléatoirement n_bits positions (bi, bj, r, c)
    dans les coefficients de moyenne fréquence des blocs DCT.

    Parameters
    ----------
    n_bits      : int    — nombre de bits
    image_shape : tuple  — (H, W) — doit être les dimensions PADDÉES
                           (utiliser padded_shape() avant d'appeler cette fonction)
    """
    rng = np.random.default_rng(SECRET_KEY)
    ph, pw = image_shape

    pool = [
        (bi, bj, r, c)
        for bi in range(0, ph, BLOCK_SIZE)
        for bj in range(0, pw, BLOCK_SIZE)
        for (r, c) in MID_FREQ_INDICES
    ]

    if n_bits > len(pool):
        raise ValueError(
            f"Image trop petite pour insérer {n_bits} bits "
            f"(capacité max = {len(pool)}). "
            f"Réduisez WATERMARK_BITS ou utilisez une image plus grande."
        )

    indices = rng.choice(len(pool), size=n_bits, replace=False)
    return [pool[i] for i in indices]
