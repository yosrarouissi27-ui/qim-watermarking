"""
attacks.py — Simulation d'attaques sur une image tatouée

Attaques disponibles
--------------------
    gaussian_noise(image, sigma)     → bruit gaussien additif
    jpeg_compression(image, quality) → compression/décompression JPEG
    median_filter(image, ksize)      → filtre médian (lissage)
    salt_pepper(image, density)      → bruit impulsionnel sel & poivre
    rotation(image, angle)           → rotation (avec recadrage)

Chaque fonction retourne une image uint8 de même shape que l'entrée.
"""

import io
import cv2
import numpy as np
from PIL import Image
from config import GAUSSIAN_NOISE_SIGMA, JPEG_QUALITY


# ─── Bruit gaussien ───────────────────────────────────────────────────────────

def gaussian_noise(image: np.ndarray, sigma: float = GAUSSIAN_NOISE_SIGMA) -> np.ndarray:
    """
    Ajoute un bruit gaussien additif à l'image.

    Parameters
    ----------
    image : np.ndarray  shape (H, W), dtype uint8
    sigma : float       — écart-type du bruit (en niveaux de gris, 0–255)

    Returns
    -------
    noisy : np.ndarray  shape (H, W), dtype uint8
    """
    noise = np.random.normal(0, sigma, image.shape).astype(np.float64)
    noisy = image.astype(np.float64) + noise
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)


# ─── Compression JPEG ─────────────────────────────────────────────────────────

def jpeg_compression(image: np.ndarray, quality: int = JPEG_QUALITY) -> np.ndarray:
    """
    Simule une compression JPEG en encodant/décodant l'image en mémoire.

    Parameters
    ----------
    image   : np.ndarray  shape (H, W), dtype uint8
    quality : int         — qualité JPEG (1 = très dégradée, 95 = très bonne)

    Returns
    -------
    compressed : np.ndarray  shape (H, W), dtype uint8
    """
    pil_img = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = np.array(Image.open(buffer).convert('L'))
    return compressed


# ─── Filtre médian ────────────────────────────────────────────────────────────

def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Applique un filtre médian (smoothing, peut effacer le watermark HF).

    Parameters
    ----------
    image : np.ndarray  shape (H, W), dtype uint8
    ksize : int         — taille du noyau (doit être impair : 3, 5, 7…)

    Returns
    -------
    filtered : np.ndarray  shape (H, W), dtype uint8
    """
    if ksize % 2 == 0:
        raise ValueError("ksize doit être impair.")
    return cv2.medianBlur(image, ksize)


# ─── Bruit sel & poivre ───────────────────────────────────────────────────────

def salt_pepper(image: np.ndarray, density: float = 0.02) -> np.ndarray:
    """
    Ajoute un bruit impulsionnel sel et poivre.

    Parameters
    ----------
    image   : np.ndarray  shape (H, W), dtype uint8
    density : float       — proportion totale de pixels bruités (0–1)

    Returns
    -------
    noisy : np.ndarray  shape (H, W), dtype uint8
    """
    noisy = image.copy()
    n_pixels = image.size
    n_salt = int(n_pixels * density / 2)
    n_pepper = int(n_pixels * density / 2)

    rng = np.random.default_rng()

    # Sel (pixels blancs)
    coords = [rng.integers(0, s, n_salt) for s in image.shape]
    noisy[coords[0], coords[1]] = 255

    # Poivre (pixels noirs)
    coords = [rng.integers(0, s, n_pepper) for s in image.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy


# ─── Rotation ─────────────────────────────────────────────────────────────────

def rotation(image: np.ndarray, angle: float = 1.0) -> np.ndarray:
    """
    Applique une légère rotation (simulation d'attaque géométrique).

    Parameters
    ----------
    image : np.ndarray  shape (H, W), dtype uint8
    angle : float       — angle en degrés

    Returns
    -------
    rotated : np.ndarray  shape (H, W), dtype uint8
    """
    h, w = image.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return rotated
