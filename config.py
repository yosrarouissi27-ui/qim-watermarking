"""
config.py — Paramètres globaux du système de tatouage QIM
"""

# ─── Clé secrète (graine pseudo-aléatoire) ───────────────────────────────────
SECRET_KEY = 42

# ─── Paramètre de quantification QIM ─────────────────────────────────────────
# Plus grand → robustesse ↑, invisibilité ↓
DELTA = 20

# ─── Taille du watermark (bits) ──────────────────────────────────────────────
WATERMARK_BITS = 64

# ─── Chemin de l'image hôte par défaut ───────────────────────────────────────
# L'image doit être en niveaux de gris ou sera convertie automatiquement
DEFAULT_IMAGE_PATH = "lena.png"

# ─── Paramètres de la DCT par blocs ──────────────────────────────────────────
BLOCK_SIZE = 8          # taille d'un bloc DCT (8×8 standard)

# ─── Fréquences moyennes à modifier (indices zig-zag dans un bloc 8×8) ───────
# Les coefficients médians sont plus robustes que hautes/basses fréquences
MID_FREQ_INDICES = [(3, 1), (2, 2), (1, 3), (4, 0), (0, 4), (3, 2), (2, 3)]

# ─── Paramètres d'attaque ────────────────────────────────────────────────────
GAUSSIAN_NOISE_SIGMA = 5    # écart-type du bruit gaussien (0–255)
JPEG_QUALITY = 70           # qualité JPEG (1–95)

# ─── Seuil de détection QIM ──────────────────────────────────────────────────
# Inutilisé directement (décodage par modulo), gardé pour référence
DETECTION_THRESHOLD = DELTA / 2
