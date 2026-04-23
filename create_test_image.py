"""
create_test_image.py — Crée une image de test si lena.png n'est pas disponible.

Génère une image synthétique en niveaux de gris avec des motifs variés
(dégradés, cercles, textures) pour tester le système de tatouage.

Usage : python create_test_image.py
"""

import numpy as np
import cv2
import os


def create_synthetic_image(size: int = 512, output_path: str = "lena.png") -> None:
    """
    Crée une image synthétique riche en textures, adaptée aux tests de watermarking.
    """
    img = np.zeros((size, size), dtype=np.float64)
    cx, cy = size // 2, size // 2

    # Fond : dégradé radial
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)
            img[i, j] = 200 * np.exp(-dist / (size * 0.6))

    # Ajout de motifs sinusoïdaux (texture)
    x = np.linspace(0, 4 * np.pi, size)
    img += 30 * np.sin(x[np.newaxis, :]) * np.cos(x[:, np.newaxis])

    # Cercles concentriques
    for r in range(30, size // 2, 40):
        cv2.circle(img.astype(np.uint8), (cx, cy), r, 150, 2)
        rr, cc = np.ogrid[:size, :size]
        mask = np.abs(np.sqrt((rr - cy) ** 2 + (cc - cx) ** 2) - r) < 1.5
        img[mask] = 120 + r // 5

    # Rectangles pour simuler des objets
    for k in range(3):
        x0 = 50 + k * 140
        img[100:180, x0:x0 + 80] += 40

    # Bruit léger pour enrichir la texture HF
    img += np.random.normal(0, 5, img.shape)

    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, img)
    print(f"  [✓] Image de test créée : {output_path} ({size}×{size} pixels)")


if __name__ == "__main__":
    create_synthetic_image()
