"""
main.py — Programme principal : Tatouage Numérique QIM sur images 2D

Usage
-----
    python main.py                       # utilise lena.png par défaut
    python main.py --image mon_image.png # image personnalisée
    python main.py --delta 20 --bits 128 # paramètres personnalisés
    python main.py --no-display          # sans affichage graphique

Flux d'exécution
----------------
    1. Charger l'image hôte en niveaux de gris
    2. Générer le watermark binaire (pseudo-aléatoire)
    3. Insérer le watermark via QIM dans le domaine DCT
    4. Sauvegarder l'image tatouée
    5. Appliquer les attaques
    6. Extraire le watermark de chaque image attaquée
    7. Calculer PSNR, SSIM, BER, NC
    8. Afficher le rapport et les visualisations
"""

import os
import sys
import argparse
import numpy as np
import cv2

# ─── Import des modules du projet ─────────────────────────────────────────────
import config
from watermark import embed, extract, generate_watermark
from attacks import gaussian_noise, jpeg_compression, median_filter, salt_pepper
from metrics import psnr, ssim, ber, nc, print_report
from visualization import (
    show_watermarking_results,
    plot_ber_comparison,
    show_watermark_bits,
)
from create_test_image import create_synthetic_image


# ─── Parsing des arguments ────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Système de tatouage QIM sur images 2D"
    )
    parser.add_argument(
        "--image", type=str, default=config.DEFAULT_IMAGE_PATH,
        help="Chemin vers l'image hôte (PNG, JPG, BMP)"
    )
    parser.add_argument(
        "--delta", type=float, default=config.DELTA,
        help="Pas de quantification QIM (défaut : %(default)s)"
    )
    parser.add_argument(
        "--bits", type=int, default=config.WATERMARK_BITS,
        help="Nombre de bits du watermark (défaut : %(default)s)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Désactive l'affichage graphique (mode batch)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Répertoire de sortie pour les images et graphiques"
    )
    return parser.parse_args()


# ─── Chargement de l'image ────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Charge une image en niveaux de gris. Crée une image de test si absente."""
    if not os.path.exists(path):
        print(f"  [!] Image '{path}' introuvable. Création d'une image de test…")
        create_synthetic_image(size=512, output_path=path)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {path}")

    print(f"  [✓] Image chargée : {path}  ({img.shape[1]}×{img.shape[0]} px)")
    return img


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    # Mise à jour des paramètres globaux si fournis en CLI
    config.DELTA = args.delta
    config.WATERMARK_BITS = args.bits

    print("\n" + "═" * 62)
    print("  🎓 TATOUAGE NUMÉRIQUE QIM — Démarrage")
    print("═" * 62)
    print(f"  Δ (delta)      : {config.DELTA}")
    print(f"  Bits watermark : {config.WATERMARK_BITS}")
    print(f"  Clé secrète    : {config.SECRET_KEY}")
    print("═" * 62 + "\n")

    # ── Étape 1 : Chargement ──────────────────────────────────────────────────
    print("[1/6] Chargement de l'image…")
    original = load_image(args.image)

    # ── Étape 2 : Génération du watermark ────────────────────────────────────
    print("[2/6] Génération du watermark binaire…")
    wm_bits = generate_watermark(config.WATERMARK_BITS)
    print(f"  [✓] Watermark : {wm_bits[:16]}… ({config.WATERMARK_BITS} bits)")

    # ── Étape 3 : Insertion QIM ───────────────────────────────────────────────
    print("[3/6] Insertion du watermark (QIM + DCT)…")
    watermarked = embed(original, wm_bits)
    wm_path = os.path.join(args.output_dir, "watermarked.png")
    cv2.imwrite(wm_path, watermarked)
    print(f"  [✓] Image tatouée sauvegardée : {wm_path}")
    print(f"  [✓] PSNR = {psnr(original, watermarked):.2f} dB  "
          f"| SSIM = {ssim(original, watermarked):.4f}")

    # ── Étape 4 : Attaques ────────────────────────────────────────────────────
    print("[4/6] Application des attaques…")
    attacked_images: dict[str, np.ndarray] = {
        "Sans attaque":      watermarked.copy(),
        "Bruit gaussien":    gaussian_noise(watermarked, sigma=config.GAUSSIAN_NOISE_SIGMA),
        "Compression JPEG":  jpeg_compression(watermarked, quality=config.JPEG_QUALITY),
        "Filtre médian":     median_filter(watermarked, ksize=3),
        "Sel & poivre":      salt_pepper(watermarked, density=0.02),
    }
    for name, img in attacked_images.items():
        safe_name = name.replace(" ", "_").replace("&", "et").lower()
        path = os.path.join(args.output_dir, f"attacked_{safe_name}.png")
        cv2.imwrite(path, img)
        print(f"  [✓] {name:<22} → {path}")

    # ── Étape 5 : Extraction ─────────────────────────────────────────────────
    print("[5/6] Extraction du watermark…")
    extraction_results: dict[str, list] = {}
    for name, img in attacked_images.items():
        extracted = extract(img, config.WATERMARK_BITS)
        extraction_results[name] = extracted
        b = ber(wm_bits, extracted)
        print(f"  [✓] {name:<22} BER = {b:.4f}")

    # ── Étape 6 : Rapport & Visualisation ────────────────────────────────────
    print("[6/6] Rapport et visualisation…\n")
    print_report(original, watermarked, wm_bits, extraction_results)

    if not args.no_display:
        # Images principales
        show_watermarking_results(
            original, watermarked,
            attacked_images={
                k: v for k, v in attacked_images.items() if k != "Sans attaque"
            },
            save_path=os.path.join(args.output_dir, "results_images.png")
        )

        # Graphique BER
        plot_ber_comparison(
            extraction_results, wm_bits,
            save_path=os.path.join(args.output_dir, "ber_comparison.png")
        )

        # Bits après attaque JPEG (la plus commune)
        show_watermark_bits(
            wm_bits,
            extraction_results["Compression JPEG"],
            attack_name="Compression JPEG",
            save_path=os.path.join(args.output_dir, "bits_jpeg.png")
        )
    else:
        print("  [i] Mode --no-display : visualisations désactivées.")

    print("\n✅ Pipeline terminé. Résultats dans :", args.output_dir)
    print("═" * 62 + "\n")


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n[!] Interrompu par l'utilisateur.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERREUR] {e}")
        raise
