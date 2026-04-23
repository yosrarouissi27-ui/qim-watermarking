# 🎓 Mini-Projet : Sécurisation des images 2D par Tatouage Numérique QIM

## Description

Système complet de **tatouage numérique** (watermarking) d'images 2D basé sur la méthode **QIM (Quantization Index Modulation)** appliquée dans le domaine **DCT** (Discrete Cosine Transform).

---

## Structure du projet

```
qim_watermarking/
│
├── config.py              # Paramètres globaux (Δ, bits, clé secrète…)
├── dct_utils.py           # DCT/IDCT 2D par blocs 8×8
├── watermark.py           # Insertion & extraction QIM
├── attacks.py             # Simulation d'attaques
├── metrics.py             # PSNR, SSIM, BER, NC
├── visualization.py       # Affichage matplotlib
├── create_test_image.py   # Génération d'image de test
├── main.py                # Programme principal (CLI)
└── README.md
```

---

## Installation

```bash
pip install opencv-python numpy matplotlib scikit-image scipy pillow
```

---

## Utilisation

### Lancement rapide
```bash
python main.py
```
> Si `lena.png` est absent, une image synthétique est créée automatiquement.

### Avec une image personnalisée
```bash
python main.py --image ma_photo.png
```

### Paramètres avancés
```bash
python main.py --delta 20 --bits 128 --output-dir resultats/
```

### Sans affichage graphique (mode batch/serveur)
```bash
python main.py --no-display
```

---

## Principe QIM

Pour chaque bit **b ∈ {0, 1}** et un coefficient DCT **x** :

- **b = 0** → quantifier vers le multiple **pair** de Δ le plus proche  
  `x_encodé = Δ × round(x / Δ)`
  
- **b = 1** → quantifier vers le multiple **impair** de Δ le plus proche  
  `x_encodé = Δ × (round((x - Δ/2) / Δ) + 0.5)`

**Décodage :**  
`b_extrait = round(x / Δ) mod 2`

---

## Métriques

| Métrique | Description | Seuil acceptable |
|----------|-------------|------------------|
| **PSNR** | Qualité visuelle (dB) | > 35 dB |
| **SSIM** | Similarité structurelle | > 0.95 |
| **BER**  | Taux d'erreur bit | < 0.15 |
| **NC**   | Corrélation normalisée | > 0.85 |

---

## Paramètres configurables (`config.py`)

| Paramètre | Défaut | Rôle |
|-----------|--------|------|
| `DELTA` | 30 | Pas de quantification QIM |
| `WATERMARK_BITS` | 64 | Nombre de bits du watermark |
| `SECRET_KEY` | 42 | Graine pseudo-aléatoire |
| `BLOCK_SIZE` | 8 | Taille des blocs DCT |
| `JPEG_QUALITY` | 70 | Qualité JPEG pour l'attaque |
| `GAUSSIAN_NOISE_SIGMA` | 5 | Écart-type du bruit gaussien |

---

## Attaques simulées

1. **Bruit gaussien** — bruit additif (σ = 5)
2. **Compression JPEG** — qualité 70
3. **Filtre médian** — noyau 3×3
4. **Bruit sel & poivre** — densité 2%

---

## Résultats typiques

```
══════════════════════════════════════════════════════════════
  RAPPORT D'ÉVALUATION — Tatouage QIM
══════════════════════════════════════════════════════════════
  Qualité image tatouée :
    PSNR :   38.50 dB  ✓ imperceptible
    SSIM :   0.9821     ✓ très bonne

  Robustesse après attaques :
  Attaque                     BER      NC      Statut
  ──────────────────────────────────────────────────────────────
  Sans attaque             0.0000  1.0000       ✓ OK
  Bruit gaussien           0.0625  0.8750   ✓ robuste
  Compression JPEG         0.0938  0.8125   ✓ robuste
  Filtre médian            0.1250  0.7500   ✓ robuste
  Sel & poivre             0.0469  0.9063   ✓ robuste
══════════════════════════════════════════════════════════════
```
