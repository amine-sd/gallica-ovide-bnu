# Notebooks — Gallica Images / Illustrations Ovide
**Stage Bnu Strasbourg — depuis avril 2026**
**Corpus** : Illustrations des *Métamorphoses* d'Ovide, 16e–17e siècles

---

## Architecture

```
notebooks/
├── gallica_utils.py                    ← fonctions partagées (importer dans chaque notebook)
│
├── 01_similarite_salomon/              ← recherche par similarité d'image (corpus Salomon)
│   ├── 01_documentation_api.ipynb      ← documentation des endpoints de l'API BnF
│   ├── 02_collecte_similarite.ipynb    ← collecte des résultats via l'API → CSV brut
│   ├── 03_analyse_resultats_bruts.ipynb← statistiques + tableau HTML sans IA
│   └── 04_similarite_mayence1545.ipynb ← même pipeline appliqué au corpus Wickram/Mayence
│
├── 02_classification_bois_cuivre/      ← classifieur ResNet50 bois / cuivre
│   ├── 01_dataset.ipynb                ← téléchargement, segmentation YOLO, split train/val/test
│   ├── 02_entrainement_v1_v2.ipynb     ← notebook générique (relancé pour v1 et v2)
│   ├── 03_entrainement_v3.ipynb        ← entraînement v3 (dataset enrichi, fine-tuning 2 étapes)
│   ├── 04_application_modeles.ipynb    ← applique v1/v2/v3 aux résultats de similarité → CSV enrichi
│   ├── 05_validation_experte.ipynb     ← génère les tableaux HTML soumis à Céline Bohnert
│   └── 06_evaluation_versions.ipynb    ← compare v1/v2/v3 face aux annotations de Céline
│
├── 03_classification_graveur/          ← classification par graveur (en cours)
│   └── 01_dataset.ipynb                ← constitution du dataset par graveur
│
├── 04_exploration/                     ← explorations ponctuelles
│   ├── comparaison_clip_dinov2.ipynb   ← CLIP (API BnF) vs DINOv2 (local)
│   ├── test_llm_iconographie_claude.ipynb
│   └── test_llm_iconographie_ollama.ipynb
│
└── 05_annexes/                         ← tâches annexes
    └── recuperation_bsb.ipynb          ← récupération de bibles illustrées BSB/MDZ
```

---

## Ordre d'exécution

```
Similarité   : 02_collecte_similarite → 03_analyse_resultats_bruts
Bois/cuivre  : 01_dataset → 02_entrainement_v1_v2 (ou 03_entrainement_v3)
               → 04_application_modeles → 05_validation_experte → 06_evaluation_versions
Graveur      : 01_dataset (puis entraînement à écrire)
```

---

## Pipeline

### gallica_utils.py
Toutes les fonctions réutilisables sont centralisées dans `gallica_utils.py`.
Importer depuis chaque notebook (profondeur uniforme `notebooks/<axe>/<fichier>.ipynb`) avec :
```python
import sys
sys.path.insert(0, "..")
from gallica_utils import charger_yolo, segmenter_corpus, charger_resnet, ...
```

### Ajouter une nouvelle source bois/cuivre (02_classification_bois_cuivre/01_dataset.ipynb)
Appeler `telecharger_pages_iiif()` avec la nouvelle URL manifest puis
`segmenter_corpus()` avec le dossier de destination — nommer les dossiers
selon la convention `{classe}_{graveur}_{ville}{année}/`.

### Entraîner une nouvelle version du modèle bois/cuivre
Changer `VERSION` dans `02_entrainement_v1_v2.ipynb` (entraînement simple), ou dupliquer
`03_entrainement_v3.ipynb` pour une nouvelle architecture/stratégie d'entraînement.

### Ajouter une version au tableau enrichi (04_application_modeles.ipynb)
Ajouter le chemin du nouveau `.pth` dans le dictionnaire `MODELES`.
Les colonnes et filtres HTML s'ajoutent automatiquement.

---

## Modèles de classification bois / cuivre

### Version 1 — `resnet50_v1.pth`

| Paramètre | Valeur |
|---|---|
| Architecture | ResNet50 fine-tuné (ImageNet → bois/cuivre) |
| Dataset | 744 images — 400 bois / 344 cuivre |
| Data augmentation | Aucune |
| Split train/val/test | 520 / 111 / 113 |
| Test accuracy | **100%** |

**Remarque :** le 100% sur un petit dataset homogène peut indiquer un surajustement.

---

### Version 2 — `resnet50_v2.pth`

| Paramètre | Valeur |
|---|---|
| Architecture | ResNet50 fine-tuné (ImageNet → bois/cuivre) |
| Dataset | 1506 images — 862 bois / 644 cuivre |
| Data augmentation | Flip horizontal |
| Split train/val/test | 1052 / 112 / 115 |
| Test accuracy | **98%** |

---

### Version 3 — `resnet50_bois_cuivre_v3.0.0.pth`

| Paramètre | Valeur |
|---|---|
| Architecture | ResNet50 — fine-tuning en 2 étapes (couche finale puis réseau complet) |
| Dataset | 29 éditions, ~2105 illustrations (456 bois / 1649 cuivre) |
| Data augmentation | Aucune — couleurs et illustrations natives conservées |
| Split train/val/test | 1473 / 315 / 317 |
| Test accuracy (set de test) | F1 ≈ 0.888, rappel ≈ 0.982 |

**Comparaison des 3 versions face aux annotations expertes de Céline**
(résultats de similarité appliqués au corpus Salomon — voir `06_evaluation_versions.ipynb`) :

| Version | Précision cuivre | Cuivres ratés (FN) | Faux cuivres (FP) |
|---|---|---|---|
| v1 | 91.3 % | 112 | 14 |
| v2 | 91.2 % | 148 | 11 |
| v3 | 78.3 % | 9 | 116 |

**v3 a un bien meilleur rappel (beaucoup moins de cuivres ratés) mais une précision plus
faible (plus de faux positifs)** que v1/v2 — compromis à surveiller pour une v4.

---

## Structure des données

```
data/
├── bois_cuivre/
│   ├── sources/                     ← pages brutes téléchargées, par édition
│   ├── segmentees/                  ← illustrations extraites par YOLO, par édition
│   └── datasets/                    ← train/val/test pour le classifieur bois/cuivre
└── bibles_mdz/
    └── segmentees/                  ← illustrations de bibles BSB/MDZ (recuperation_bsb.ipynb)

modeles/
└── bois_cuivre/
    ├── resnet50_v1.pth
    ├── resnet50_v2.pth
    └── resnet50_bois_cuivre_v3.0.0.pth

resultats/
├── csv/                              ← CSV bruts et enrichis (résultats de similarité, métriques)
├── similarite/
│   ├── Tableau_html/                 ← tableaux HTML de résultats de similarité
│   └── Stats/                        ← figures (corpus similaires, stats globales)
├── evaluation_modeles/bois_cuivre/   ← courbes, matrices de confusion, historique d'entraînement
├── Validation_cuivre_bois/
│   ├── Validation_until_1800/        ← tableau de validation experte (corpus filtré)
│   └── Validation_complet/           ← tableau de validation experte (corpus complet)
└── Datavis/                          ← visualisations générales (cartes, métriques comparatives)
```

---

## Modèles pré-entraînés utilisés

### YOLOv5 — Segmentation des illustrations

| Paramètre | Valeur |
|---|---|
| Modèle | `seglinglin/Historical-Illustration-Extraction` |
| Source | Hugging Face — https://huggingface.co/seglinglin/Historical-Illustration-Extraction |
| Architecture & code | Ultralytics — https://github.com/ultralytics/yolov5 |
| Entraîné sur | Documents historiques imprimés — manuscrits, livres anciens |
| Seuil de confiance | 0.25 |

**Rôle dans le projet :** détecte et découpe les illustrations dans les pages numérisées
des éditions des *Métamorphoses*.

**Choix du modèle :** sélectionné empiriquement parmi les modèles disponibles sur
Hugging Face pour la détection dans les documents historiques. Validé visuellement
sur les corpus bois et cuivre — aucune évaluation formelle n'a été conduite.

---

### ResNet50 — Classification bois / cuivre

| Paramètre | Valeur |
|---|---|
| Modèle de base | `ResNet50` — poids ImageNet (`ResNet50_Weights.IMAGENET1K_V1`) |
| Source | torchvision — https://pytorch.org/vision/stable/models/resnet.html |
| Fine-tuning | Couche finale remplacée — `fc(2048 → 2)` — classification binaire |

**Rôle dans le projet :** classifie chaque illustration extraite par YOLO comme
gravure sur bois ou gravure sur cuivre.

**Choix du modèle :** ResNet50 est un standard établi pour la classification
d'images. Son pré-entraînement sur ImageNet lui permet d'extraire des
caractéristiques visuelles génériques — textures, contours, structures —
directement applicables à la distinction entre les traits fins du burin (cuivre)
et les lignes plus épaisses de la taille de bois.

---

## Note mémoire GPU

YOLO (`01_dataset.ipynb`) et ResNet50 (`02/03_entrainement_*.ipynb`, `04_application_modeles.ipynb`)
ne peuvent pas coexister en mémoire GPU. Appeler `liberer_yolo()` en fin de segmentation.
Si une erreur OOM (*Out Of Memory*) survient, faire **Kernel → Restart**.

---

## Sources bibliographiques (classification bois/cuivre, v1/v2)

| Dossier | ARK | Lieu & Date | Graveur | Technique |
|---|---|---|---|---|
| `bois_salomon_rouille_lyon1557` | `btv1b2200047r` | Lyon, 1557 | Bernard Salomon | bois |
| `bois_wickram_behem_mayence1545` | `bsb10139926` | Mayence, 1545 | Jörg Wickram | bois |
| `bois_solis_feyerabend_francfort1581` | `bsb00087854` | Francfort, 1581 | Virgil Solis | bois |
| `cuivre_savery_farnaby_paris1637` | `bsb10863401` | Paris, 1637 | Clein & Savery | cuivre |

> Le dataset v3 reprend ces 4 sources et les étend à 29 éditions au total
> (voir `03_entrainement_v3.ipynb`, dictionnaire `SOURCES`).
