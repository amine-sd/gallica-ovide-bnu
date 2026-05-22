# Notebooks — Gallica Images / Illustrations Ovide
**Stage Bnu Strasbourg — Avril 2026**   
**Corpus** : Illustrations des *Métamorphoses* d'Ovide, 16e–17e siècles

---

## Architecture

```
notebooks/
├── utils.py                       ← fonctions partagées (importer dans chaque notebook)
├── 01_documentation_api.ipynb     ← documentation des endpoints de l'API BnF
├── 02_collecte_similarite.ipynb   ← collecte des résultats via l'API → CSV brut
├── 03_analyse_tableau.ipynb       ← statistiques + tableau HTML sans IA
├── 04_dataset_segmentation.ipynb  ← YOLO → illustrations segmentées par source
├── 05_classification_resnet.ipynb ← fine-tuning ResNet50 → modèle .pth
└── 06_tableau_enrichi.ipynb       ← application des modèles → tableau HTML final
```

---

## Ordre d'exécution

```
02 → 03                  (analyse et tableau sans IA)
02 → 04 → 05 → 06        (pipeline complet avec IA)
```

---

## Pipeline

### utils.py
Toutes les fonctions réutilisables sont centralisées dans `gallica_utils.py`.
Importer dans chaque notebook avec :
```python
import sys
sys.path.insert(0, "..")
from utils import charger_yolo, segmenter_corpus, charger_resnet, ...
```

### Ajouter une nouvelle source bois/cuivre (notebook 04)
Appeler `telecharger_pages_iiif()` avec la nouvelle URL manifest puis
`segmenter_corpus()` avec le dossier de destination — nommer les dossiers
selon la convention `{classe}_{graveur}_{ville}{année}/`.

### Entraîner une nouvelle version du modèle (notebook 05)
Changer uniquement `VERSION = "vN"` dans la cellule de configuration.

### Ajouter une version au tableau enrichi (notebook 06)
Ajouter le chemin du nouveau `.pth` dans le dictionnaire `MODELES`.
Les colonnes et filtres HTML s'ajoutent automatiquement.

---

## Modèles de classification bois / cuivre

### Version 1 — `resnet50_bois_cuivre_v1.pth`

| Paramètre | Valeur |
|---|---|
| Architecture | ResNet50 fine-tuné (ImageNet → bois/cuivre) |
| Dataset | 744 images — 400 bois / 344 cuivre |
| Data augmentation | Aucune |
| Split train/val/test | 520 / 111 / 113 |
| Test accuracy | **100%** |

**Sources bois :** Salomon Lyon 1557 (163), Mayence 1545 (50), BSB 87854 (187)  
**Sources cuivre :** Munich Paris 1637 (21), 4 PDFs Gallica (323)

**Remarque :** Le 100% sur un petit dataset homogène peut indiquer un surajustement.
Le modèle a tendance à être très confiant mais moins sensible aux cas limites.
Appliqué aux 1840 (en prenant que 10 rows) résultats de similarité Salomon → **33 cuivres détectés**.

---

### Version 2 — `resnet50_bois_cuivre_v2.pth`

| Paramètre | Valeur |
|---|---|
| Architecture | ResNet50 fine-tuné (ImageNet → bois/cuivre) |
| Dataset | 1506 images — 862 bois / 644 cuivre |
| Data augmentation | Flip horizontal — dossiers `{source}_flip/` |
| Split train/val/test | 1052 / 112 / 115 |
| Test accuracy | **98%** |

**Sources bois :** Salomon Lyon 1557 (192), Solis Francfort 1581 (187), Wickram Mayence 1545 (52)  
**Sources cuivre :** Clein Paris 1637 (18), Crispin de Passe — *Metamorphoseon* (135), Crispin de Passe — *Nasonis* (136), Renouard *traduites* (16), Renouard *traduittes* (17)

**Remarque :** Le 98% sur un dataset plus grand et plus varié est plus fiable que
le 100% de v1. Les 2 erreurs sont des faux négatifs cuivre — des gravures sur cuivre
confondues avec du bois. Quand le modèle prédit cuivre, il a toujours raison (précision = 1.00).


---

## Structure des données

```
data/
├── images_brutes/                  ← pages originales téléchargées
│   ├── bois_salomon_lyon1557/
│   ├── bois_solis_francfort1581/
│   ├── bois_wickram_mayence1545/
│   ├── cuivre_clein_paris1637/
│   └── cuivre_pdf/{edition}/
│
├── illustrations_segmentees/       ← illustrations extraites par YOLO
│   ├── bois_salomon_lyon1557/
│   ├── bois_salomon_lyon1557_flip/
│   ├── bois_solis_francfort1581/
│   ├── bois_solis_francfort1581_flip/
│   ├── bois_wickram_mayence1545/
│   ├── bois_wickram_mayence1545_flip/
│   ├── cuivre_clein_paris1637/
│   ├── cuivre_clein_paris1637_flip/
│   └── cuivre_pdf/
│       ├── cuivre_passe_metamorphoseon/
│       ├── cuivre_passe_metamorphoseon_flip/
│       ├── cuivre_passe_nasonis/
│       ├── cuivre_passe_nasonis_flip/
│       ├── cuivre_renouard_traduites/
│       ├── cuivre_renouard_traduites_flip/
│       ├── cuivre_renouard_traduittes/
│       └── cuivre_renouard_traduittes_flip/
│
├── pdf_cuivre/                     ← PDFs sources (accès restreint sur Gallica)
└── dataset/
    ├── bois/                       ← toutes les images bois rassemblées
    ├── cuivre/                     ← toutes les images cuivre rassemblées
    ├── train/bois/ train/cuivre/
    ├── val/bois/   val/cuivre/
    └── test/bois/  test/cuivre/

modeles/
├── resnet50_bois_cuivre_v1.pth
└── resnet50_bois_cuivre_v2.pth

resultats/
├── similarite_salomon_brut.csv
├── similarite_salomon_enrichi.csv
├── tableau_similarite_salomon.html          ← sans colonnes IA
├── tableau_similarite_salomon_enrichi.html  ← avec colonnes IA v1/v2
├── matrice_confusion_v1.png
└── matrice_confusion_v2.png
```

---

## Modèles pré-entraînés utilisés

### YOLOv5 — Segmentation des illustrations

| Paramètre | Valeur |
|---|---|
| Modèle | `seglinglin/Historical-Illustration-Extraction` |
| Source | Hugging Face — https://huggingface.co/seglinglin/Historical-Illustration-Extraction |
| Architecture & code | Ultralytics — https://github.com/ultralytics/yolov5 |
| Poids fine-tunés | seglinglin — https://huggingface.co/seglinglin/Historical-Illustration-Extraction |
| Entraîné sur | Documents historiques imprimés — manuscrits, livres anciens|
| Fichier utilisé | `illustration_extraction.pt` |

| Auteur du modèle de base | Glenn Jocher — Ultralytics (YOLOv5) |
| Papier original YOLO | Redmon et al., *You Only Look Once*, CVPR 2016 |
| Licence | Publique — Hugging Face |


**Rôle dans le projet :** détecte et découpe les illustrations dans les pages numérisées
des éditions des *Métamorphoses*. Utilisé avec un seuil de confiance de 0.25 —
les détections en dessous de ce seuil sont ignorées.

**Choix du modèle :** sélectionné empiriquement parmi les modèles disponibles sur
Hugging Face pour la détection dans les documents historiques. Validé visuellement
sur les corpus bois et cuivre — aucune évaluation formelle n'a été conduite.

---

### ResNet50 — Classification bois / cuivre

| Paramètre | Valeur |
|---|---|
| Modèle de base | `ResNet50` — poids ImageNet (`ResNet50_Weights.IMAGENET1K_V1`) |
| Source | torchvision — https://pytorch.org/vision/stable/models/resnet.html |
| Architecture | Réseau convolutif profond à connexions résiduelles, 50 couches |
| Pré-entraîné sur | ImageNet — 1,2 million d'images, 1000 classes |
| Auteurs | He, Zhang, Ren, Sun — Microsoft Research, CVPR 2016 |
| Poids distribués par | torchvision — Meta AI |
| Fine-tuning | Couche finale remplacée — `fc(2048 → 2)` — classification binaire |

**Rôle dans le projet :** classifie chaque illustration extraite par YOLO comme
gravure sur bois ou gravure sur cuivre. Fine-tuné sur notre dataset de gravures
des *Métamorphoses*.

| Hyperparamètre | Valeur |
|---|---|
| Optimiseur | Adam, lr=1e-4 |
| Scheduler | StepLR — step=5, gamma=0.5 |
| Loss | CrossEntropyLoss avec class weights |
| Époques | 20 |
| Batch size | 16 |

**Choix du modèle :** ResNet50 est un standard établi pour la classification
d'images. Son pré-entraînement sur ImageNet lui permet d'extraire des
caractéristiques visuelles génériques — textures, contours, structures —
directement applicables à la distinction entre les traits fins du burin (cuivre)
et les lignes plus épaisses de la taille de bois.

**Microsoft Research** → a inventé l'architecture ResNet
**Meta AI / torchvision** → distribue les poids pré-entraînés qu'on utilise

---


## Note mémoire GPU

YOLO (notebook 04) et ResNet50 (notebooks 05/06) ne peuvent pas coexister
en mémoire GPU. Le notebook 04 appelle `liberer_yolo()` en fin d'exécution.
Si une erreur OOM (*Out Of Memory* lancé par PyTorch) survient dans 05 ou 06, faire **Kernel → Restart**.

---

## Sources bibliographiques

| Dossier | ARK | Lieu & Date | Graveur | Technique |
|---|---|---|---|---|
| `bois_salomon_lyon1557` | `btv1b2200047r` | Lyon, 1557 | Bernard Salomon | bois |
| `bois_wickram_mayence1545` | `bsb10139926` | Mayence, 1545 | Jörg Wickram | bois |
| `bois_solis_francfort1581` | `bsb00087854` | Francfort, 1581 | Virgil Solis | bois |
| `cuivre_clein_paris1637` | `bsb10863401` | Paris, 1637 | Clein & Savery | cuivre |
| `cuivre_passe_metamorphoseon` | `bpt6k15218623` | — | Crispin de Passe | cuivre |
| `cuivre_passe_nasonis` | `bpt6k1522448r` | — | Crispin de Passe | cuivre |
| `cuivre_renouard_traduites` | `bpt6k6277348n` | — | non renseigné | cuivre |
| `cuivre_renouard_traduittes` | `bpt6k722055` | — | non renseigné | cuivre |

> Les ARKs `bpt6k*` correspondent à des PDFs téléchargés manuellement depuis Gallica —
> ces documents ont un accès IIIF restreint (HTTP 403) et ne sont pas indexés
> dans l'API Fouille d'image BnF.