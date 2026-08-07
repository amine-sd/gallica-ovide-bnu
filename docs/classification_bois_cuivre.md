# Classification bois / cuivre

Classifieur ResNet50 binaire — distingue gravure sur bois et gravure
sur cuivre parmi les illustrations segmentées des *Métamorphoses*
d'Ovide.

Versions finales retenues : **v4.0.0** et **v4.1.1** (voir plus bas).

---

## Notebooks

| Fichier | Rôle |
|---|---|
| `01_dataset.ipynb` | Téléchargement des pages, segmentation YOLO, split train/val/test (par édition, voir v4) |
| `02_entrainement.ipynb` | Notebook générique — `VERSION` + `STRATEGIE` en config, relancé pour toutes les versions |
| `03_application_modeles.ipynb` | Applique les modèles aux résultats de similarité du corpus Salomon → CSV enrichi |
| `04_validation_experte.ipynb` | Génère les tableaux HTML de validation soumis à Céline Bohnert |
| `05_evaluation_versions.ipynb` | Compare toutes les versions face aux annotations de Céline — **évaluation de référence** |


### Ordre d'exécution

```
01_dataset → 02_entrainement (VERSION + STRATEGIE à adapter)
           → 03_application_modeles → 04_validation_experte → 05_evaluation_versions
```

---

## Pipeline — comment étendre

### Ajouter une nouvelle source bois/cuivre (`01_dataset.ipynb`)
Appeler `telecharger_pages_iiif()` avec la nouvelle URL manifest puis
`segmenter_corpus()` avec le dossier de destination — nommer les dossiers selon la
convention `{classe}_{graveur}_{ville}{année}/`. Puis ajouter l'édition au
dictionnaire `SOURCES` (section 7) pour qu'elle entre dans le split.

### Entraîner une nouvelle version (`02_entrainement.ipynb`)
Changer `VERSION` et `STRATEGIE` dans la cellule de configuration :
- `STRATEGIE = "simple"` — entraînement en une passe, réseau dégelé dès le départ (utilisé pour v1/v2).
- `STRATEGIE = "deux_etapes"` — couche finale puis réseau complet (ou `layer4`+`fc` seulement si `DEGEL_ETAPE2="layer4"`), early stopping (utilisé pour v3/v4).



### Ajouter une version au tableau enrichi (`03_application_modeles.ipynb`)
Ajouter le chemin du nouveau `.pth` dans le dictionnaire `MODELES`. Les colonnes et
filtres HTML s'ajoutent automatiquement.

---

## Historique des versions

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

**Comparaison des 3 premières versions face aux annotations expertes de Céline**
(résultats de similarité appliqués au corpus Salomon — voir `05_evaluation_versions.ipynb`) :

| Version | Précision cuivre | Cuivres ratés (FN) | Faux cuivres (FP) |
|---|---|---|---|
| v1 | 91.3 % | 112 | 14 |
| v2 | 91.2 % | 148 | 11 |
| v3 | 78.3 % | 9 | 116 |

**v3 a un bien meilleur rappel (beaucoup moins de cuivres ratés) mais une précision
plus faible (plus de faux positifs)** que v1/v2 — compromis à surveiller pour une v4.

**Défaut découvert plus tard :** le split train/val/test de v3 se faisait par tirage
aléatoire des *images individuelles*, pas par édition — des illustrations d'un même
livre pouvaient donc se retrouver à la fois en train et en test. Ça explique
l'écart énorme entre son score interne (100%, F1≈0.888) et son score sur données
réelles indépendantes. Corrigé en v4 (voir ci-dessous).

---

### Version 4 — split corrigé, trois variantes (v4.0.0 / v4.1.0 / v4.1.1)

**Socle commun aux trois** :

| Paramètre | Valeur |
|---|---|
| Architecture | ResNet50 — fine-tuning en 2 étapes |
| Dataset | 22 éditions cuivre + 5 éditions bois, 1629 illustrations (456 bois / 1173 cuivre) |
| Correction de fuite | Split train/val/test **par édition entière** (jamais par image) — voir `01_dataset.ipynb`, fonction `repartir_editions_par_groupe()` |
| Doublons retirés | 3 tomes `ht_molin` fusionnés en une édition ; `goltzius_couleur` exclue (doublon exact de `goltzius`, même scan recoloré) ; un seul exemplaire gardé par graveur réimprimé (`baur`, `depasse`, `gaultier`, `tempesta` — plaques de cuivre probablement réutilisées d'un tirage à l'autre) |
| Rééquilibrage | `WeightedRandomSampler` sur le train (bois/cuivre vus à fréquence égale par epoch) + `TRANSFORM_TRAIN` (flip, rotation, jitter) — pas de poids de classe dans la loss (éviterait une double compensation) |
| Seed | fixé (42) — runs reproductibles et comparables entre eux |
| Split train/val/test | 1169 / 227 / 233 |

**Ce qui distingue les trois variantes (étape 2 du fine-tuning uniquement)** :

| Version | Étape 2 | Test interne |
|---|---|---|
| `resnet50_bois_cuivre_v4.0.0.pth` | Réseau complet dégelé, `LR=1e-4`, pas de weight_decay | **98.3 %** (F1 ≈ 0.98) |
| `resnet50_bois_cuivre_v4.1.0.pth` | Réseau complet dégelé, `LR=5e-5`, `weight_decay=1e-4` | 96.6 % — moins bon, écarté |
| `resnet50_bois_cuivre_v4.1.1.pth` | Dégel partiel (`layer4` + `fc` seulement), `LR=1e-4`, pas de weight_decay | **98.3 %**, meilleur rappel bois (98.4 % vs 95.1 % pour v4.0.0) |

**Comparaison finale face aux annotations expertes de Céline**
(désaccords + accords, dénominateur commun aux 6 versions — voir
`05_evaluation_versions.ipynb` → `resultats/csv/metriques_v1v2v3v4.csv`) :

| Version | Accuracy désaccords | Précision (cuivre) | Rappel (cuivre) | F1 |
|---|---|---|---|---|
| v1 | 62.0 % | 91.8 % | 81.3 % | 0.862 |
| v2 | 48.0 % | 91.9 % | 72.6 % | 0.811 |
| v3 | 60.5 % | 79.5 % | 98.8 % | 0.881 |
| **v4.0.0** | 66.1 % | 81.3 % | **99.4 %** | **0.894** |
| v4.1.0 | 63.5 % | 82.5 % | 95.6 % | 0.886 |
| **v4.1.1** | **67.9 %** | **84.7 %** | 94.4 % | 0.893 |

*Note méthodologique* : "Précision"/"Rappel" ci-dessus ne portent que sur la classe
**cuivre** (pas de vérité bois fiable côté accords, validés par titre par Céline, pas
par image — voir le commentaire détaillé dans `05_evaluation_versions.ipynb`). Le
dénominateur (images retenues) est strictement identique pour les 6 versions à
chaque exécution — nécessaire car ce notebook retélécharge les images depuis l'API
Gallica à chaque run, et des échecs réseau isolés faisaient significativement bouger
les chiffres d'un run à l'autre avant ce correctif.

**v4.0.0 et v4.1.1 dépassent nettement v3 et toutes les versions précédentes sur
F1** — quasi ex æquo entre eux, avec deux profils différents :
- **v4.0.0** : meilleur rappel (99.4 %, ne rate quasiment aucun cuivre) — à privilégier si l'objectif est de ne rien manquer.
- **v4.1.1** : meilleure précision et meilleure accuracy globale (le moins de fausses alertes cuivre) — à privilégier si l'objectif est d'éviter les faux positifs. Entraîne aussi beaucoup moins de paramètres (dégel partiel), donc plus stable et moins coûteux.

Contrairement à v3, ces scores ne sont pas gonflés par une fuite de données : le
split par édition garantit qu'aucune image de test n'a été vue, même partiellement,
pendant l'entraînement — la comparaison est donc fiable, pas seulement optimiste.

**Variante écartée — v4.1.0** : moins bonne que v4.0.0 et v4.1.1 sur toutes les
mesures. Le réglage testé (LR réduit + weight_decay) n'a pas aidé sur ce dataset.
Conservée dans `modeles/bois_cuivre/` à titre de comparaison, non retenue.

**Limite connue, non résolue par v4** : le bois ne repose en réalité que sur **4
sources visuelles indépendantes** (`bois_solis_feyerabend_francfort1581` est une
copie en miroir de `bois_salomon_rouille_lyon1557` — pratique de copie de gravure
documentée) sur les 5 éditions disponibles. Toute évaluation sur du bois totalement
hors du corpus Ovide doit être interprétée avec cette réserve. 

**Axe clôturé sur cette base** — v4.0.0 et v4.1.1 sont les deux candidats retenus, à
choisir selon la priorité (rappel vs précision) du cas d'usage.

---

## Structure des données

```
data/
└── bois_cuivre/
    ├── sources/                     ← pages brutes téléchargées, par édition
    ├── segmentees/                  ← illustrations extraites par YOLO, par édition
    ├── datasets/                    ← train/val/test pour le classifieur (reconstruit à chaque run de 01_dataset.ipynb)
    └── test_bnf_etiquete/           ← corpus BnF étiqueté (voir validation_bnf_etiquete.ipynb, ce dossier)

modeles/
└── bois_cuivre/
    ├── resnet50_v1.pth
    ├── resnet50_v2.pth
    ├── resnet50_bois_cuivre_v3.0.0.pth
    ├── resnet50_bois_cuivre_v4.0.0.pth      ← version retenue (meilleur rappel)
    ├── resnet50_bois_cuivre_v4.1.0.pth      ← variante testée, non retenue
    └── resnet50_bois_cuivre_v4.1.1.pth      ← version retenue (meilleure précision)

resultats/
├── csv/                              ← CSV bruts et enrichis (résultats de similarité, métriques)
├── similarite/                       ← tableaux HTML et figures de résultats de similarité (à plat)
├── evaluation_modeles/bois_cuivre/   ← courbes, matrices de confusion, historique d'entraînement, trace du split (split_v4_editions.json)
├── Validation_cuivre_bois/
│   ├── Validation_until_1800/        ← tableau de validation experte (corpus filtré)
│   └── Validation_complet/           ← tableau de validation experte (corpus complet)
└── Datavis/                          ← visualisations générales (métriques comparatives)

retours_celine/                       ← annotations manuelles de Céline Bohnert (gitignoré)
├── validation_celine_desaccords_complet.csv   ← vérité par image (désaccords v1/v2/v3)
├── validation_titres_celine.csv               ← vérité par titre (accord v1+v2+v3=cuivre)
└── validation_titres_celine_complet.csv
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

**Rôle dans le projet :** détecte et découpe les illustrations dans les pages
numérisées des éditions des *Métamorphoses*.

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

