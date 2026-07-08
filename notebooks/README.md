# Notebooks — Gallica Images / Illustrations Ovide
**Stage Bnu Strasbourg — depuis avril 2026**
**Corpus** : Illustrations des *Métamorphoses* d'Ovide, 16e–17e siècles

Chaque sous-dossier a son propre `README.md` avec le détail de son pipeline, ses
notebooks et (le cas échéant) ses résultats. Ce fichier ne garde que les infos
générales communes à tous les axes.

---

## gallica_utils.py

Toutes les fonctions réutilisables (téléchargement IIIF, segmentation YOLO,
chargement/prédiction ResNet50, génération de tableaux HTML...) sont centralisées
dans `gallica_utils.py`, à la racine de `notebooks/`. Importer depuis chaque
notebook (profondeur uniforme `notebooks/<axe>/<fichier>.ipynb`) avec :

```python
import sys
sys.path.insert(0, "..")
from gallica_utils import charger_yolo, segmenter_corpus, charger_resnet, ...
```

---

## Dossiers

```
notebooks/
├── gallica_utils.py                     ← fonctions partagées (voir ci-dessus)
│
├── documentation_similarite_salomon/    ← recherche par similarité d'image (corpus Salomon)
├── classification_bois_cuivre/          ← classifieur ResNet50 bois / cuivre — axe clôturé
├── classification_graveur/              ← classification par graveur — en cours
├── 04_exploration/                      ← explorations ponctuelles
└── 05_annexes/                          ← tâches annexes
```

- **[`documentation_similarite_salomon/`](documentation_similarite_salomon/README.md)** — documentation de l'API BnF
  (CLIP, filtres, endpoints), collecte et analyse des résultats de similarité d'image sur
  le corpus de Bernard Salomon (1557).
- **[`classification_bois_cuivre/`](classification_bois_cuivre/README.md)** —
  classifieur ResNet50 bois/cuivre. Axe **clôturé** : v4.0.0 et v4.1.1 retenues comme
  versions finales, voir le README du dossier pour la fiche technique complète et
  l'historique des versions (v1 à v4.1.1).
- **[`classification_graveur/`](classification_graveur/README.md)** —
  classification par graveur, axe en cours (constitution du dataset).
- **[`04_exploration/`](04_exploration/README.md)** — explorations ponctuelles :
  comparaison CLIP/DINOv2, tests LLM sur l'iconographie, clustering de bibles
  illustrées, cartes de circulation, et la validation BnF exploratoire de l'axe
  bois/cuivre (déplacée ici, voir raisons dans son README).
- **[`05_annexes/`](05_annexes/README.md)** — tâches annexes (récupération de bibles
  illustrées BSB/MDZ).

---

## Base d'illustrations segmentées — partagée entre axes

`classification_bois_cuivre/01_dataset.ipynb` et
`classification_graveur/01_constitution_dataset.ipynb` alimentent tous les deux le
même réservoir d'illustrations segmentées, `data/editions_ovide/segmentees/`,
chacun avec ses propres éditions et son propre notebook de récupération. Une
édition retrouvée et segmentée par l'un des deux axes devient donc immédiatement
disponible pour l'autre — les deux notebooks se construisent mutuellement cette
base commune plutôt que de la dupliquer chacun de leur côté.

---

## Note mémoire GPU (générale)

YOLO et ResNet50 ne peuvent pas coexister en mémoire GPU dans le même notebook.
Appeler `liberer_yolo()` (voir `gallica_utils.py`) avant de charger un modèle
ResNet50. Si une erreur OOM (*Out Of Memory*) survient, faire **Kernel → Restart**.
