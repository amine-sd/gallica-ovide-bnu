# Notebooks — Gallica Images / Illustrations Ovide
**Stage Bnu Strasbourg — depuis avril 2026**
**Corpus** : Illustrations des *Métamorphoses* d'Ovide, 16e–17e siècles

La documentation détaillée de chaque axe (pipeline, notebooks, résultats) vit dans
[`docs/`](../docs/), un fichier par axe, nommé d'après son dossier. Ce fichier-ci
ne garde que les infos générales communes à tous les axes.

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
├── corpus_bibles/                       ← récupération, tri et classification par thème des Bibles (BSB/MDZ)
├── mini_rag_iconographique/             ← mini RAG iconographique (recherche + génération) — exploratoire
├── comparaison_ovide_bibles/             ← proximité par embeddings (UMAP) — à part du RAG
└── visualisations/                      ← notebooks de visualisations spécifiques (cartes, frises...)
```

- **[`documentation_similarite_salomon/`](../docs/documentation_similarite_salomon.md)** — documentation de l'API BnF
  (CLIP, filtres, endpoints), collecte et analyse des résultats de similarité d'image sur
  le corpus de Bernard Salomon (1557).
- **[`classification_bois_cuivre/`](../docs/classification_bois_cuivre.md)** —
  classifieur ResNet50 bois/cuivre. Axe **clôturé** : v4.0.0 et v4.1.1 retenues comme
  versions finales, voir sa fiche dans `docs/` pour le détail technique complet et
  l'historique des versions (v1 à v4.1.1).
- **[`classification_graveur/`](../docs/classification_graveur.md)** —
  classification par graveur, axe en cours (constitution du dataset).
- **[`corpus_bibles/`](../docs/corpus_bibles.md)** — récupération de Bibles illustrées
  depuis BSB/MDZ, tri visuel, puis classification par thème iconographique (atelier
  web) : le corpus Bibles utilisé par `mini_rag_iconographique/`.
- **[`mini_rag_iconographique/`](../docs/mini_rag_iconographique.md)** — mini RAG :
  recherche vectorielle (SigLIP, hybride avec un index texte e5) sur les bases
  Bibles + Ovide, génération de réponse par LLM local (Ollama, texte et vision),
  interface de démo Gradio (`app_rag.py`). Axe exploratoire.
- **[`comparaison_ovide_bibles/`](../docs/comparaison_ovide_bibles.md)** — proximité
  Ovide/Bibles par embeddings SigLIP (projection UMAP, correspondances par thème) — à
  part du RAG, ne fait pas partie de son pipeline.
- **[`visualisations/`](../docs/visualisations.md)** — notebooks produisant chacun une
  visualisation HTML autonome. Pour l'instant : cartes et frises chronologiques de la
  circulation des éditions (villes, graveurs, copies), à partir du corpus de référence
  de Céline Bohnert (`retours_celine/BNU_corpus.ods`).

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
