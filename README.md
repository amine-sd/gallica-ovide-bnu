# Gallica Images — BNU Strasbourg

**Stage — BNU Strasbourg, depuis avril 2026**
**Corpus** : illustrations des éditions imprimées des *Métamorphoses* d'Ovide (16e–17e s.), numérisées via Gallica (BnF) et la Bayerische Staatsbibliothek (MDZ/BSB).

---

## Objectif du stage

Étudier la circulation et la réutilisation des illustrations gravées dans les éditions
anciennes des *Métamorphoses* d'Ovide, à l'aide d'outils de vision par ordinateur :
- retrouver des illustrations visuellement similaires à travers le corpus (recherche par similarité d'image),
- caractériser ces illustrations (technique de gravure, graveur, thème iconographique) par classification automatique,
- faciliter la validation experte des résultats produits par les modèles.

## Axes de travail

| Axe | État | Description |
|---|---|---|
| **Recherche par similarité** | Mature | Recherche d'illustrations similaires à un corpus de référence (gravures de Bernard Salomon, Lyon 1557) via l'API de recherche d'image de la BnF (embeddings CLIP), étendue à d'autres corpus (Wickram, Mayence 1545) |
| **Classification bois / cuivre** | Mature (v3) | Classifieur ResNet50 distinguant gravure sur bois et gravure sur cuivre, 3 versions entraînées et comparées, validées par une experte (Céline) |
| **Classification par graveur** | En cours | Constitution du jeu de données ; entraînement non encore réalisé |
| **Explorations complémentaires** | Exploratoire | Comparaison CLIP / DINOv2, description iconographique par LLM (Claude / Ollama), récupération de corpus BSB |

## Résultats clés (classification bois/cuivre)

Trois versions du classifieur ResNet50 ont été entraînées sur des jeux de données
croissants (v1 : 744 images → v3 : dataset enrichi à 29 éditions). Validation contre
les annotations expertes de Céline sur les résultats de similarité appliqués au
corpus Salomon :

| Version | Précision cuivre | Cuivres ratés (faux négatifs) | Faux cuivres (faux positifs) |
|---|---|---|---|
| v1 | 91.3 % | 112 | 14 |
| v2 | 91.2 % | 148 | 11 |
| v3 | 78.3 % | 9 | 116 |

**v3 détecte beaucoup plus de cuivres (meilleur rappel) mais au prix d'une précision
plus faible** — compromis précision/rappel à surveiller pour les prochaines versions.
Détail des calculs : [notebooks/modeles/bois_cuivre/stats.ipynb](notebooks/modeles/bois_cuivre/stats.ipynb).

## Structure du dépôt

```
working_dir/
├── notebooks/        ← tout le code (voir notebooks/README.md pour le détail du pipeline)
├── data/              ← données (brutes, segmentées, datasets) — volumineux, exclu de Git
│   ├── bibles_mdz/     ← corpus de bibles MDZ/BSB
│   └── bois_cuivre/    ← sources, illustrations segmentées et dataset d'entraînement bois/cuivre
├── modeles/            ← poids des modèles entraînés (.pth) — exclus de Git
├── resultats/          ← CSV, tableaux HTML de validation, visualisations, métriques
├── retours_celine/     ← annotations de validation expertes (Céline)
└── yolov5_repo/        ← dépendance externe (segmentation d'illustrations), à cloner séparément
```

## Outils et modèles utilisés

- **YOLOv5** (`seglinglin/Historical-Illustration-Extraction`) — segmentation des illustrations dans les pages numérisées
- **ResNet50** (torchvision, pré-entraîné ImageNet) — classification bois/cuivre, fine-tuné en 3 versions
- **CLIP** — embeddings d'image via l'API de recherche de la BnF
- **DINOv2** — comparé à CLIP pour la recherche par similarité (exploratoire)
- **Claude / Ollama (llava, llama3.2)** — description iconographique automatique (exploratoire)

## Documentation détaillée

Voir [notebooks/README.md](notebooks/README.md) pour le détail du pipeline, l'ordre
d'exécution des notebooks et les fiches techniques des modèles.
