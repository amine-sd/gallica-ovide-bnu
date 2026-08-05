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
| **Classification bois / cuivre** | Clôturé (v4.1.1) | Classifieur ResNet50 distinguant gravure sur bois et gravure sur cuivre, 4 versions entraînées et comparées, validées par une experte (Céline) |
| **Classification par graveur** | En cours | Constitution du jeu de données ; premier test exploratoire (embeddings gelés DINOv2/SigLIP) concluant, fine-tuning à venir |
| **Corpus Bibles (BSB/MDZ)** | Mature | Récupération de Bibles illustrées, tri visuel, puis classification par thème iconographique (atelier web) — second corpus utilisé par le mini RAG |
| **Mini RAG iconographique** | Exploratoire | Recherche vectorielle (SigLIP, hybride avec un index texte) sur Bibles + Ovide, génération de réponse par LLM local (Ollama, texte et vision) — interface de démo Gradio |
| **Comparaison Ovide / Bibles** | Exploratoire | Proximité stylistique entre les deux corpus par embeddings SigLIP (projection UMAP par thème, galerie des meilleures correspondances) — analyse à part, hors pipeline du RAG |
| **Visualisations** | Mature | Cartes et frises de la circulation des éditions, graveurs et plaques gravées entre éditeurs (carte animée, graphe en réseau D3, ateliers web de correction) |

## Structure du dépôt

```
working_dir/
├── notebooks/        ← tout le code (voir notebooks/README.md pour le détail du pipeline)
├── docs/               ← documentation détaillée de chaque axe (1 fichier par dossier de notebooks/)
├── data/              ← données (brutes, segmentées, datasets) — volumineux, exclu de Git
│   ├── bibles_mdz/     ← corpus de bibles MDZ/BSB
│   └── editions_ovide/ ← sources, illustrations segmentées et datasets (bois/cuivre, graveur)
├── modeles/            ← poids des modèles entraînés (.pth) — exclus de Git
├── resultats/          ← CSV, tableaux HTML de validation, visualisations, métriques
├── retours_celine/     ← annotations de validation expertes (Céline)
└── yolov5_repo/        ← dépendance externe (segmentation d'illustrations), à cloner séparément
```

## Installation

```
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate sous Windows
pip install -r requirements.txt
git clone https://github.com/ultralytics/yolov5 yolov5_repo
```

`requirements.txt` est un `pip freeze` de l'environnement du stage (GPU CUDA) — voir son
en-tête pour l'installation sans GPU. Les poids du modèle de segmentation
(`seglinglin/Historical-Illustration-Extraction`, fine-tuné sur ce corpus) se téléchargent
automatiquement depuis Hugging Face au premier appel de `charger_yolo()` — rien à
télécharger à la main.

## Outils et modèles utilisés

- **YOLOv5** (`seglinglin/Historical-Illustration-Extraction`) — segmentation des illustrations dans les pages numérisées
- **ResNet50** (torchvision, pré-entraîné ImageNet) — classification bois/cuivre, fine-tuné en 4 versions
- **CLIP** — embeddings d'image via l'API de recherche de la BnF, et clustering exploratoire (corpus Bibles)
- **DINOv2** — embeddings pour la classification par graveur (test exploratoire, embeddings gelés sans fine-tuning)
- **SigLIP** — embeddings image + texte dans un espace partagé, base du mini RAG iconographique et de la comparaison Ovide/Bibles
- **Ollama (llama3.2, Qwen2.5-VL) / multilingual-e5-small** — génération de réponse (texte et vision) et recherche hybride pour le mini RAG
- **UMAP** — projection 2D des embeddings SigLIP pour la comparaison Ovide/Bibles

## Documentation détaillée

Voir [notebooks/README.md](notebooks/README.md) pour le détail du pipeline, l'ordre
d'exécution des notebooks et les fiches techniques des modèles.
