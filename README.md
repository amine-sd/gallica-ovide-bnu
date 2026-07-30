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
| **Mini RAG iconographique** | Exploratoire | Recherche vectorielle (SigLIP, hybride avec un index texte) sur Bibles + Ovide, génération de réponse par LLM local (Ollama, texte et vision) — interface de démo Gradio |
| **Explorations complémentaires** | Exploratoire | Comparaison CLIP / DINOv2, description iconographique par LLM (Claude / Ollama), récupération de corpus BSB |

## Structure du dépôt

```
working_dir/
├── notebooks/        ← tout le code (voir notebooks/README.md pour le détail du pipeline)
├── docs/               ← documentation detaillee de chaque axe (1 fichier par dossier de notebooks/)
├── data/              ← données (brutes, segmentées, datasets) — volumineux, exclu de Git
│   ├── bibles_mdz/     ← corpus de bibles MDZ/BSB
│   └── editions_ovide/ ← sources, illustrations segmentées et datasets (bois/cuivre, graveur)
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
- **SigLIP** — embeddings image + texte dans un espace partagé, base du mini RAG iconographique
- **Ollama (llama3.2, Qwen2.5-VL) / multilingual-e5-small** — génération de réponse (texte et vision) et recherche hybride pour le mini RAG

## Documentation détaillée

Voir [notebooks/README.md](notebooks/README.md) pour le détail du pipeline, l'ordre
d'exécution des notebooks et les fiches techniques des modèles.
