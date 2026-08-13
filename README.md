# Projet Ovide en images à la BNU

**Outils d'intelligence artificielle pour l'étude des gravures accompagnant les
*Métamorphoses* d'Ovide.**

---

## Quoi ? — ce que contient ce dépôt

Une chaîne d'outils de vision par ordinateur appliquée à un corpus patrimonial : elle part
de pages numérisées d'éditions anciennes et en produit un corpus d'illustrations
interrogeable — segmenté, décrit automatiquement, et explorable par similarité visuelle,
par langage naturel ou par visualisation interactive.

```
Pages numérisées  →  Segmentation  →  Illustrations isolées  →  ┬─ classification bois/cuivre
(Gallica, BSB/MDZ)                                              ├─ classification par graveur
                                                                ├─ recherche par similarité
                                                                └─ mini RAG iconographique
                                                                       ↓
                                                            validation experte (C. Bohnert)
                                                            → corrections réinjectées
```

Deux principes structurent l'ensemble du code. **La segmentation est l'étape pivot** : une
gravure n'occupe souvent qu'une fraction de sa page, et tout le reste de la chaîne travaille
sur des illustrations isolées, jamais sur des pages entières. **Aucun résultat de modèle
n'est considéré comme acquis sans validation experte** : les interfaces web de correction
livrées avec le code enregistrent les corrections de la chercheuse *à part*, sans jamais
modifier la donnée source.

Le travail se répartit en sept axes, chacun documenté par une fiche technique :

| Axe | État | Fiche technique |
|---|---|---|
| Recherche par similarité | Mature | [`documentation_similarite_salomon.md`](docs/documentation_similarite_salomon.md) |
| Classification bois / cuivre | Clôturé | [`classification_bois_cuivre.md`](docs/classification_bois_cuivre.md) |
| Classification par graveur | En cours | [`classification_graveur.md`](docs/classification_graveur.md) |
| Corpus Bibles (BSB/MDZ) | Mature | [`corpus_bibles.md`](docs/corpus_bibles.md) |
| Mini RAG iconographique | Exploratoire | [`mini_rag_iconographique.md`](docs/mini_rag_iconographique.md) |
| Comparaison Ovide / Bibles | Exploratoire | [`comparaison_ovide_bibles.md`](docs/comparaison_ovide_bibles.md) |
| Visualisations | Mature | [`visualisations.md`](docs/visualisations.md) |

Chaque fiche détaille le pipeline de l'axe, ses choix méthodologiques, ses résultats
chiffrés et ses **limites connues**.

## Pourquoi ? — le problème traité

Entre le 16e et le 18e siècle, les éditions imprimées des *Métamorphoses* ont donné lieu à
des milliers de gravures illustrant les mêmes épisodes mythologiques. Ces images ne sont pas
réinventées à chaque édition : les plaques gravées se revendent d'un imprimeur à l'autre, se
rééditent des décennies plus tard, et les ateliers concurrents copient les compositions à
succès, parfois en les inversant en miroir.

Cette circulation iconographique est documentée de longue date par les spécialistes du livre
ancien, mais jamais à l'échelle d'un corpus complet : repérer les motifs récurrents, les
filiations entre graveurs et les recoupements avec d'autres corpus suppose de comparer des
milliers d'images entre elles. Fait à l'œil, ce travail reste nécessairement partiel.

Ce constat s'inscrit dans un manque plus large : les fonds anciens illustrés restent
nettement sous-représentés dans les *visual studies* numériques, à la différence d'objets
plus anciens (pièces archéologiques, manuscrits enluminés) ou plus récents (fonds
photographiques, presse imprimée, médias contemporains) qui bénéficient d'une attention
numérique croissante.

Les outils rassemblés ici visent donc à rendre ce corpus interrogeable à son échelle réelle,
pour instruire trois questions : comment circulent les éditions et les plaques gravées d'une
ville et d'un atelier à l'autre ? quelles relations lient éditeurs et graveurs ? et dans
quelle mesure l'iconographie ovidienne recoupe-t-elle l'imagerie biblique de la même période
— d'où le second corpus de Bibles illustrées constitué en cours de projet ?

## Qui ? — le cadre et l'équipe

Ce dépôt est le volet technique du stage que j'ai effectué d'avril à août 2026 à la
Bibliothèque nationale et universitaire de Strasbourg, dans le cadre de ma formation
d'ingénieur à l'ENSICAEN (parcours ISIA — Image, Son et Intelligence Artificielle).

Le stage s'inscrit dans la résidence de recherche *« Ovide en images à la BNU : de l'imprimé
ancien à la donnée »*, menée par une équipe pluridisciplinaire :

- **Céline Bohnert** — maîtresse de conférences (CRIMEL, Université de Reims
  Champagne-Ardenne), chercheuse en résidence : porteuse des questions scientifiques et
  autorité de validation des résultats produits par les modèles ;
- **Rosanne Wingert** — responsable Bibliothèques & Données numériques (BNU), tutrice de
  stage : cadrage du projet et interface avec les partenaires ;
- **moi-même, Amine Saidi** — ingénieur stagiaire, en charge de l'ensemble du volet
  technique.


## Où ? — se repérer

```
working_dir/
├── notebooks/          ← tout le code (voir notebooks/README.md pour le pipeline)
├── docs/               ← fiche technique détaillée par axe (1 fichier par dossier de notebooks/)
├── data/               ← données brutes, segmentées et jeux d'entraînement — exclu de Git
│   ├── editions_ovide/ ← sources, illustrations segmentées, datasets (bois/cuivre, graveur)
│   └── bibles_mdz/     ← corpus de Bibles MDZ/BSB et classement thématique validé
├── modeles/            ← poids des modèles entraînés (.pth) — exclus de Git
├── resultats/          ← CSV, tableaux HTML de validation, visualisations, métriques
├── retours_celine/     ← annotations de validation expertes — exclu de Git
├── deploiement_hf/     ← publication du classifieur bois/cuivre sur Hugging Face
└── yolov5_repo/        ← dépendance externe (segmentation), à cloner séparément
```

**Les données sources** viennent de Gallica (BnF), de la Bayerische Staatsbibliothek
(BSB/MDZ, Munich) et de la Biblioteca Digital Ovidiana. Le tableau de référence du corpus,
documenté à la main par Céline Bohnert, est dans `retours_celine/`.

**Les livrables consultables** sans exécuter de code : les visualisations de
`resultats/Datavis/` sont des pages HTML autonomes, à ouvrir directement dans un navigateur.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate sous Windows
pip install -r requirements.txt
git clone https://github.com/ultralytics/yolov5 yolov5_repo
```

`requirements.txt` est un `pip freeze` de l'environnement du stage (GPU CUDA) — voir son
en-tête pour l'installation sans GPU. Les poids du modèle de segmentation se téléchargent
automatiquement depuis Hugging Face au premier appel de `charger_yolo()`.

## Pour aller plus loin

- [`notebooks/README.md`](notebooks/README.md) — ordre d'exécution des notebooks, fonctions
  partagées (`gallica_utils.py`), base d'illustrations commune aux axes.
- [`docs/`](docs/) — une fiche par axe : méthode, résultats, limites connues.
