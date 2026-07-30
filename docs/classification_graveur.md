# Classification par graveur

**Axe en cours.** Objectif : identifier le graveur d'une illustration (au-delà de la
seule distinction bois/cuivre traitée dans `../classification_bois_cuivre/`).

## Constitution du dataset

Pour chaque graveur, une ou plusieurs **éditions** imprimées des *Métamorphoses*
qui lui sont attribuées sont retrouvées et traitées séparément, puis rassemblées
sous le même nom de graveur (dict `GRAVEURS` du notebook).

Le pipeline, par édition, est toujours le même :

1. **Retrouver l'édition numérisée** — pas de source unique : selon les cas,
   Gallica (API BnF), la Bibliothèque d'État de Bavière (BSB Munich, IIIF), un PDF
   téléchargé à la main, ou la Biblioteca Digital Ovidiana. Les éditions étant
   dispersées entre plusieurs bibliothèques numériques, le notebook définit une
   fonction de récupération par type de source plutôt qu'une seule.
2. **Récupérer les pages scannées** de l'édition (page entière — texte + illustration).
3. **Segmenter chaque page avec YOLO** pour isoler l'illustration de la page (texte,
   marges, ornements exclus) — sauf pour la Biblioteca Digital Ovidiana, dont les
   illustrations sont déjà détourées à la source.
4. **Ranger le résultat** dans `data/editions_ovide/segmentees/{nom_edition}/`, un
   sous-dossier par édition — partagé avec l'axe bois/cuivre.

Les tentatives infructueuses : (ARK introuvable, API sans réponse, édition non
numérisée).

**État actuel** : 22 graveurs, ~2100 illustrations. Restent à sourcer : Altzenbach
(exemplaire physique Bnu Strasbourg, à numériser) et Mulder (édition non trouvée à
ce jour).

## Notebooks

| Fichier | Rôle |
|---|---|
| `01_constitution_dataset.ipynb` | Constitution du dataset par graveur |


