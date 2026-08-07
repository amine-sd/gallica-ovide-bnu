# Classification par graveur

Objectif : identifier le graveur d'une illustration (au-delà de la
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
| `02_reconnaissance_main.ipynb` | Test exploratoire — embeddings gelés (DINOv2 vs SigLIP), avant tout fine-tuning |

## Reconnaissance de la main — premier test (embeddings gelés)

Avant d'investir dans un entraînement complet, test de ce que des embeddings d'image
pré-entraînés (sans fine-tuning) donnent déjà pour séparer les graveurs par leur style.
Limité aux 9 graveurs ayant au moins 100 illustrations (les 13 autres n'ont pas assez
d'exemples pour un split train/test qui veuille dire quelque chose) :
Tempesta, Baur, De Passe, Solis, Borcht, Salomon, Monconet, Mathieu, Bouche.

**Point méthodologique clé** : 3 de ces 9 graveurs (Tempesta, Baur, De Passe) ont deux
éditions segmentées chacun — seul cas permettant un vrai test de généralisation à une
édition jamais vue. Les 6 autres n'ont qu'une seule édition : un split image par image
ne peut pas exclure qu'une partie du score reflète des artefacts de scan/papier propres
à l'édition plutôt que la main du graveur.

**Résultat** : sur le test par édition jamais vue (410 illustrations), DINOv2 atteint
90,7 % de précision au plus-proche-voisin et 94,1 % avec une sonde linéaire — très loin
du hasard (11,1 % sur 9 classes). L'écart avec SigLIP (89,5 % / 81,2 %) n'est
significatif qu'avec la sonde linéaire (McNemar non significatif en plus-proche-voisin,
p=0,36). Signal assez fort pour justifier un fine-tuning (tête MLP sur DINOv2 gelé,
même méthodologie multi-graines que `classification_bois_cuivre`).

**Une confusion expliquée, pas une erreur du modèle** : 20 % des illustrations de Tempesta
tenues à l'écart sont confondues avec Moncornet (et 3 % avec Mathieu) — jamais dispersées
vers les autres graveurs. Le tableau de référence de Céline (`retours_celine/BNU_corpus.ods`,
feuille Synthèse) note explicitement Moncornet et Mathieu comme « copie Antonio Tempesta » ;
à l'œil, la première planche de Moncornet reprend la composition de Tempesta quasi trait
pour trait, inversée en miroir — même constat que celui déjà documenté côté
`../visualisations/` sur la circulation des plaques et les copies entre graveurs. DINOv2 ne
se trompe pas : il détecte une vraie parenté iconographique, y compris quand elle vient
d'une copie assumée plutôt que d'une main partagée — une validation du signal plutôt qu'une
faiblesse.

