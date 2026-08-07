# Corpus Bibles (BSB/MDZ)

Constitution et classification par thème iconographique du corpus de Bibles illustrées
utilisé comme second corpus (aux côtés d'Ovide) par `../mini_rag_iconographique/`.

## Pipeline

```
recuperation_bsb.ipynb → galerie_tri_bibles.ipynb → test_clusters_bibles.ipynb → atelier_regroupement.py
```

1. **Récupération** (`recuperation_bsb.ipynb`) — recherche via l'API MDZ (Münchener
   DigitalisierungsZentrum) filtrée sur `"(biblia tafeln)"`, 1551-1750, avec une table
   HTML de sélection soumise à Céline pour écarter les entrées non pertinentes.
   Téléchargement IIIF + segmentation YOLO (jusqu'à 10 illustrations par Bible retenue)
   dans `data/bibles_mdz/segmentees/{bsb_id}/`. Dernier run : **398 Bibles traitées,
   390 avec illustrations, 3570 illustrations au total**.
2. **Tri visuel** (`galerie_tri_bibles.ipynb`) — galerie HTML (toutes les illustrations,
   chemins relatifs, à ouvrir sur la machine qui a les images) où Céline coche les
   illustrations à garder. Backup complet dans `segmentees_backup/`, puis suppression
   du reste. Dernier run : **731 illustrations gardées sur 131 Bibles**.
3. **Test de granularité** (`test_clusters_bibles.ipynb`) — exploratoire seulement, rien
   n'est copié ni déplacé (travaille sur une copie, `regroupement/`). Embeddings CLIP
   (`ViT-B-32`, niveaux de gris pour neutraliser la coloration) puis K-Means, grilles de
   vignettes par cluster pour trouver la bonne granularité avant de passer à l'atelier.
4. **Classification finale** (`atelier_regroupement.py`) — atelier web (Flask,
   `http://localhost:8050`) : propose un premier regroupement automatique (mêmes
   embeddings CLIP + K-Means), puis correction manuelle planche par planche — déplacer
   une image, valider une classe (elle se fige, le re-clustering ne la touche plus),
   supprimer une image ou une classe, voir en grand avec navigation clavier. La
   suppression est aussi possible directement depuis la vue agrandie (icône ou touche
   Suppr), sans repasser par les vignettes. « Enregistrer » réécrit
   `data/bibles_mdz/classes_celine/` — c'est à la fois la sauvegarde et le point de
   reprise de la session suivante.


## État actuel

`data/bibles_mdz/classes_celine/` — **13 classes, toutes validées** : Babel, Caïn et
Abel, Chasses du paradis, Création (du monde / de l'homme / d'Ève / hors entraînement),
Déluge (avant / après), Images curieuses, Images mixtes, Sacrifice d'Abraham, Tentation.
393 illustrations classées au total.

Ce classement par thème est directement réutilisé comme métadonnée `theme` pour les
Bibles dans la base vectorielle du mini RAG (voir `../mini_rag_iconographique/`).

## Notebooks / scripts

| Fichier | Rôle |
|---|---|
| `recuperation_bsb.ipynb` | Récupération des Bibles depuis BSB/MDZ, segmentation YOLO |
| `galerie_tri_bibles.ipynb` | Tri visuel (garder/écarter), backup puis nettoyage |
| `test_clusters_bibles.ipynb` | Test exploratoire de granularité de clustering (CLIP + K-Means) |
| `atelier_regroupement.py` | Atelier web — classification finale par thème, source de vérité |
