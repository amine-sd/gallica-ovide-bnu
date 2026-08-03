# Explorations (dossier dissous)

Le dossier `notebooks/04_exploration/` n'existe plus — réorganisation du dépôt
("Nettoyage dir"). Ce qui restait d'utile a été redistribué :

- **`validation_bnf_etiquete.ipynb`** (validation externe du classifieur bois/cuivre
  sur un corpus étiqueté par la BnF) est revenu dans
  [`classification_bois_cuivre/`](classification_bois_cuivre.md) — voir la fiche v4
  de ce dossier pour le détail (métadonnée technique BnF peu fiable, corpus instable
  d'un run à l'autre, méthode gardée à titre exploratoire mais écartée comme
  référence).
- Les notebooks de clustering des illustrations de Bibles
  (`test_clusters_bibles.ipynb`, `atelier_regroupement.py`,
  `galerie_bibles.py`/`galerie_tri_bibles.ipynb`) sont maintenant dans
  [`corpus_bibles/`](corpus_bibles.md).

Les autres fichiers qui vivaient ici (`04_similarite_mayence1545.ipynb`,
`comparaison_clip_dinov2.ipynb`, `clustering_ovide_bibles.ipynb`,
`test_llm_iconographie_claude.ipynb`, `test_llm_iconographie_ollama.ipynb`) ont été
supprimés du dépôt, pas déplacés.
