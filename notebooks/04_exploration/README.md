# Explorations

Notebooks et scripts ponctuels, hors des pipelines principaux (similarité,
classification bois/cuivre, classification par graveur). Documentation plus légère
que les autres dossiers — certains fichiers n'ont pas de fiche détaillée.

## Bois / cuivre

- **`validation_bnf_etiquete.ipynb`** — validation externe du classifieur bois/cuivre
  sur un corpus étiqueté par la BnF (vérité terrain officielle). Déplacé ici depuis
  `../classification_bois_cuivre/` : la métadonnée technique BnF s'est révélée
  peu fiable à l'inspection visuelle, et le corpus retéléchargé change de
  composition à chaque exécution — méthode conservée à titre exploratoire, plus
  utilisée comme référence. Voir la fiche v4 dans
  `../classification_bois_cuivre/README.md` pour le détail.

## Similarité

- **`04_similarite_mayence1545.ipynb`** — pipeline de recherche par similarité (voir
  `../documentation_similarite_salomon/`) appliqué au corpus Wickram/Mayence 1545.
- **`comparaison_clip_dinov2.ipynb`** — comparaison entre CLIP (via l'API BnF) et
  DINOv2 (en local) pour la similarité d'image.

## Iconographie / LLM

- **`test_llm_iconographie_claude.ipynb`** / **`test_llm_iconographie_ollama.ipynb`**
  — tests d'identification de thèmes iconographiques par LLM (Claude vs Ollama en local).

## Bibles illustrées (BSB/MDZ) — clustering et regroupement

Corpus `data/bibles_mdz/` (voir `../05_annexes/recuperation_bsb.ipynb` pour la
récupération). Pas de fiche détaillée pour ces fichiers pour l'instant :

- `clustering_ovide_bibles.ipynb` — clustering des illustrations de bibles.
- `test_clusters_bibles.ipynb` — tests sur les clusters obtenus.
- `atelier_regroupement.py` — script de regroupement d'illustrations.
- `galerie_bibles.py` / `galerie_tri_bibles.ipynb` — génération/tri de galeries
  d'illustrations pour inspection visuelle.
