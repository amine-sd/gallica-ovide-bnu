# Comparaison Ovide / Bibles — proximité par embeddings

Analyse à part, extraite de `mini_rag_iconographique/vector_base.ipynb` : ne fait pas
partie du RAG (rien dans `rag_generation.ipynb` / `app_rag.py` n'en dépend). Question posée :
l'iconographie d'Ovide (créations du monde/de l'homme, déluge) "retombe"-t-elle
visuellement sur les mêmes thèmes du côté des Bibles, ou sur des thèmes différents ?

## Principe

Un même espace de points : chaque illustration (Bible ou Ovide) est placée par projection
UMAP 2D de son embedding SigLIP 768D (niveaux de gris, pour neutraliser la coloration).
La couleur encode le thème (taxonomie commune aux deux corpus — les 12 classes fines des
Bibles sont regroupées sous les 3 thèmes partagés avec Ovide), la forme encode la source
(cercle = Bible, triangle = Ovide).

Deux mesures de correspondance, dans les deux sens : pour chaque illustration Ovide, sa
meilleure correspondance Bible (n'importe quel thème) — et inversement. Une vérification
par *trustworthiness* confirme que la projection 2D reflète bien la proximité réelle en
768D (dernier run : 0,93 — proche de 1,0 = parfait), pas un artefact de mise en page.

## Prérequis

- `mini_rag_iconographique/vector_base.ipynb` déjà exécuté au moins une fois — il génère
  les deux fichiers lus ici (`data/vector_bases/bibles_siglip.pkl` et
  `ovide_bnu_corpus_comparaison_siglip.pkl`). Ce second fichier n'est utile qu'à ce
  notebook — le RAG lui-même s'appuie sur la base Ovide complète
  (`ovide_corpus_complet_siglip.pkl`, 2191 illustrations), pas sur ce sous-ensemble de 73.
- `lib/d3.v7.min.js` (à côté du notebook, gitignoré) — voir "Sortie" ci-dessous.

## Notebooks

| Fichier | Rôle |
|---|---|
| `comparaison_ovide_bibles.ipynb` | Projection UMAP, correspondances Ovide↔Bible, export HTML interactif, vérification trustworthiness |

## Sortie

[`resultats/comparaison_ovide_bibles/proximite_bibles_ovide.html`](../resultats/comparaison_ovide_bibles/proximite_bibles_ovide.html)
— page autonome, à ouvrir dans un navigateur (survol = image en grand, clic sur un point =
isoler ses liens, curseur de seuil de similarité, mode comparaison jusqu'à 4 images).
D3.js est **embarqué directement dans le HTML** (depuis `lib/d3.v7.min.js`) plutôt que
chargé depuis un CDN — la page fonctionne donc sans connexion internet et n'est pas
affectée par un navigateur qui bloquerait le chargement de scripts distants sur une page
ouverte en local (`file://`), ce qui a été observé en pratique.
