# Visualisations

Notebooks produisant chacun une visualisation spécifique et autonome (page HTML,
consultable directement dans un navigateur) — par opposition aux pipelines de données
des autres dossiers. Pas de fil conducteur unique entre les fichiers : chacun répond à
une question ponctuelle, avec sa propre source et sa propre sortie.

## Circulation des éditions (cartes et frises chronologiques)

Visualisation géographique et chronologique de la circulation des éditions illustrées
des *Métamorphoses* d'Ovide — d'où elles viennent, quand, et comment un même graveur
(ou une copie de son travail) se déplace d'une ville à l'autre.

Source commune aux deux notebooks : `retours_celine/BNU_corpus.ods` (feuille `Synthèse`),
le tableau de référence documenté par Céline Bohnert (titre, ville, année, technique,
graveur, liens).

## Notebooks

| Fichier | Rôle |
|---|---|
| `carte_circulation.ipynb` | Carte Leaflet animée par un slider (1490-1750) — 19 villes, flèches de circulation (un graveur actif dans plusieurs villes) et flèches de copie (une édition copiant le travail d'un autre graveur). Frise et carte fusionnées : le slider fait avancer la carte dans le temps. |
| `nuage_editions_villes.ipynb` | Carte + frise chronologique **côte à côte** (pas fusionnées), limité à Venise, Paris et Lyon — un nuage de points en spirale autour de chaque ville sur une vraie carte géographique, et la même liste d'éditions positionnée par année sur une frise SVG juste à côté. Couleur = technique (bois/cuivre), pour repérer si une ville bascule plus tôt ou plus tard vers le cuivre. |

## Sorties

`resultats/Datavis/carte_circulation.html` et
`resultats/Datavis/nuage_editions_venise_paris_lyon.html` — pages HTML autonomes,
consultables directement dans un navigateur.
