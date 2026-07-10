# Visualisations

Notebooks produisant chacun une visualisation spécifique et autonome (page HTML,
consultable directement dans un navigateur) — par opposition aux pipelines de données
des autres dossiers. Pas de fil conducteur unique entre les fichiers : chacun répond à
une question ponctuelle, avec sa propre source et sa propre sortie.

## Circulation des éditions, des graveurs et des plaques

Cinq angles sur la même question : d'où viennent les éditions illustrées des
*Métamorphoses* d'Ovide, quand, et comment un même graveur — ou les mêmes plaques
gravées, revendues ou héritées d'un éditeur à l'autre — circulent dans le temps.

Source commune aux cinq notebooks : `retours_celine/BNU_corpus.ods` (feuille `Synthèse`),
le tableau de référence documenté par Céline Bohnert (titre, ville, année, technique,
graveur, éditeur, liens).

Dans `01` à `04`, une même édition parfois scindée en plusieurs lignes "tome" dans le
tableau source (même ville/année/éditeur/titre abrégé/graveur, un titre complet différent
par tome) est fusionnée en une seule édition avant tout traitement (`fusionner_tomes()`,
dupliquée à l'identique dans chacun des quatre notebooks) : sans cette fusion, une même
publication compterait deux ou trois fois.

## Notebooks

| Fichier | Rôle |
|---|---|
| `01_carte_circulation.ipynb` | Carte Leaflet animée par un slider (1490-1750) — 19 villes, flèches de circulation (un graveur actif dans plusieurs villes) et flèches de copie (une édition copiant le travail d'un autre graveur). Frise et carte fusionnées : le slider fait avancer la carte dans le temps. |
| `02_nuage_editions_villes.ipynb` | Trois cartes géographiques historiques empilées, une par ville (Venise, Paris, Lyon) — un nuage de points en spirale autour de chaque ville, un point par éditeur (ou par graveur, bouton bascule global) ayant publié dans cette ville, taille ∝ nombre d'éditions, couleur propre à chaque groupe. Tableau détaillé dépliable sous chaque carte. |
| `03_reemploi_plaques.ipynb` | Frise SVG en couloirs, sur l'ensemble du corpus (19 villes) — une ligne par jeu de plaques gravées, un point par édition l'utilisant. Distingue réimpression par le même éditeur (trait neutre), transmission à un autre éditeur (flèche rouge) et cas incertain (pointillé, éditeur non identifié), plus les flèches de copie entre jeux de plaques différents (même logique que `01`, adaptée aux couloirs). Une ligne réunit en général ≥2 éditions réemployées ; quelques lignes "solo" (une seule édition) sont ajoutées uniquement pour ancrer une flèche de copie. Les éditions à graveur composite (plusieurs personnes citées dans la colonne graveur) sont écartées : pas d'identité de plaques unique à tracer. Pas de carte : question de filiation entre éditeurs dans le temps, pas de géographie. |
| `04_reseau_editions.ipynb` | Graphe en réseau (D3.js, disposition par simulation de forces) sur l'ensemble du corpus — un nœud par édition (couleur = ville), un lien par relation entre deux éditions : réemploi de plaques (mêmes 3 types que `03`) et copies (même logique que `01`), combinés dans un seul graphe. Demandé pour rassembler en une vue ce que `01`-`03` traitent séparément, quitte à ce que ce soit dense. Abscisse = année (axe gradué), ordonnée libre (juste pour étaler les nœuds) : hybride entre graphe en réseau et frise chronologique. |
| `05_vignette_plaques.ipynb` |  |

## Sorties

`resultats/Datavis/carte_circulation.html`, `resultats/Datavis/nuage_editions_venise_paris_lyon.html`,
`resultats/Datavis/reemploi_plaques.html`, `resultats/Datavis/reseau_editions.html` et
`resultats/Datavis/vignette_plaques.html` — pages HTML autonomes, consultables directement dans
un navigateur.

## Technologies utilisées

**Côté notebook (préparation des données)**
- `odfpy` — lecture cellule par cellule de `BNU_corpus.ods` (le lecteur ODF de `pandas.read_excel`
  ignore silencieusement certaines colonnes du fichier, notamment `titre abrégé`)
- `pandas` — à partir du tableau brut lu par `odfpy`, sélection des seules colonnes utiles à
  la carte et nettoyage (ville canonique, année entière, titre abrégé, lien résolu) en un
  tableau `editions` (une ligne par édition), source unique du reste du notebook ; sert aussi
  à repérer les cases encore vides à raffiner dans le tableau source de Céline
- Python standard (`re`, `json`) — normalisation de texte (noms de graveurs, désambiguïsation
  des mentions de copie) et sérialisation des données pour le template HTML

**Côté carte (page HTML générée, autonome — aucun build, tout est un template Python)**
- [Leaflet](https://leafletjs.com/) 1.9.4 — carte interactive, marqueurs de ville, flèches
  courbes (polylines) et infobulles
- [Leaflet.PolylineDecorator](https://github.com/bbecquet/Leaflet.PolylineDecorator) 1.6.0 —
  pointes de flèche orientées le long des flèches courbes
- [MapLibre GL JS](https://maplibre.org/) 3.6.2 + [maplibre-gl-leaflet](https://github.com/maplibre/maplibre-gl-leaflet)
  0.0.20 — fond de carte vectoriel intégré dans Leaflet
- [OpenHistoricalMap](https://www.openhistoricalmap.org/) (style `map-styles/main`) — fond de
  carte avec frontières et toponymes d'époque plutôt qu'un simple habillage graphique
- [@openhistoricalmap/maplibre-gl-dates](https://github.com/openhistoricalmap/maplibre-gl-dates)
  1.3.0 — filtrage du fond de carte par date, synchronisé avec le slider temporel
- HTML/CSS/JS "vanilla" — pas de framework front ; toute l'interaction (slider, lecture
  ▶/⏸, réglage de vitesse, mise à jour des flèches et marqueurs) est écrite à la main dans
  le template Python de la cellule de génération

`03_reemploi_plaques.ipynb` n'utilise ni Leaflet ni MapLibre : pas de question géographique,
juste une frise en SVG pur (générée en Python, même principe que la frise fusionnée à la
carte de `01_carte_circulation.ipynb`) et le même HTML/CSS/JS "vanilla" pour l'interaction.

`04_reseau_editions.ipynb` introduit [D3.js](https://d3js.org/) v7 (`d3-force` pour la
disposition du graphe par simulation physique, `d3-drag` et `d3-zoom` pour l'interaction) :
contrairement aux trois autres notebooks, où toutes les positions sont calculées à l'avance en
Python, un graphe en réseau n'a pas de position "naturelle" — la disposition est calculée
dans le navigateur, au chargement de la page.

`05_vignette_plaques.ipynb` 
