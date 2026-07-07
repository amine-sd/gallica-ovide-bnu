# Similarité — corpus Salomon

Recherche par similarité d'image via l'API BnF « Fouille d'image »,
appliquée aux 184 gravures de Bernard Salomon (*Métamorphose d'Ovide figurée*, Lyon
1557) — pour retrouver, dans tout Gallica, les illustrations visuellement proches et
tracer la circulation de son iconographie dans les éditions ultérieures.

## Notebooks

| Fichier | Rôle |
|---|---|
| `01_documentation_api_bnf.ipynb` | Référence — documente les 6 endpoints de l'API production `galimages-search.bnf.fr` et leur format exact |

| `02_collecte_similarite.ipynb` | Collecte — pour chacune des 184 illustrations Salomon avec embedding CLIP valide, recalcule l'embedding depuis la vignette segmentée par YOLO (plus précis que l'embedding de page entière), interroge l'API pour les `N_RESULTATS` résultats les plus similaires → `resultats/csv/salomon_segmente.csv` |

| `03_analyse_resultats_bruts.ipynb` | Statistiques (scores, technique, genre, mode chromatique), figures et tableau HTML filtrable |

## Ordre d'exécution

```
02_collecte_similarite → 03_analyse_resultats_bruts
```

`01_documentation_api_bnf.ipynb` est indépendant, à consulter plutôt qu'à exécuter.
