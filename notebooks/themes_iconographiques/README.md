# Thèmes iconographiques — mini RAG (recherche + génération)

Comparaison CLIP/DINOv2/SigLIP pour la séparation des thèmes iconographiques, puis
construction d'un mini RAG (Retrieval-Augmented Generation) : requête texte ou image
→ recherche vectorielle dans le corpus Bibles + Ovide → réponse rédigée par un LLM
local (Ollama).

---

## Notebooks

| Fichier | Rôle |
|---|---|
| `vector_base.ipynb` | Comparaison CLIP/DINOv2/SigLIP (précision 1-plus-proche-voisin, test de McNemar), fine-tuning léger de SigLIP, construction des bases vectorielles initiales (`bibles_siglip.pkl` 470, `ovide_bnu_corpus_siglip.pkl` 73), visualisation UMAP de proximité |
| `vector_base_corpus_complet.ipynb` | Étend la base Ovide à **2191 illustrations / 28 éditions** (au lieu des 73 déjà thématisées) — rapproche chaque dossier segmenté d'une ligne de la feuille `Synthèse` de `retours_celine/BNU_corpus.ods`, vectorise tout avec SigLIP → `ovide_corpus_complet_siglip.pkl` |
| `themes_precis_famille7.ipynb` | Thème précis planche par planche pour les 4 éditions ayant une feuille `#N` dédiée dans `BNU_corpus.ods` (Solis, Salomon, Wickram, Savery — voir détail plus bas) — ajoute `theme_precis` / `description_planche` à la base |
| `index_metadonnees.ipynb` | Second index vectoriel — embeddings texte (`multilingual-e5-small`) des métadonnées/descriptions, pour la recherche hybride (voir plus bas). Pour les 12 thèmes bibliques, `THEME_MOTS_CLES_BIBLE` ajoute des mots-clés narratifs (noms propres, objets) en plus du seul intitulé du thème |
| `retrieval.ipynb` | Brique retrieval seule : mécanique de recherche (texte/image → SigLIP → cosinus), démos texte EN/FR, sanity check leave-one-out |
| `rag_generation.ipynb` | Brique génération : texte (`llama3.2`) et vision (`qwen2.5vl`), prompts, démos |
| `app_rag.py` | Application de démo (Gradio, façon ChatGPT) — voir "Lancer l'appli" plus bas |
| `architecture_rag.html` | Schéma explicatif du pipeline (2 phases : construction de la base / requête en direct) — à ouvrir dans un navigateur, pensé pour une présentation (ex. au tuteur de stage) |

### Ordre d'exécution

```
vector_base.ipynb → vector_base_corpus_complet.ipynb → themes_precis_famille7.ipynb
                  → index_metadonnees.ipynb
retrieval.ipynb / rag_generation.ipynb (mecanique, independants apres les 4 ci-dessus)
app_rag.py (consomme tout, lance l'interface)
```

---

## Pipeline retrieval

1. **Encodage** — `embed_texte()` / `embed_image()` (SigLIP, `google/siglip-base-patch16-224`) projettent requête texte et images dans le **même espace 768D** — SigLIP n'a pas de couche de projection séparée comme CLIP, donc les embeddings déjà stockés dans les bases restent directement comparables aux requêtes.
2. **Recherche visuelle** — cosinus contre `X_index` (2661 illustrations : 470 Bibles + 2191 Ovide).
3. **Recherche hybride** (`rechercher_hybride()`, requêtes texte uniquement) — combine trois classements par **Reciprocal Rank Fusion** :
   - similarité visuelle (SigLIP) ;
   - similarité métadonnées/descriptions (`multilingual-e5-small`) — nécessaire car SigLIP seul échoue sur les requêtes factuelles/thématiques (`"Cadmus combat un serpent"` faisait remonter un document *Caïn et Abel* avant le bon document — SigLIP n'est entraîné qu'à comparer image↔texte, jamais texte↔texte) ;
   - **correspondance exacte** (`masque_correspondance_exacte()`) — un mot ≥4 caractères de `ville`/`graveur`/`technique` qui apparaît tel quel dans la requête vaut un rang 1 dans ce 3e classement (dernier rang sinon). Corrige les confusions sémantiques sur des mots proches (ex. *"Lyon"* faisait remonter une illustration *"Hippomène en **Lion**"* avant l'ajout de ce filtre) sans agir comme un filtre dur — une requête sans aucun mot filtrable (ex. *"Cadmus combat un serpent"*) retombe simplement sur les deux premiers classements.

   RRF plutôt qu'une moyenne pondérée des scores bruts : les échelles de similarité sont incomparables entre les trois classements (SigLIP texte-image ~0.1-0.2, e5 ~0.75-0.85, correspondance exacte binaire), la RRF ne compare que des rangs.
4. **Génération** — `contexte_depuis_resultats()` transforme les résultats (thème, similarité, métadonnées) en texte, envoyé à un LLM Ollama local avec le prompt système. Le LLM ne voit **jamais** les vecteurs ni la base complète, seulement le texte du top-k (et les images du top-k pour la brique vision). Deux prompts système coexistent dans `app_rag.py` : `PROMPT_SYSTEME` (texte seul, `llama3.2`) et `PROMPT_SYSTEME_VISION` (`qwen2.5vl`) — seul ce dernier est branché sur `repondre()`, la fonction que l'interface Gradio appelle réellement. `PROMPT_SYSTEME` reste démontré dans `rag_generation.ipynb` (brique texte isolée) mais n'est pas utilisé par l'appli live.

---

## Thèmes précis (`themes_precis_famille7.ipynb`)

`BNU_corpus.ods` contient, en plus de la feuille `Synthèse` (catalogue de 112
éditions), des feuilles `#N` qui cataloguent certaines éditions **planche par
planche** (foliotation + titre thématique). Quatre éditions ont une correspondance
exacte (même ark que leur feuille `#N`) :

| Édition | Feuille | Structure | Rattachement | Couverture |
|---|---|---|---|---|
| Solis | `#7` | 1 ligne = 1 thème court | calibration folio→page (9 ancres, formule `page = index_recto_verso + 20`, validée à l'identique sur les 9) | 181/184 |
| Salomon | `#0_CORPUS_REF` | même séquence de thèmes que Solis, **pas d'ancre par ligne** | position (ordre physique du livre) — confiance moindre | 161/161 |
| Wickram | `#1` | 1 ligne = 1 planche, **plusieurs épisodes combinés**, page explicite | page directe (url par ligne) | 50 |
| Savery | `#11` | idem Wickram | page directe (url par ligne) | 15/18 |

Solis/Salomon → colonne `theme_precis` (titre court, ex. *"La création du Monde"*).
Wickram/Savery → colonne `description_planche` (paragraphe multi-épisodes, ex.
*"Cadmus tue le dragon... / Actéon en cerf... / Narcisse..."*) — à ne **pas** citer
comme un thème unique, plusieurs scènes y sont mélangées.

**Non fait** : les 24 autres éditions (~1785 illustrations) n'ont ni l'un ni
l'autre — seulement les métadonnées d'édition (`vector_base_corpus_complet.ipynb`).
Piste non explorée : héritage par famille iconographique (une édition sans feuille
dédiée mais de la même famille que Solis/Salomon pourrait réutiliser leur séquence
de thèmes, avec un recalage folio propre à vérifier).

---

## Modèles utilisés

| Rôle | Modèle | Remarque |
|---|---|---|
| Embeddings image + texte (espace partagé) | `google/siglip-base-patch16-224` | Niveaux de gris à l'encodage (neutralise le coloriage entre éditions) |
| Embeddings texte (métadonnées) | `intfloat/multilingual-e5-small` (`sentence-transformers`) | SigLIP réutilisé pour du texte↔texte donnait de mauvais résultats (validé empiriquement) — e5 est entraîné pour ça |
| Génération texte | `llama3.2` (Ollama, local) | Prompt volontairement court — empiler des consignes de prudence pousse ce modèle à sur-refuser |
| Génération vision | `qwen2.5vl` (Ollama, local) | `llama3.2-vision` incompatible avec cette version d'Ollama (`unknown model architecture: 'mllama'`) ; `llava:7b` testé et écarté — refus réflexe non lié au contenu (~15-25% des appels, ex. *"je n'ai pas accès aux images"* suivi quand même d'une description) |

**GPU** : Ollama doit tourner en mode GPU (`ollama ps` → colonne `PROCESSOR` =
`100% GPU`). Si un paquet Ollama tiers (ex. snap communautaire) tourne en CPU pur
malgré un GPU disponible, réinstaller la version officielle
(`curl -fsSL https://ollama.com/install.sh | sh`).

---

## Lancer l'appli (`app_rag.py`)

```bash
# Prerequis : Ollama demarre avec les modeles telecharges
ollama serve
ollama pull llama3.2
ollama pull qwen2.5vl

./.venv/bin/python notebooks/themes_iconographiques/app_rag.py
```

Ouvre `http://localhost:7860`. `share=True` tente un lien public temporaire
(gradio.live) — bloqué sur le réseau BNU (port 7000), utiliser en local ou
configurer une redirection de port Windows si besoin d'un accès distant.

Le menu **Options** (sous la zone de saisie) permet de choisir le modèle de
génération : **Vision** (`qwen2.5vl`, par défaut — voit réellement les images) ou
**Texte seul** (`llama3.2` — ne reçoit que les métadonnées). Utile pour comparer
concrètement les deux approches sur la même requête.

---

## Limites connues

- **Couverture des thèmes précis** : 406/2191 illustrations Ovide (4 éditions sur 28).
- **Recherche hybride — confusions sémantiques résiduelles** : le filtre de
  correspondance exacte (ville/graveur/technique) corrige le cas *"Lyon"* /
  *"Hippomène en Lion"*, mais ne couvre que ces trois colonnes — d'autres
  confusions sémantiques restent possibles sur des champs non filtrés (ex.
  thème, titre).
- **e5 confondait des thèmes bibliques proches (résolu)** : *"Adam et Arbre"*
  faisait remonter le thème *"Cain et Abel"* au lieu de *"Tentation"* — le texte
  envoyé à e5 pour les Bibles était juste *"Illustration biblique, theme X"*,
  trop pauvre pour discriminer des récits voisins de la Genèse. Corrigé en
  ajoutant des mots-clés narratifs par thème (`THEME_MOTS_CLES_BIBLE` dans
  `index_metadonnees.ipynb`). Nuance : en isolant e5 seul, "Cain et Abel"
  reste légèrement favori (0.863→0.842) — c'est la fusion RRF avec le signal
  visuel (net en faveur de "Tentation") qui corrige le résultat final. Pas de
  régression observée sur les requêtes déjà validées (Lyon, Solis, Cadmus,
  cuivre, Babel, Eve, déluge).
- **Génération vision non déterministe** : même requête, réponses parfois
  différentes d'un run à l'autre (température non nulle) — voir `rag_generation.ipynb`
  pour des exemples côte à côte.
- **Contamination inter-images (génération vision)** : quand le top-k contient
  des illustrations de sujets différents (ex. requête *"Cadmus combat un
  serpent"* retrouvant aussi des planches Glaucus/Scylla ou Hercule/Achéloüs
  sans rapport), le LLM vision a tendance à attribuer le personnage de la
  requête (ou d'une image voisine du même lot) à une image dont la métadonnée
  ne le mentionne pas — comme s'il maintenait une cohérence narrative sur tout
  le lot plutôt que de traiter chaque image indépendamment. Même symptôme
  observé sur la *source* plutôt que le personnage : sur *"l'arche de Noé et le
  déluge"*, une illustration du corpus **Ovide** (déluge de Deucalion/Pyrrha,
  sans arche) taguée `[ovide]` dans le contexte texte a quand même été décrite
  comme faisant partie du récit biblique de Noé — alors que le tag de source
  était bien present dans le texte envoyé au modèle. Testé sur 5 formulations de
  prompt système (voir ci-dessous) : une consigne explicite ("chaque image a SA
  PROPRE métadonnée") réduit le phénomène sur certaines images mais ne
  l'élimine pas sur d'autres, y compris en le répétant. Probable limite du
  prompt seul (les images sont envoyées dans un seul message Ollama) — un
  correctif complet demanderait un appel Ollama séparé par image plutôt qu'un
  lot, non implémenté (coût : 4x plus d'appels/latence).
- **Fidélité visuelle imparfaite (`qwen2.5vl`)** : vérifié en comparant directement
  les images sources au texte généré (requête *"Actéon transformé en cerf"`) —
  le modèle capte de vrais détails visuels (torches, bâtiment circulaire, chiens,
  bois de cerf naissants...) mais **complète ce qu'il voit avec des attentes
  narratives tirées de la requête/métadonnées**, plutôt que de décrire
  strictement ce qui est présent. Deux exemples concrets :
  - Une illustration Salomon (dont le `theme_precis` est attribué par *position*
    dans le livre, confiance moindre — voir plus bas) montre en réalité trois
    femmes brandissant des torches sur une figure agenouillée, **sans aucun
    cerf ni transformation visible** ; le modèle a quand même décrit "un
    personnage transformé en cerf", clairement importé de la requête plutôt
    qu'observé.
  - Sur une gravure Tempesta montrant Diane (reconnaissable à son croissant de
    lune) et ses nymphes surprenant Actéon, le modèle a inventé un "arc et une
    flèche" (Actéon tient en réalité une lance) et "un autre personnage
    masculin" (inexistant), tout en **ratant le croissant de lune** — le détail
    le plus diagnostique de toute la scène pour confirmer l'identification.

  Limite générale connue des LLM vision, pas un bug spécifique à ce pipeline —
  mais à garder en tête : les affirmations détaillées du modèle sur le contenu
  visuel (postures, objets tenus, second plan) ne doivent pas être prises pour
  argent comptant sans vérification, même quand l'identification globale du
  sujet est correcte.

## Historique — itérations du prompt système vision

Cinq formulations testées empiriquement (mêmes images récupérées, requêtes
*"editions imprimees a Lyon"* et *"Cadmus combat un serpent"*, 2-3 répétitions
par version pour limiter le bruit de la non-déterminisme) avant de retenir la
version actuelle (V5, dans `app_rag.py` et `rag_generation.ipynb`) :

| Version | Piste testée | Résultat |
|---|---|---|
| V1 (initiale) | baseline | Bon équilibre général, mais hallucine parfois un nom sur une illustration sans métadonnée (ex. *"probablement Diane"*) et sous-exploite parfois un `theme_precis` pourtant fiable |
| V2 | distinguer explicitement observation vs métadonnées | N'empêche pas l'hallucination sur les cas sans métadonnée ; pas d'amélioration nette |
| V3 | structurer la réponse en 2 parties (description / identification) | Corrige bien le cas "sans métadonnée", mais devient trop prudent même quand la métadonnée est fiable (marque des cas confirmés comme "incertains") |
| V4 | "si SA PROPRE métadonnée le nomme, utilise-le sans hésiter ; sinon ne l'invente pas" | Corrige la sous-exploitation des `theme_precis` confirmés ; révèle un nouveau problème — la contamination inter-images (voir Limites connues) |
| **V5 (retenue)** | V4 + consigne explicite "chaque image a SA PROPRE métadonnée, distincte des autres images du lot et de la requête" | Réduit la contamination sur une des deux images-pièges testées, aucun effet mesurable sur l'autre — amélioration partielle, limite documentée ci-dessus |

## Historique — bug d'exécution nbconvert (résolu)

`retrieval.ipynb` et `rag_generation.ipynb` avaient chacun 2 cellules dont la
sortie n'était pas enregistrée lors d'un `nbconvert --execute`, malgré une
exécution apparemment correcte dans le kernel. Cause réelle, identifiée en
branchant le filtre de correspondance exacte (ci-dessus) : une fusion de
cellules antérieure avait déplacé un appel de démo (`afficher_grille(...)`,
`generer_reponse_rag_vision(...)`) dans une cellule placée **avant** la
définition de ces fonctions plus loin dans le notebook — une vraie
`NameError`, masquée à l'époque par une inspection incomplète de la sortie
d'erreur. Corrigé en séparant les définitions (restées en place) des appels de
démo (déplacés en fin de notebook, après tout ce dont ils dépendent) — les
deux notebooks s'exécutent maintenant intégralement via `nbconvert --execute`
avec un `execution_count` séquentiel sur toutes les cellules.
