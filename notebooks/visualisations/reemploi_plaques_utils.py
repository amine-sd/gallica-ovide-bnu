"""
reemploi_plaques_utils.py — logique partagée entre `03_reemploi_plaques.ipynb`
(export statique) et `reemploi_plaques_editeur.py`.

Convention du projet (voir gallica_utils.py, rag_utils.py) : les fonctions de
chargement retournent les objets chargés, les autres les prennent en paramètre
explicite plutôt que de dépendre de variables globales.

Trois briques :
  - lecture/normalisation du corpus (`charger_editions`)
  - regroupement en jeux de plaques + résolution des flèches de copie
    (`construire_plaques`, `resoudre_fleches_copies`), toutes deux acceptant
    des corrections manuelles en overlay (voir `corrections.py` conceptuel :
    `charger_corrections`/`sauver_corrections` ci-dessous)
  - mise en page + génération HTML (`construire_svg`, `generer_html`)
"""

import json
import math
import os
import re
from collections import defaultdict

from odf.opendocument import load as charger_ods
from odf.table import Table, TableRow, TableCell
from odf.text import P
from odf import teletype

COL_GRAVEUR = "graveur\xa0: Nom, Prénom"
LIBELLES_TECHNIQUE = {"bois": "Bois", "cuivre": "Cuivre", "inconnue": "Technique inconnue"}

FRISE_ANNEE_MIN, FRISE_ANNEE_MAX = 1490, 1750
FRISE_TICKS = [1500, 1600, 1700]
LARGEUR = 2600
MARGE = {"gauche": 210, "droite": 20, "haut": 30, "bas": 36}
HAUTEUR_LIGNE = 24
RAYON_POINT = 6
PAS_EMPILEMENT = RAYON_POINT * 2 + 16  
ECART_PLANCHER = 45
ECART_DEMIVIE = 50


# ─────────────────────────────────────────────
# Lecture et normalisation du corpus
# ─────────────────────────────────────────────

def lire_feuille_ods(chemin, nom_feuille):
    """Lit une feuille ODS cellule par cellule (pandas ignore des colonnes de ce fichier)."""
    doc = charger_ods(chemin)
    table = next(t for t in doc.spreadsheet.getElementsByType(Table)
                 if t.getAttribute("name") == nom_feuille)
    lignes_brutes = table.getElementsByType(TableRow)

    def valeurs_ligne(ligne):
        valeurs, col = {}, 0
        for cellule in ligne.getElementsByType(TableCell):
            rep = cellule.getAttribute("numbercolumnsrepeated")
            rep = int(rep) if rep else 1
            paras = cellule.getElementsByType(P)
            texte = " ".join(teletype.extractText(p) for p in paras)
            for k in range(rep):
                valeurs[col + k] = texte
            col += rep
        return valeurs

    entetes = valeurs_ligne(lignes_brutes[0])
    colonnes = {i: t.strip() for i, t in entetes.items() if t.strip()}

    lignes = []
    for ligne in lignes_brutes[1:]:
        rep = ligne.getAttribute("numberrowsrepeated")
        rep = int(rep) if rep else 1
        valeurs = valeurs_ligne(ligne)
        if not any(v.strip() for v in valeurs.values()):
            continue  # ligne vide (fin de feuille)
        d = {nom: valeurs.get(i, "").strip() for i, nom in colonnes.items()}
        lignes.extend([d] * rep)
    return lignes


def extraire_annee(valeur):
    """Renvoie la première année à 4 chiffres trouvée (ex: "1527 / 1528 ?" -> 1527)."""
    m = re.search(r"\d{4}", str(valeur))
    return int(m.group()) if m else None


def extraire_lien(row):
    """Choisit le premier lien exploitable, par ordre de préférence (même logique que
    01_carte_circulation.ipynb et 02_nuage_editions_villes.ipynb)."""
    for col in ["version numérisée 1", "version numérisée 2", "Biblioteca Digital Ovidiana", "url catalogue"]:
        val = row.get(col, "")
        if not val:
            continue
        premier = val.split(";")[0].strip()
        if premier.startswith("http"):
            return premier
    return None


def graveur_ou_inconnu(g):
    """None pour un graveur non identifié ('?', 'inaccessible', case vide) : ces éditions sont
    écartées, pas d'identité de plaques, pas de ligne dans la frise."""
    g = (g or "").strip()
    if not g or g.lower() in {"?", "inaccessible"}:
        return None
    return g


def graveur_multiple(g):
    """True si le champ graveur nomme plusieurs personnes (crédit composite), séparées par
    "/" ou " et ". Une virgule seule ne suffit pas (aussi séparateur nom/prénom)."""
    return "/" in g or re.search(r"\bet\b", g, re.IGNORECASE) is not None


def categorie_technique(t):
    t = (t or "").strip().lower()
    if t == "bois":
        return "bois"
    if t == "cuivre":
        return "cuivre"
    return "inconnue"


def fusionner_tomes(corpus):
    """Une même édition parfois scindée en plusieurs tomes (même ville/année/éditeur/titre
    abrégé/graveur) : fusionnée en une seule, sinon elle compterait plusieurs fois."""
    groupes, ordre = {}, []
    for ligne in corpus:
        cle = (ligne.get("ville", ""), ligne.get("année", ""), ligne.get("publisher", ""),
               ligne.get("titre abrégé", ""), ligne.get(COL_GRAVEUR, ""))
        if cle not in groupes:
            groupes[cle] = []
            ordre.append(cle)
        groupes[cle].append(ligne)

    champs_premier_non_vide = ["url catalogue", "version numérisée 1", "version numérisée 2",
                                "Biblioteca Digital Ovidiana", "copies de cette édition"]
    fusionne = []
    for cle in ordre:
        lignes_tomes = groupes[cle]
        base = dict(lignes_tomes[0])
        if len(lignes_tomes) > 1:
            for champ in champs_premier_non_vide:
                for ligne in lignes_tomes:
                    if ligne.get(champ, "").strip():
                        base[champ] = ligne[champ]
                        break
        fusionne.append(base)
    return fusionne


def charger_editions(chemin_corpus, verbeux=True):
    """Lit BNU_corpus.ods (feuille Synthèse) et renvoie la liste `editions` (dicts) prête à
    l'emploi pour construire_plaques()."""
    corpus = lire_feuille_ods(chemin_corpus, "Synthèse")
    nb_avant_fusion = len(corpus)
    corpus = fusionner_tomes(corpus)
    if verbeux and len(corpus) != nb_avant_fusion:
        print(nb_avant_fusion - len(corpus), "lignes fusionnées (tomes d'une même édition regroupés)")

    editions = []
    nb_graveurs_multiples = 0
    for row in corpus:
        annee = extraire_annee(row.get("année", ""))
        if annee is None:
            continue
        graveur = graveur_ou_inconnu(row.get(COL_GRAVEUR, ""))
        if graveur is None:
            continue
        if graveur_multiple(graveur):
            nb_graveurs_multiples += 1
            continue
        titre = row.get("titre abrégé") or row.get("titre complet") or ""
        editions.append({
            "ville": row.get("ville", "").strip() or "Ville inconnue",
            "annee": annee,
            "titre": titre.strip(),
            "technique": categorie_technique(row.get("technique", "")),
            "graveur": graveur,
            "publisher": row.get("publisher", "").strip() or "Éditeur non identifié",
            "lien": extraire_lien(row),
            "copies": row.get("copies de cette édition", "").strip(),
        })

    if verbeux:
        print(len(editions), "éditions avec un graveur identifiable, sur", len(corpus), "au total")
        if nb_graveurs_multiples:
            print(nb_graveurs_multiples, "éditions écartées (graveur composite, plusieurs personnes citées)")
        print(len({e["graveur"] for e in editions}), "jeux de plaques distincts")
    return editions


# ─────────────────────────────────────────────
# Regroupement en jeux de plaques ("lignes"/couloirs)
# ─────────────────────────────────────────────

def editeur_fiable(pub):
    """Un éditeur non identifié ('s.n.', case vide) ne permet pas de dire avec certitude si
    deux éditions se succèdent chez le même éditeur ou changent de main."""
    return pub not in {"s.n.", "Éditeur non identifié"}


def type_segment(pub1, pub2):
    if not (editeur_fiable(pub1) and editeur_fiable(pub2)):
        return "incertain"
    return "reprise" if pub1 == pub2 else "transfert"


def normaliser_candidat_anonyme(c):
    c = re.sub(r"anonyme\s*(\d{4})", r"Anonyme\1", c, flags=re.IGNORECASE)
    if re.fullmatch(r"\d{4}", c.strip()):
        c = "Anonyme" + c.strip()
    return c.strip()


def eclater_enumeration(fragment):
    parties = [p.strip() for p in fragment.split(",")]
    if len(parties) > 1 and all(re.fullmatch(r"(anonyme\s*)?\d{4}", p, re.IGNORECASE) for p in parties):
        return parties
    return [fragment]


def mention_ambigue(texte):
    """"ou" ou "?" signale une attribution hésitante entre plusieurs graveurs : on ne devine pas."""
    return bool(re.search(r"\bou\b", texte, re.IGNORECASE)) or "?" in texte


def extraire_candidats_copie(texte):
    t = re.sub(r"\(.*?\)", "", texte)
    t = re.sub(r"^\s*(copie|même famille que)\s*", "", t, flags=re.IGNORECASE)
    bruts = re.split(r"\s+et\s+", t)
    candidats = []
    for c in bruts:
        c = c.strip(" .,;?\xa0")
        if not c:
            continue
        for sous in eclater_enumeration(c):
            sous = normaliser_candidat_anonyme(sous.strip(" .,;?\xa0"))
            if sous:
                candidats.append(sous)
    return candidats


def tokens_nom(nom):
    nom = re.sub(r"\(.*?\)", "", nom)
    nom = nom.replace(",", " ")
    return set(t.lower() for t in re.findall(r"[a-zà-öø-ÿ']+", nom) if len(t) >= 3)


def trouver_graveur_connu(candidat, registre):
    if candidat in registre:
        return candidat
    tc = tokens_nom(candidat)
    if not tc:
        return None
    meilleur, meilleur_score = None, 0
    for nom_reg in registre:
        if nom_reg.lower().startswith("anonyme"):
            continue
        score = len(tc & tokens_nom(nom_reg))
        if score > meilleur_score:
            meilleur, meilleur_score = nom_reg, score
    return meilleur if meilleur_score >= 1 else None


def _determiner_solo_necessaires(editions, groupes_plaques, registre_graveurs):
    """Jeux de plaques à édition unique qui ont quand même besoin d'une ligne, pour porter
    une flèche de copie résoluble (comme source ou comme cible)."""
    solo = {}
    for e in editions:
        texte = e.get("copies", "")
        if not texte or mention_ambigue(texte):
            continue
        for c in extraire_candidats_copie(texte):
            match = trouver_graveur_connu(c, registre_graveurs)
            if match is None:
                continue
            instances_cible = groupes_plaques.get(match, [])
            anterieures = [inst for inst in instances_cible if inst["annee"] <= e["annee"] and inst is not e]
            if not anterieures:
                continue
            if len(groupes_plaques.get(e["graveur"], [])) == 1:
                solo[e["graveur"]] = groupes_plaques[e["graveur"]]
            if len(instances_cible) == 1:
                solo[match] = instances_cible
    return solo


def ordonner_graveurs(items, ordre):
    """items : liste [(graveur, editions_triees), ...]. `ordre` :
      - None / "auto" : le plus réemployé d'abord (défaut historique)
      - "alphabetique" : nom du graveur
      - "chronologique" : année de la 1ère édition affichée
      - liste de noms : ordre explicite (custom, ex. glissé-déposé) — les graveurs absents de
        la liste sont ajoutés à la fin, dans l'ordre "auto"."""
    if ordre is None or ordre == "auto":
        return sorted(items, key=lambda kv: (-len(kv[1]), kv[1][0]["annee"]))
    if ordre == "alphabetique":
        return sorted(items, key=lambda kv: kv[0].lower())
    if ordre == "chronologique":
        return sorted(items, key=lambda kv: kv[1][0]["annee"])
    if isinstance(ordre, (list, tuple)):
        position = {nom: i for i, nom in enumerate(ordre)}
        auto = {kv[0]: i for i, kv in enumerate(
            sorted(items, key=lambda kv: (-len(kv[1]), kv[1][0]["annee"])))}
        return sorted(items, key=lambda kv: (0, position[kv[0]]) if kv[0] in position
                      else (1, auto[kv[0]]))
    raise ValueError(f"ordre inconnu : {ordre!r}")


def construire_plaques(editions, ordre=None, verbeux=True):
    """Regroupe les éditions par graveur (identité du jeu de plaques), ajoute les lignes
    "solo" nécessaires à l'ancrage des copies, puis ordonne les lignes selon `ordre` (voir
    ordonner_graveurs). Renvoie (groupes_plaques, plaques, registre_graveurs)."""
    groupes_plaques = defaultdict(list)
    for e in editions:
        groupes_plaques[e["graveur"]].append(e)
    groupes_plaques = dict(groupes_plaques)
    registre_graveurs = sorted(groupes_plaques)

    lanes = {g: sorted(eds, key=lambda e: e["annee"]) for g, eds in groupes_plaques.items() if len(eds) > 1}
    solo = _determiner_solo_necessaires(editions, groupes_plaques, registre_graveurs)
    for g, eds in solo.items():
        lanes[g] = eds

    items = ordonner_graveurs(list(lanes.items()), ordre)

    plaques = []
    for rang, (graveur, eds) in enumerate(items):
        segments = [
            {"an1": a["annee"], "an2": b["annee"], "type": type_segment(a["publisher"], b["publisher"])}
            for a, b in zip(eds, eds[1:])
        ]
        plaques.append({"graveur": graveur, "rang": rang, "editions": eds, "segments": segments})

    if verbeux:
        nb_transferts = sum(1 for p in plaques for s in p["segments"] if s["type"] == "transfert")
        nb_reprises = sum(1 for p in plaques for s in p["segments"] if s["type"] == "reprise")
        nb_incertains = sum(1 for p in plaques for s in p["segments"] if s["type"] == "incertain")
        nb_reemployees = sum(1 for p in plaques if len(p["editions"]) > 1)
        print(len(groupes_plaques), "jeux de plaques au total —", nb_reemployees,
              "réemployés (≥2 éditions), affichés dans la frise")
        print(nb_transferts, "transmissions à un autre éditeur,", nb_reprises,
              "réimpressions par le même éditeur,", nb_incertains, "cas incertains")
    return groupes_plaques, plaques, registre_graveurs


# ─────────────────────────────────────────────
# Flèches de copie (résolution automatique + corrections manuelles)
# ─────────────────────────────────────────────

def _point_pour(graveur, annee, points_par_graveur):
    for an, i in points_par_graveur.get(graveur, []):
        if an == annee:
            return i
    return None


def resoudre_fleches_copies(editions, points, points_par_graveur, registre_graveurs,
                             corrections_liens=None, verbeux=True):
    """corrections_liens : liste de dicts {"action": "override"|"ajout"|"suppression",
    "type": "copie"|"reprise"|"transfert" (optionnel, "copie" par défaut — rétrocompatible
    avec les corrections enregistrées avant l'ajout de ce champ), "graveur_source",
    "annee_source", "graveur_cible", "annee_cible"}.
    - "override" : remplace la résolution automatique de cette édition source par la cible donnée.
    - "suppression" : retire le lien (auto ou override) de cette édition source, sans le remplacer.
    - "ajout" : lien supplémentaire, indépendant de toute résolution automatique.
    Une correction ("override"/"suppression") désactive la résolution automatique pour cette
    édition source précise ; les autres éditions continuent d'être résolues normalement.
    Seul le type "copie" est résolu automatiquement (colonne texte libre) ; "reprise" et
    "transfert" n'existent que comme corrections manuelles — l'utilisateur s'en sert pour
    signaler que deux éditions de graveurs différents partagent en réalité le même jeu de
    plaques (visuellement au même titre que les segments auto-résolus au sein d'une ligne,
    mais entre deux lignes, donc affiché comme une flèche courbe plutôt qu'un segment droit)."""
    corrections_liens = corrections_liens or []
    par_source = {}
    ajouts = []
    for c in corrections_liens:
        if c["action"] == "ajout":
            ajouts.append(c)
        else:
            par_source[(c["graveur_source"], c["annee_source"])] = c

    fleches_copies = []
    non_resolus_copies = []

    for e in editions:
        cle_source = (e["graveur"], e["annee"])
        correction = par_source.get(cle_source)
        if correction is not None:
            if correction["action"] == "suppression":
                continue
            i_source = _point_pour(e["graveur"], e["annee"], points_par_graveur)
            i_cible = _point_pour(correction["graveur_cible"], correction["annee_cible"], points_par_graveur)
            if i_source is not None and i_cible is not None:
                fleches_copies.append({
                    "i_source": i_source, "i_cible": i_cible,
                    "graveur_source": e["graveur"], "graveur_cible": correction["graveur_cible"],
                    "an_source": e["annee"], "an_cible": correction["annee_cible"],
                    "manuel": True, "type": correction.get("type", "copie"),
                })
            continue  # corrigée : pas de résolution automatique en plus pour cette source

        texte = e.get("copies", "")
        if not texte:
            continue
        i_source = _point_pour(e["graveur"], e["annee"], points_par_graveur)
        if i_source is None:
            non_resolus_copies.append((e, texte, "(source)", "édition source non affichée (jeu de plaques jamais réemployé)"))
            continue
        if mention_ambigue(texte):
            non_resolus_copies.append((e, texte, "(toute la mention)", "attribution ambiguë (ou/possibilité multiple)"))
            continue
        matches_uniques = {}
        for c in extraire_candidats_copie(texte):
            match = trouver_graveur_connu(c, registre_graveurs)
            if match is None:
                non_resolus_copies.append((e, texte, c, "aucune correspondance"))
            else:
                matches_uniques.setdefault(match, []).append(c)
        for match in matches_uniques:
            cibles = points_par_graveur.get(match)
            if not cibles:
                non_resolus_copies.append((e, texte, match, "graveur cible non affiché (jamais réemployé)"))
                continue
            anterieures = [(an, i) for an, i in cibles if an <= e["annee"] and i != i_source]
            if not anterieures:
                non_resolus_copies.append((e, texte, match, "pas d'édition antérieure affichée pour ce graveur"))
                continue
            an_cible, i_cible = max(anterieures)
            fleches_copies.append({
                "i_source": i_source, "i_cible": i_cible,
                "graveur_source": e["graveur"], "graveur_cible": match,
                "an_source": e["annee"], "an_cible": an_cible,
                "manuel": False, "type": "copie",
            })

    for c in ajouts:
        i_source = _point_pour(c["graveur_source"], c["annee_source"], points_par_graveur)
        i_cible = _point_pour(c["graveur_cible"], c["annee_cible"], points_par_graveur)
        if i_source is not None and i_cible is not None:
            fleches_copies.append({
                "i_source": i_source, "i_cible": i_cible,
                "graveur_source": c["graveur_source"], "graveur_cible": c["graveur_cible"],
                "an_source": c["annee_source"], "an_cible": c["annee_cible"],
                "manuel": True, "type": c.get("type", "copie"),
            })

    _calculer_courbures(fleches_copies)

    if verbeux:
        nb_manuelles = sum(1 for f in fleches_copies if f["manuel"])
        print(len(fleches_copies), "flèches de copie affichées", f"(dont {nb_manuelles} manuelles)" if nb_manuelles else "")
        print(len(non_resolus_copies), "mentions non résolues ou non affichables")
    return fleches_copies, non_resolus_copies


def _courbure_pour_rang(rang):
    amplitude = 0.16 + 0.1 * (rang // 2)
    signe = 1 if rang % 2 == 0 else -1
    return amplitude * signe


def _calculer_courbures(fleches_copies):
    """Plusieurs flèches peuvent partager un point (origine ou arrivée) : courbure différente
    (signe alterné, amplitude croissante) pour éviter qu'elles se chevauchent."""
    groupes_par_point = {}
    for idx, f in enumerate(fleches_copies):
        groupes_par_point.setdefault(f["i_cible"], []).append(idx)
        groupes_par_point.setdefault(f["i_source"], []).append(idx)
    for idx, f in enumerate(fleches_copies):
        rang_max = max(
            groupes_par_point[f["i_cible"]].index(idx),
            groupes_par_point[f["i_source"]].index(idx),
        )
        f["bulge"] = _courbure_pour_rang(rang_max)


# ─────────────────────────────────────────────
# Mise en page (positions x/y) + génération du SVG/HTML
# ─────────────────────────────────────────────

def frise_x(annee):
    t = (annee - FRISE_ANNEE_MIN) / (FRISE_ANNEE_MAX - FRISE_ANNEE_MIN)
    return MARGE["gauche"] + t * (LARGEUR - MARGE["gauche"] - MARGE["droite"])


def y_ligne(rang, hauteur_ligne):
    return MARGE["haut"] + rang * hauteur_ligne + hauteur_ligne / 2


def positions_xy_plaque(eds, y_centre):
    """Position (x, y) par édition d'un même jeu de plaques (eds trié par année).
    Éditions de la MÊME année : empilées verticalement autour de y_centre plutôt que
    décalées horizontalement (qui faussait leur position temporelle réelle) — la case
    "Anonyme1693" (3 éditions en 1693) est le cas qui a motivé ce choix.
    Éditions d'années différentes décalées horizontalement vers la droite uniquement (pour
    ne pas fausser l'ordre chronologique), d'un bonus qui garantit un espace minimal (pour
    que les flèches restent visibles entre deux points d'années successives) sans jamais
    aplatir deux écarts différents à la même distance — voir ECART_PLANCHER/ECART_DEMIVIE."""
    par_annee = defaultdict(list)
    for i, e in enumerate(eds):
        par_annee[e["annee"]].append(i)

    # Une seule position x calculée par ANNÉE (pas par édition) : le bonus d'espacement se
    # cascade d'année en année, jamais entre deux éditions de la même année, qui doivent
    # rester à la même abscisse (empilées verticalement seulement). Calculer le bonus par
    # édition plutôt que par année ferait dépendre le résultat de l'ordre interne du groupe
    # (une édition déjà décalée par le bonus n'a plus la même abscisse "naturelle" que ses
    # consœurs de la même année, qui déclencherait alors un bonus à tort pour elles aussi).
    #
    # L'écart utilisé pour calculer le bonus est TOUJOURS l'écart naturel entre les deux
    # années elles-mêmes (frise_x(annee) - frise_x(annee_précédente)), jamais un écart par
    # rapport à la position déjà décalée du point précédent : un écart par rapport à la
    # position décalée peut devenir négatif après plusieurs décalages en cascade (années
    # rapprochées les unes des autres), ce qui ferait diverger la formule du bonus (division
    # par un dénominateur proche de zéro, voire négatif) et pourrait faire passer un point
    # plus récent AVANT un point plus ancien sur la frise. En repartant toujours de l'écart
    # naturel (toujours positif, années triées), le décalage cumulé reste strictement croissant.
    xs, ys = [None] * len(eds), [None] * len(eds)
    x_precedent, annee_precedente = None, None
    for annee in sorted(par_annee):
        idxs = par_annee[annee]
        if x_precedent is None:
            x_final = frise_x(annee)
        else:
            ecart_naturel = frise_x(annee) - frise_x(annee_precedente)
            bonus = ECART_PLANCHER * ECART_DEMIVIE / (ecart_naturel + ECART_DEMIVIE)
            x_final = x_precedent + ecart_naturel + bonus
        n = len(idxs)
        for k, i in enumerate(idxs):
            xs[i] = x_final
            ys[i] = y_centre + (k - (n - 1) / 2) * PAS_EMPILEMENT
        x_precedent, annee_precedente = x_final, annee
    return xs, ys


def retrait_vers(x1, y1, x2, y2, retrait):
    """Recule (x2, y2) de `retrait` unités le long du segment (x1,y1)->(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return x2, y2
    t = max(dist - retrait, 0) / dist
    return x1 + dx * t, y1 + dy * t


def courbe_copie(x1, y1, x2, y2, bulge):
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy) or 1
    nx, ny = -dy / dist, dx / dist
    cx = (x1 + x2) / 2 + nx * dist * bulge
    cy = (y1 + y2) / 2 + ny * dist * bulge
    return f"M {x1:.1f} {y1:.1f} Q {cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}"


def hauteur_ligne_effective(plaques):
    """Hauteur de ligne uniforme, agrandie si au moins une ligne empile des éditions de
    même année (évite une mise en page à hauteurs de ligne variables, plus simple à
    calculer et à faire glisser-déposer plus tard)."""
    max_empilement = 1
    for p in plaques:
        par_annee = defaultdict(int)
        for e in p["editions"]:
            par_annee[e["annee"]] += 1
        if par_annee:
            max_empilement = max(max_empilement, max(par_annee.values()))
    return HAUTEUR_LIGNE + (max_empilement - 1) * PAS_EMPILEMENT * 2


def construire_donnees_svg(plaques, segments_supprimes=None):
    """Calcule points/segments/positions. Renvoie (elements_svg, points, hauteur_ligne, hauteur_totale)
    — sans les flèches de copie (calculées à part, une fois `points` connu).
    `segments_supprimes` : ensemble de (graveur, annee_a, annee_b) à ne pas dessiner — un
    segment automatique (réimpression/transmission/incertain intra-ligne) que l'utilisateur a
    supprimé car il juge que ces deux éditions ne partagent en réalité pas le même jeu de
    plaques. Les deux points restent affichés (seul le trait qui les relie disparaît)."""
    segments_supprimes = segments_supprimes or set()
    hl = hauteur_ligne_effective(plaques)
    hauteur_plot = len(plaques) * hl
    hauteur = MARGE["haut"] + hauteur_plot + MARGE["bas"]

    elements_svg = []
    for p in plaques:
        y0 = MARGE["haut"] + p["rang"] * hl
        if p["rang"] % 2 == 1:
            elements_svg.append(f'<rect x="0" y="{y0}" width="{LARGEUR}" height="{hl}" class="bande-zebra"/>')
        elements_svg.append(
            f'<text x="8" y="{y_ligne(p["rang"], hl) + 4:.1f}" class="etiquette-plaque" data-graveur="{p["graveur"]}">{p["graveur"]}</text>'
        )

    for t in FRISE_TICKS:
        x = frise_x(t)
        elements_svg.append(f'<line x1="{x:.1f}" y1="{MARGE["haut"]}" x2="{x:.1f}" y2="{MARGE["haut"] + hauteur_plot:.1f}" class="grille-frise"/>')
        elements_svg.append(f'<text x="{x:.1f}" y="{MARGE["haut"] + hauteur_plot + 18:.1f}" text-anchor="middle" class="etiquette-annee-frise">{t}</text>')

    points = []
    for p in plaques:
        y = y_ligne(p["rang"], hl)
        xs, ys = positions_xy_plaque(p["editions"], y)
        for k, s in enumerate(p["segments"]):
            annee_a, annee_b = p["editions"][k]["annee"], p["editions"][k + 1]["annee"]
            if (p["graveur"], annee_a, annee_b) in segments_supprimes:
                continue
            x1, y1v = xs[k], ys[k]
            x2, y2v = xs[k + 1], ys[k + 1]
            marqueur = ' marker-end="url(#fleche-transfert)"' if s["type"] == "transfert" else ""
            x2t, y2t = retrait_vers(x1, y1v, x2, y2v, RAYON_POINT + 4) if s["type"] == "transfert" else (x2, y2v)
            elements_svg.append(
                f'<line x1="{x1:.1f}" y1="{y1v:.1f}" x2="{x2t:.1f}" y2="{y2t:.1f}" class="segment-{s["type"]}"{marqueur} '
                f'data-graveur="{p["graveur"]}" data-annee-a="{annee_a}" data-annee-b="{annee_b}" data-type="{s["type"]}"/>'
            )
        for j, e in enumerate(p["editions"]):
            points.append({
                "fx": round(xs[j], 1), "fy": round(ys[j], 1),
                "graveur": p["graveur"], "annee": e["annee"], "titre": e["titre"],
                "ville": e["ville"], "editeur": e["publisher"], "technique": e["technique"],
                "lien": e["lien"], "rang": p["rang"],
            })
    return elements_svg, points, hl, hauteur


LIBELLE_RELATION_LIEN = {
    "copie": "copie",
    "reprise": "réutilise les mêmes plaques (réimpression)",
    "transfert": "réutilise les mêmes plaques (transmises à un autre éditeur)",
}


def ajouter_fleches_copies_svg(elements_svg, points, fleches_copies):
    for f in fleches_copies:
        x1, y1 = points[f["i_cible"]]["fx"], points[f["i_cible"]]["fy"]
        x2, y2 = points[f["i_source"]]["fx"], points[f["i_source"]]["fy"]
        x2r, y2r = retrait_vers(x1, y1, x2, y2, RAYON_POINT + 4)
        type_lien = f.get("type", "copie")
        titre_infobulle = (f'{f["graveur_source"]} ({f["an_source"]}) {LIBELLE_RELATION_LIEN[type_lien]} '
                            f'{f["graveur_cible"]} ({f["an_cible"]})')
        classes = ["fleche-copie"]
        if type_lien != "copie":
            classes.append(f"fleche-copie-{type_lien}")
        if f.get("manuel"):
            classes.append("fleche-copie-manuelle")
        # Seuls "copie" et "transfert" ont une pointe de flèche (comme les segments
        # transfert au sein d'une ligne) ; "reprise" reste un simple trait, comme son
        # équivalent automatique intra-ligne.
        marqueur = ""
        if type_lien == "copie":
            marqueur = ' marker-end="url(#fleche-copie-tete)"'
        elif type_lien == "transfert":
            marqueur = ' marker-end="url(#fleche-manuel-transfert-tete)"'
        elements_svg.append(
            f'<path d="{courbe_copie(x1, y1, x2r, y2r, f["bulge"])}" class="{" ".join(classes)}"{marqueur} '
            f'data-i-source="{f["i_source"]}" data-i-cible="{f["i_cible"]}" data-type="{type_lien}">'
            f'<title>{titre_infobulle}</title></path>'
        )


def ajouter_points_svg(elements_svg, points):
    for i, pt in enumerate(points):
        elements_svg.append(f'<circle class="point-frise" data-i="{i}" cx="{pt["fx"]}" cy="{pt["fy"]}" r="{RAYON_POINT}"/>')


def construire_lignes_tableau(plaques):
    lignes = []
    for p in plaques:
        for j, e in enumerate(p["editions"]):
            lien_html = f'<a href="{e["lien"]}" target="_blank">voir</a>' if e["lien"] else ""
            graveur_cell = p["graveur"] if j == 0 else ""
            lignes.append(
                f'<tr><td>{graveur_cell}</td><td>{e["annee"]}</td><td>{e["ville"]}</td>'
                f'<td>{e["publisher"]}</td><td>{e["titre"]}</td>'
                f'<td>{LIBELLES_TECHNIQUE[e["technique"]]}</td><td>{lien_html}</td></tr>'
            )
    return "\n".join(lignes)


# ─────────────────────────────────────────────
# Corrections (overlay JSON — ordre des lignes + liens de copie)
# ─────────────────────────────────────────────

def charger_corrections(chemin):
    if not os.path.exists(chemin):
        return {"ordre": None, "liens": [], "segments_supprimes": []}
    with open(chemin, encoding="utf-8") as f:
        corrections = json.load(f)
    corrections.setdefault("segments_supprimes", [])
    return corrections


def sauver_corrections(chemin, corrections):
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(corrections, f, ensure_ascii=False, indent=1)


def construire_tout(chemin_corpus, chemin_corrections=None, verbeux=True):
    """Pipeline complet : charge editions + corrections, construit plaques et flèches de
    copie. Renvoie un dict avec tout ce qu'il faut pour générer le HTML."""
    editions = charger_editions(chemin_corpus, verbeux=verbeux)
    corrections = charger_corrections(chemin_corrections) if chemin_corrections else {"ordre": None, "liens": [], "segments_supprimes": []}

    groupes_plaques, plaques, registre_graveurs = construire_plaques(editions, ordre=corrections.get("ordre"), verbeux=verbeux)
    segments_supprimes = {(s["graveur"], s["annee_a"], s["annee_b"]) for s in corrections.get("segments_supprimes", [])}
    elements_svg, points, hl, hauteur = construire_donnees_svg(plaques, segments_supprimes=segments_supprimes)

    points_par_graveur = defaultdict(list)
    for i, pt in enumerate(points):
        points_par_graveur[pt["graveur"]].append((pt["annee"], i))
    for g in points_par_graveur:
        points_par_graveur[g].sort()
    points_par_graveur = dict(points_par_graveur)

    fleches_copies, non_resolus_copies = resoudre_fleches_copies(
        editions, points, points_par_graveur, registre_graveurs,
        corrections_liens=corrections.get("liens"), verbeux=verbeux,
    )
    ajouter_fleches_copies_svg(elements_svg, points, fleches_copies)
    ajouter_points_svg(elements_svg, points)

    return {
        "editions": editions, "corrections": corrections,
        "groupes_plaques": groupes_plaques, "plaques": plaques, "registre_graveurs": registre_graveurs,
        "elements_svg": elements_svg, "points": points, "points_par_graveur": points_par_graveur,
        "fleches_copies": fleches_copies, "non_resolus_copies": non_resolus_copies,
        "hauteur_ligne": hl, "hauteur": hauteur, "largeur": LARGEUR,
    }


# ─────────────────────────────────────────────
# Génération HTML — un seul template, deux modes :
#   editable=False : export statique (03_reemploi_plaques.ipynb), lecture seule
#   editable=True  : atelier web (reemploi_plaques_editeur.py), barre d'outils +
#                     appels /api/... qui persistent dans le fichier de corrections
# ─────────────────────────────────────────────

_CSS_BASE = r"""
  :root {
    --surface: #fffaf0; --texte-fort: #2b1e15; --texte-att: #6b5c4f; --trait: #d8cfc0;
    --contour-point: #3e2c23; --c-lien: #2a78d6; --c-transfert: #c0392b; --c-copie: #1d4e74; --bande: #f2e9d8;
    --c-manuel: #1d9e75;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface: #1a1a19; --texte-fort: #f2ece2; --texte-att: #c3baa9; --trait: #3a352c;
      --contour-point: #f2ece2; --c-lien: #3987e5; --c-transfert: #e0685a; --c-copie: #7fb3dd; --bande: #232019;
      --c-manuel: #3fcf9e;
    }
  }
  body { margin:0; font-family:Georgia,serif; background:var(--surface); color:var(--texte-fort); }
  .page { max-width:none; margin:0; padding:16px 24px 32px; box-sizing:border-box; position:relative; }
  h1 { font-size:19px; margin:0 0 4px; }
  p.souschapo { font-size:13px; color:var(--texte-att); margin:0 0 12px; }

  .legende { display:flex; gap:18px; flex-wrap:wrap; font-size:12px; margin:0 0 14px; }
  .legende .item { display:flex; align-items:center; gap:6px; }
  .legende .trait { display:inline-block; width:26px; height:0; border-top-width:2px; }
  .legende .reprise { border-top:2px solid var(--texte-att); }
  .legende .transfert { border-top:2px solid var(--c-transfert); }
  .legende .incertain { border-top:2px dashed var(--texte-att); }
  .legende .copie { border-top:2px dashed var(--c-copie); }
  .legende .manuel { border-top:2px dashed var(--c-manuel); }

  .cadre-frise { border:1px solid var(--trait); border-radius:6px; box-shadow:0 1px 6px rgba(0,0,0,.15); }
  .frise { display:block; width:100%; height:auto; }
  .bande-zebra { fill:var(--bande); }
  .etiquette-plaque { font-size:11px; fill:var(--texte-fort); }
  .grille-frise { stroke:var(--trait); stroke-width:1; }
  .etiquette-annee-frise { font-size:10px; fill:var(--texte-att); }
  .segment-reprise { stroke:var(--texte-att); stroke-width:1.5; }
  .segment-transfert { stroke:var(--c-transfert); stroke-width:2.5; }
  .segment-incertain { stroke:var(--texte-att); stroke-width:1.5; stroke-dasharray:3,3; }
  .fleche-copie { fill:none; stroke:var(--c-copie); stroke-width:1.8; stroke-dasharray:2,4; opacity:.9; }
  .fleche-copie-manuelle { stroke:var(--c-manuel); }
  .fleche-copie-reprise, .fleche-copie-transfert { stroke-dasharray:none; }
  .fleche-copie-transfert { stroke-width:2.2; }
  .fleche-copie.selectionnable { cursor:pointer; }
  .fleche-copie.selectionnable:hover { stroke-width:3.2; opacity:1; }
  .segment-reprise.selectionnable, .segment-transfert.selectionnable, .segment-incertain.selectionnable {
    cursor:pointer; }
  .segment-reprise.selectionnable:hover, .segment-incertain.selectionnable:hover { stroke-width:3; stroke:var(--c-manuel); }
  .segment-transfert.selectionnable:hover { stroke-width:4; }
  .point-frise { fill:var(--surface); stroke:var(--contour-point); stroke-width:1.6; cursor:pointer;
    transition:r .15s; }
  .point-frise:hover, .point-frise.actif { r:9; fill:var(--c-lien); }
  .point-frise.source-selectionnee { r:9; fill:var(--c-manuel); stroke:var(--c-manuel); }

  .action-tableau { margin:14px 0 0; }
  button.bascule { font-family:Georgia,serif; font-size:12px; background:none;
    border:1px solid var(--trait); color:var(--texte-fort); border-radius:4px; padding:5px 10px;
    cursor:pointer; }
  button.bascule.actif { background:var(--trait); font-weight:bold; }
  button.bascule:disabled { opacity:.4; cursor:default; }
  table.tableau-detaille { width:100%; border-collapse:collapse; font-size:12px; margin:8px 0;
    display:none; }
  table.tableau-detaille.visible { display:table; }
  table.tableau-detaille th, table.tableau-detaille td { text-align:left; padding:4px 8px;
    border-bottom:1px solid var(--trait); }
  table.tableau-detaille a { color:var(--c-lien); }

  .infobulle { position:absolute; pointer-events:none; background:var(--surface);
    border:1px solid var(--contour-point); border-radius:5px; padding:6px 10px; font-size:12px;
    max-width:260px; opacity:0; transition:opacity .1s; box-shadow:0 2px 8px rgba(0,0,0,.3); z-index:2000; }
  .infobulle.epinglee { pointer-events:auto; }
  .infobulle a { color:var(--c-lien); }
  .infobulle .fermer-infobulle { position:absolute; top:2px; right:6px; cursor:pointer;
    color:var(--texte-att); font-size:13px; }
"""

_CSS_EDITION = r"""
  .barre-edition { display:flex; gap:16px; flex-wrap:wrap; align-items:center; margin:0 0 12px;
    font-size:12px; }
  .groupe-outils { display:flex; gap:4px; align-items:center; }
  .groupe-outils .etiquette { color:var(--texte-att); margin-right:2px; }
  .aide-edition { color:var(--texte-att); font-size:11.5px; margin:6px 0 0; }
  .aide-edition.actif { color:var(--c-manuel); font-weight:bold; }
  .msg-edition { font-size:12px; color:var(--c-manuel); min-height:14px; }

  .panneau-ordre { display:none; border:1px solid var(--trait); border-radius:6px; padding:8px 10px;
    margin:0 0 12px; max-width:420px; max-height:280px; overflow-y:auto; font-size:12.5px; }
  .panneau-ordre.visible { display:block; }
  .panneau-ordre .ligne-ordre { display:flex; align-items:center; gap:8px; padding:5px 4px;
    border-bottom:1px solid var(--trait); border-top:2px solid transparent; cursor:grab;
    user-select:none; }
  .panneau-ordre .ligne-ordre:last-child { border-bottom:none; }
  .panneau-ordre .ligne-ordre.en-glissement { opacity:.35; }
  .panneau-ordre .ligne-ordre.survol-depot { border-top-color:var(--c-manuel); }
  .panneau-ordre .poignee { color:var(--texte-att); font-size:14px; line-height:1; }
  .panneau-ordre .nom-graveur { flex:1; }

  .menu-type-lien-fond { display:none; position:fixed; inset:0; background:rgba(0,0,0,.25); z-index:2999; }
  .menu-type-lien-fond.visible { display:block; }
  .menu-type-lien { display:none; position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
    background:var(--surface); border:1px solid var(--contour-point); border-radius:8px;
    padding:16px 18px; box-shadow:0 4px 20px rgba(0,0,0,.35); z-index:3000; font-size:13px;
    max-width:320px; }
  .menu-type-lien.visible { display:block; }
  .menu-type-lien p { margin:0 0 10px; }
  .menu-type-lien button { display:block; width:100%; margin:0 0 6px; padding:7px 10px;
    font-family:Georgia,serif; font-size:12.5px; border:1px solid var(--trait); border-radius:4px;
    background:var(--surface); color:var(--texte-fort); cursor:pointer; text-align:left; }
  .menu-type-lien button:hover { background:var(--trait); }
  .menu-type-lien button.annuler { color:var(--texte-att); margin-top:2px; }
"""


def _js_interactions():
    """Attache l'infobulle + le bouton tableau. Rappelable après un redessin (édition)."""
    return r"""
function attacherInteractions() {
  document.querySelectorAll('.point-frise').forEach(cercle => {
    const p = points[+cercle.dataset.i];
    cercle.addEventListener('mouseenter', () => {
      if (infobulleEpinglee || modeEditionLiens) return;
      cercle.classList.add('actif');
      infobulle.innerHTML = contenuApercu(p);
      infobulle.style.opacity = 1;
    });
    cercle.addEventListener('mousemove', (ev) => {
      if (infobulleEpinglee || modeEditionLiens) return;
      positionnerInfobulle(ev);
    });
    cercle.addEventListener('mouseleave', () => {
      if (infobulleEpinglee || modeEditionLiens) return;
      cercle.classList.remove('actif');
      infobulle.style.opacity = 0;
    });
    cercle.addEventListener('click', (ev) => {
      ev.stopPropagation();
      if (modeEditionLiens) { clicPointEdition(+cercle.dataset.i); return; }
      positionnerInfobulle(ev);
      infobulle.innerHTML = contenuDetaille(p) + '<span class="fermer-infobulle" title="Fermer">×</span>';
      infobulle.style.opacity = 1;
      infobulle.classList.add('epinglee');
      infobulleEpinglee = true;
      document.querySelectorAll('.point-frise.actif').forEach(c => c.classList.remove('actif'));
      cercle.classList.add('actif');
      infobulle.querySelector('.fermer-infobulle').addEventListener('click', fermerInfobulle);
    });
  });
}

const infobulle = document.getElementById('infobulle');
const page = document.querySelector('.page');
let infobulleEpinglee = false;

function contenuApercu(p) {
  return '<b>' + p.graveur + '</b><br>' +
    '<span>' + p.ville + ', ' + p.annee + ' · ' + libellesTechnique[p.technique] + '</span>' +
    '<br><i>' + p.editeur + '</i>' +
    (p.titre ? '<br>' + p.titre : '');
}
function contenuDetaille(p) {
  const lien = p.lien ? '<br><a href="' + p.lien + '" target="_blank">→ voir</a>' : '';
  return contenuApercu(p) + lien;
}
function positionnerInfobulle(ev) {
  const r = page.getBoundingClientRect();
  infobulle.style.left = (ev.clientX - r.left + 14) + 'px';
  infobulle.style.top = (ev.clientY - r.top + 14) + 'px';
}
function fermerInfobulle() {
  infobulleEpinglee = false;
  infobulle.classList.remove('epinglee');
  infobulle.style.opacity = 0;
  document.querySelectorAll('.point-frise.actif').forEach(c => c.classList.remove('actif'));
}
document.addEventListener('click', (ev) => {
  if (infobulleEpinglee && !infobulle.contains(ev.target) && !ev.target.classList.contains('point-frise')) {
    fermerInfobulle();
  }
});

const boutonTableau = document.getElementById('boutonTableau');
boutonTableau.addEventListener('click', () => {
  const tableau = document.getElementById('tableauDetaille');
  const visible = tableau.classList.toggle('visible');
  boutonTableau.textContent = visible ? 'Masquer le tableau détaillé' : 'Afficher le tableau détaillé';
});

attacherInteractions();
"""


def _js_edition():
    """Barre d'outils d'édition : tri, déplacement de ligne, édition des liens de copie.
    Toutes les actions passent par /api/... (voir reemploi_plaques_editeur.py) qui
    persistent dans le fichier de corrections et renvoient les données fraîches."""
    return r"""
let sourceSelectionnee = null;
const msgEdition = document.getElementById('msgEdition');

async function appelerApi(chemin, corps) {
  msgEdition.textContent = 'Enregistrement...';
  try {
    const r = await fetch(chemin, {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(corps)});
    const d = await r.json();
    if (!d.ok) { msgEdition.textContent = 'Erreur : ' + (d.erreur || '?'); return; }
    redessiner(d);
    msgEdition.textContent = d.message || 'Enregistré.';
  } catch (e) {
    msgEdition.textContent = 'Erreur réseau : ' + e;
  }
}

function redessiner(d) {
  document.querySelector('.frise').setAttribute('viewBox', '0 0 ' + d.largeur + ' ' + d.hauteur);
  document.getElementById('coucheFrise').innerHTML = d.frise_svg;
  document.getElementById('tableauDetaille').querySelector('tbody').innerHTML = d.lignes_tableau;
  points = d.points;
  attacherInteractions();
  attacherClicsFleches();
  attacherClicsSegments();
  remplirPanneauOrdre();
}

// Ordre courant des lignes = ordre d'apparition des graveurs dans `points`
// (les points sont émis groupe de plaques par groupe de plaques, dans l'ordre des lignes).
function ordreGraveursActuel() {
  const vus = new Set(), ordre = [];
  for (const p of points) {
    if (!vus.has(p.graveur)) { vus.add(p.graveur); ordre.push(p.graveur); }
  }
  return ordre;
}

document.querySelectorAll('#boutonsTri button').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('#boutonsTri button').forEach(x => x.classList.remove('actif'));
    b.classList.add('actif');
    appelerApi('/api/ordre', {ordre: b.dataset.ordre});
  });
});

const panneauOrdre = document.getElementById('panneauOrdre');
const boutonReordonner = document.getElementById('boutonReordonner');
boutonReordonner.addEventListener('click', () => {
  const visible = panneauOrdre.classList.toggle('visible');
  boutonReordonner.classList.toggle('actif', visible);
  if (visible) remplirPanneauOrdre();
});

function remplirPanneauOrdre() {
  if (!panneauOrdre.classList.contains('visible')) return;
  const ordre = ordreGraveursActuel();
  panneauOrdre.innerHTML = ordre.map((g, i) => `
    <div class="ligne-ordre" draggable="true" data-i="${i}">
      <span class="poignee" title="Glisser pour réordonner">⠿</span>
      <span class="nom-graveur">${g}</span>
    </div>`).join('');

  let indexGlisse = null;
  panneauOrdre.querySelectorAll('.ligne-ordre').forEach(ligne => {
    ligne.addEventListener('dragstart', () => {
      indexGlisse = +ligne.dataset.i;
      ligne.classList.add('en-glissement');
    });
    ligne.addEventListener('dragend', () => {
      ligne.classList.remove('en-glissement');
    });
    ligne.addEventListener('dragover', (ev) => {
      ev.preventDefault();
      ligne.classList.add('survol-depot');
    });
    ligne.addEventListener('dragleave', () => {
      ligne.classList.remove('survol-depot');
    });
    ligne.addEventListener('drop', (ev) => {
      ev.preventDefault();
      ligne.classList.remove('survol-depot');
      const indexCible = +ligne.dataset.i;
      if (indexGlisse === null || indexGlisse === indexCible) return;
      const nouveau = ordre.slice();
      const [deplace] = nouveau.splice(indexGlisse, 1);
      nouveau.splice(indexCible, 0, deplace);
      document.querySelectorAll('#boutonsTri button').forEach(x => x.classList.remove('actif'));
      appelerApi('/api/ordre', {ordre: nouveau});
    });
  });
}
remplirPanneauOrdre();

const boutonEditionLiens = document.getElementById('boutonEditionLiens');
boutonEditionLiens.addEventListener('click', () => {
  modeEditionLiens = !modeEditionLiens;
  boutonEditionLiens.classList.toggle('actif', modeEditionLiens);
  document.getElementById('aideEdition').classList.toggle('actif', modeEditionLiens);
  sourceSelectionnee = null;
  document.querySelectorAll('.point-frise.source-selectionnee').forEach(c => c.classList.remove('source-selectionnee'));
  fermerInfobulle();
});

function clicPointEdition(i) {
  if (sourceSelectionnee === null) {
    sourceSelectionnee = i;
    document.querySelector('.point-frise[data-i="' + i + '"]').classList.add('source-selectionnee');
    msgEdition.textContent = 'Source : ' + points[i].graveur + ' (' + points[i].annee + ') — clique la seconde édition liée.';
    return;
  }
  if (sourceSelectionnee === i) {
    document.querySelector('.point-frise[data-i="' + i + '"]').classList.remove('source-selectionnee');
    sourceSelectionnee = null;
    msgEdition.textContent = '';
    return;
  }
  const src = points[sourceSelectionnee], cible = points[i];
  document.querySelector('.point-frise[data-i="' + sourceSelectionnee + '"]').classList.remove('source-selectionnee');
  sourceSelectionnee = null;
  demanderTypeLien(src, cible);
}

const menuTypeLien = document.getElementById('menuTypeLien');
const menuTypeLienFond = document.getElementById('menuTypeLienFond');
let choixLienEnAttente = null;

function demanderTypeLien(src, cible) {
  choixLienEnAttente = {src, cible};
  menuTypeLien.classList.add('visible');
  menuTypeLienFond.classList.add('visible');
}
function fermerMenuTypeLien() {
  menuTypeLien.classList.remove('visible');
  menuTypeLienFond.classList.remove('visible');
  choixLienEnAttente = null;
}
menuTypeLienFond.addEventListener('click', fermerMenuTypeLien);
menuTypeLien.querySelectorAll('button').forEach(btn => {
  btn.addEventListener('click', () => {
    const type = btn.dataset.type;
    const attente = choixLienEnAttente;
    fermerMenuTypeLien();
    if (!type || !attente) return;
    appelerApi('/api/lien', {
      action: 'override', type,
      graveur_source: attente.src.graveur, annee_source: attente.src.annee,
      graveur_cible: attente.cible.graveur, annee_cible: attente.cible.annee,
    });
  });
});

const LIBELLE_TYPE_LIEN = {copie: 'copie', reprise: 'réimpression', transfert: 'transmission'};

function attacherClicsFleches() {
  document.querySelectorAll('.fleche-copie').forEach(chemin => {
    chemin.classList.add('selectionnable');
    chemin.addEventListener('click', (ev) => {
      ev.stopPropagation();
      if (modeEditionLiens) return;  // en mode édition, laisser cliquer les points
      const iSource = +chemin.dataset.iSource, iCible = +chemin.dataset.iCible;
      const src = points[iSource], cible = points[iCible];
      const libelle = LIBELLE_TYPE_LIEN[chemin.dataset.type] || 'lien';
      if (!confirm('Supprimer ce lien (' + libelle + ') : ' + src.graveur + ' (' + src.annee + ') → ' + cible.graveur + ' (' + cible.annee + ') ?')) return;
      appelerApi('/api/lien_supprimer', {
        graveur_source: src.graveur, annee_source: src.annee,
        graveur_cible: cible.graveur, annee_cible: cible.annee,
      });
    });
  });
}
attacherClicsFleches();

const LIBELLE_TYPE_SEGMENT = {reprise: 'réimpression', transfert: 'transmission', incertain: 'incertain'};

function attacherClicsSegments() {
  document.querySelectorAll('.segment-reprise, .segment-transfert, .segment-incertain').forEach(trait => {
    trait.classList.add('selectionnable');
    trait.addEventListener('click', (ev) => {
      ev.stopPropagation();
      if (modeEditionLiens) return;
      const libelle = LIBELLE_TYPE_SEGMENT[trait.dataset.type] || 'lien';
      if (!confirm('Supprimer ce segment (' + libelle + ') entre ' + trait.dataset.graveur + ' ' +
          trait.dataset.anneeA + ' et ' + trait.dataset.anneeB + ' ? Les deux éditions ' +
          'resteront affichées, seul le trait qui les relie disparaîtra.')) return;
      appelerApi('/api/segment_supprimer', {
        graveur: trait.dataset.graveur,
        annee_a: +trait.dataset.anneeA, annee_b: +trait.dataset.anneeB,
      });
    });
  });
}
attacherClicsSegments();
"""


def generer_html(largeur, hauteur, frise_svg, lignes_tableau, points, libelles_technique, editable=False):
    """Génère la page HTML complète. `editable=True` ajoute la barre d'outils d'édition
    (tri, réordonnancement, correction des liens) et le JS qui appelle les endpoints
    /api/... de reemploi_plaques_editeur.py — sans backend, ces contrôles n'ont pas de
    sens, donc absents de l'export statique (editable=False)."""
    barre_edition = ""
    if editable:
        barre_edition = """
  <div class="barre-edition">
    <div class="groupe-outils" id="boutonsTri">
      <span class="etiquette">Ordre des lignes :</span>
      <button class="bascule actif" data-ordre="auto">Auto (plus réemployé d'abord)</button>
      <button class="bascule" data-ordre="alphabetique">Alphabétique</button>
      <button class="bascule" data-ordre="chronologique">Chronologique</button>
    </div>
    <div class="groupe-outils">
      <button class="bascule" id="boutonReordonner" title="Réordonner librement les lignes">Réordonner (↕)</button>
      <button class="bascule" id="boutonEditionLiens">Éditer les liens</button>
    </div>
    <span class="msg-edition" id="msgEdition"></span>
  </div>
  <div class="panneau-ordre" id="panneauOrdre"></div>
  <p class="aide-edition" id="aideEdition">Mode édition activé — clique un point (l'édition), puis un
    second point (l'autre édition liée), pour choisir le type de lien entre les deux (réimpression,
    transmission ou copie). Clique n'importe quelle flèche déjà tracée pour la supprimer.</p>
  <div class="menu-type-lien-fond" id="menuTypeLienFond"></div>
  <div class="menu-type-lien" id="menuTypeLien">
    <p>Quel type de lien entre ces deux éditions ?</p>
    <button data-type="reprise">Réimpression — même éditeur, mêmes plaques</button>
    <button data-type="transfert">Transmission — plaques passées à un autre éditeur</button>
    <button data-type="copie">Copie — imitation, pas les mêmes plaques</button>
    <button class="annuler" data-type="">Annuler</button>
  </div>
"""

    js_edition = _js_edition() if editable else ""
    css_edition = _CSS_EDITION if editable else ""

    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>Réemploi des plaques gravées entre éditeurs</title>
<style>{_CSS_BASE}{css_edition}</style></head><body>
<div class="page">
  <h1>Réemploi des plaques gravées entre éditeurs</h1>
  <p class="souschapo">Une ligne = un jeu de plaques, un point = une édition qui l'utilise,
    positionnée par année. La plupart des lignes réunissent au moins deux éditions réemployées ;
    quelques-unes n'en montrent qu'une seule, ajoutée pour ancrer une flèche de copie vers ou
    depuis un autre jeu de plaques. Cliquer un point affiche le détail avec le lien "voir" (le
    survol seul n'affiche qu'un aperçu).</p>
  <div class="legende">
    <div class="item"><span class="trait reprise"></span>réimpression (même éditeur)</div>
    <div class="item"><span class="trait transfert"></span>transmission à un autre éditeur</div>
    <div class="item"><span class="trait incertain"></span>incertain (éditeur non identifié)</div>
    <div class="item"><span class="trait copie"></span>copie des illustrations d'un autre jeu de plaques</div>
    <div class="item"><span class="trait manuel"></span>lien corrigé/ajouté manuellement (réimpression, transmission ou copie)</div>
  </div>
{barre_edition}
  <div class="cadre-frise">
    <svg class="frise" viewBox="0 0 {largeur} {hauteur}">
      <defs>
        <marker id="fleche-transfert" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--c-transfert)"/>
        </marker>
        <marker id="fleche-copie-tete" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--c-copie)"/>
        </marker>
        <marker id="fleche-manuel-transfert-tete" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--c-manuel)"/>
        </marker>
      </defs>
      <g id="coucheFrise">{frise_svg}</g>
    </svg>
  </div>
  <div class="action-tableau">
    <button class="bascule" id="boutonTableau">Afficher le tableau détaillé</button>
    <table class="tableau-detaille" id="tableauDetaille">
      <thead><tr><th>Plaques</th><th>Année</th><th>Ville</th><th>Éditeur</th><th>Titre</th><th>Technique</th><th>Lien</th></tr></thead>
      <tbody>
        {lignes_tableau}
      </tbody>
    </table>
  </div>
  <div class="infobulle" id="infobulle"></div>
</div>
<script>
  let points = {json.dumps(points, ensure_ascii=False)};
  const libellesTechnique = {json.dumps(libelles_technique, ensure_ascii=False)};
  let modeEditionLiens = false;
  {_js_interactions()}
  {js_edition}
</script>
</body></html>"""
