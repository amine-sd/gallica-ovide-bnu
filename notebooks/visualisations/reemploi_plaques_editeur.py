#!/usr/bin/env python3
"""
Atelier de correction — Réemploi des plaques gravées entre éditeurs.

Source de vérité pour les données = retours_celine/BNU_corpus.ods (jamais modifié).
Les corrections faites ici (ordre des lignes, liens de copie ajoutés/corrigés/supprimés)
sont enregistrées à part, dans retours_celine/corrections_reemploi_plaques.json, et
relues aussi bien par ce serveur que par 03_reemploi_plaques.ipynb (export statique) —
même principe que l'atelier de regroupement des Bibles (classes_celine/) : la donnée
source n'est jamais touchée, seules les corrections le sont.

Fonctions :
  - Ordre des lignes : auto (le plus réemployé d'abord) / alphabétique / chronologique,
    ou manuel via le panneau "Réordonner" (glisser-déposer)
  - Édition des liens : "Éditer les liens" puis cliquer deux points pour créer/corriger un
    lien entre eux (choix du type : réimpression, transmission ou copie) ; cliquer n'importe
    quelle flèche déjà tracée — qu'elle soit auto-résolue ou manuelle — propose de la supprimer
  - Suppression d'un segment automatique intra-ligne (réimpression/transmission/incertain) :
    cliquer directement le trait entre deux points d'une même ligne propose de le supprimer,
    si l'utilisateur juge que ces deux éditions ne partagent en réalité pas le même jeu de
    plaques (les deux points restent affichés, seul le trait disparaît)

Chaque action recalcule tout côté serveur et repousse la frise/le tableau à jour au
navigateur (pas de rechargement de page).

Lancement :  python reemploi_plaques_editeur.py   puis  http://localhost:8060
"""

import os

from flask import Flask, request, jsonify

import reemploi_plaques_utils as u

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHEMIN_CORPUS = os.path.join(RACINE, "retours_celine", "BNU_corpus.ods")
CHEMIN_CORRECTIONS = os.path.join(RACINE, "retours_celine", "corrections_reemploi_plaques.json")
PORT = 8060

app = Flask(__name__)
ETAT = {"editions": None}


def recalculer():
    """Relit les corrections (le corpus lui-même ne change pas en cours de session,
    chargé une seule fois) et reconstruit plaques/points/SVG. Renvoie un dict prêt à
    passer à generer_html() ou à sérialiser pour une réponse AJAX."""
    if ETAT["editions"] is None:
        ETAT["editions"] = u.charger_editions(CHEMIN_CORPUS, verbeux=False)
    corrections = u.charger_corrections(CHEMIN_CORRECTIONS)
    groupes_plaques, plaques, registre_graveurs = u.construire_plaques(
        ETAT["editions"], ordre=corrections.get("ordre"), verbeux=False)
    segments_supprimes = {(s["graveur"], s["annee_a"], s["annee_b"]) for s in corrections.get("segments_supprimes", [])}
    elements_svg, points, hl, hauteur = u.construire_donnees_svg(plaques, segments_supprimes=segments_supprimes)
    points_par_graveur = {}
    for i, pt in enumerate(points):
        points_par_graveur.setdefault(pt["graveur"], []).append((pt["annee"], i))
    for g in points_par_graveur:
        points_par_graveur[g].sort()
    fleches_copies, non_resolus_copies = u.resoudre_fleches_copies(
        ETAT["editions"], points, points_par_graveur, registre_graveurs,
        corrections_liens=corrections.get("liens"), verbeux=False,
    )
    u.ajouter_fleches_copies_svg(elements_svg, points, fleches_copies)
    u.ajouter_points_svg(elements_svg, points)
    return {
        "corrections": corrections, "plaques": plaques, "points": points,
        "elements_svg": elements_svg, "hauteur": hauteur, "largeur": u.LARGEUR,
    }


def reponse_fraiche(message=""):
    r = recalculer()
    return jsonify(
        ok=True, message=message,
        largeur=r["largeur"], hauteur=r["hauteur"],
        frise_svg="\n".join(r["elements_svg"]),
        lignes_tableau=u.construire_lignes_tableau(r["plaques"]),
        points=r["points"],
    )


def _remplacer_lien_source(corrections, graveur_source, annee_source, nouvelle_entree):
    """Retire toute correction override/suppression existante pour cette édition source
    (une source ne peut avoir qu'une seule correction active à la fois — la plus récente
    gagne) puis ajoute la nouvelle. Les entrées "ajout" (non utilisées par cet atelier,
    mais possibles via le fichier de corrections) sont laissées intactes."""
    liens = [
        c for c in corrections.get("liens", [])
        if c["action"] == "ajout" or c["graveur_source"] != graveur_source or c["annee_source"] != annee_source
    ]
    liens.append(nouvelle_entree)
    corrections["liens"] = liens


@app.route("/")
def index():
    r = recalculer()
    return u.generer_html(
        largeur=r["largeur"], hauteur=r["hauteur"],
        frise_svg="\n".join(r["elements_svg"]),
        lignes_tableau=u.construire_lignes_tableau(r["plaques"]),
        points=r["points"], libelles_technique=u.LIBELLES_TECHNIQUE,
        editable=True,
    )


@app.route("/api/ordre", methods=["POST"])
def api_ordre():
    data = request.get_json()
    corrections = u.charger_corrections(CHEMIN_CORRECTIONS)
    corrections["ordre"] = data.get("ordre")
    u.sauver_corrections(CHEMIN_CORRECTIONS, corrections)
    return reponse_fraiche("Ordre mis à jour.")


LIBELLE_TYPE_LIEN = {"copie": "copie", "reprise": "réimpression", "transfert": "transmission"}


@app.route("/api/lien", methods=["POST"])
def api_lien():
    data = request.get_json()
    champs = ("graveur_source", "annee_source", "graveur_cible", "annee_cible")
    if any(c not in data for c in champs):
        return jsonify(ok=False, erreur="champ manquant"), 400
    type_lien = data.get("type", "copie")
    if type_lien not in LIBELLE_TYPE_LIEN:
        return jsonify(ok=False, erreur="type de lien invalide"), 400
    entree = {"action": "override", "type": type_lien, **{c: data[c] for c in champs}}
    corrections = u.charger_corrections(CHEMIN_CORRECTIONS)
    _remplacer_lien_source(corrections, entree["graveur_source"], entree["annee_source"], entree)
    u.sauver_corrections(CHEMIN_CORRECTIONS, corrections)
    return reponse_fraiche(
        f"Lien ({LIBELLE_TYPE_LIEN[type_lien]}) corrigé : {entree['graveur_source']} ({entree['annee_source']}) → "
        f"{entree['graveur_cible']} ({entree['annee_cible']})."
    )


@app.route("/api/lien_supprimer", methods=["POST"])
def api_lien_supprimer():
    data = request.get_json()
    if "graveur_source" not in data or "annee_source" not in data:
        return jsonify(ok=False, erreur="champ manquant"), 400
    entree = {"action": "suppression", "graveur_source": data["graveur_source"], "annee_source": data["annee_source"]}
    corrections = u.charger_corrections(CHEMIN_CORRECTIONS)
    _remplacer_lien_source(corrections, entree["graveur_source"], entree["annee_source"], entree)
    u.sauver_corrections(CHEMIN_CORRECTIONS, corrections)
    return reponse_fraiche(f"Lien supprimé depuis {entree['graveur_source']} ({entree['annee_source']}).")


@app.route("/api/segment_supprimer", methods=["POST"])
def api_segment_supprimer():
    """Supprime un segment automatique (réimpression/transmission/incertain) au sein d'une
    même ligne : l'utilisateur juge que ces deux éditions ne partagent en réalité pas le
    même jeu de plaques. Les deux points restent affichés, seul le trait disparaît."""
    data = request.get_json()
    champs = ("graveur", "annee_a", "annee_b")
    if any(c not in data for c in champs):
        return jsonify(ok=False, erreur="champ manquant"), 400
    entree = {c: data[c] for c in champs}
    corrections = u.charger_corrections(CHEMIN_CORRECTIONS)
    segments = [s for s in corrections.get("segments_supprimes", []) if s != entree]
    segments.append(entree)
    corrections["segments_supprimes"] = segments
    u.sauver_corrections(CHEMIN_CORRECTIONS, corrections)
    return reponse_fraiche(f"Segment supprimé : {entree['graveur']} ({entree['annee_a']} → {entree['annee_b']}).")


if __name__ == "__main__":
    print("=" * 55)
    print("  Atelier de correction — Réemploi des plaques gravées")
    print("=" * 55)
    print("Corpus      :", CHEMIN_CORPUS)
    print("Corrections :", CHEMIN_CORRECTIONS)
    recalculer()
    print(f"\nOuvre dans ton navigateur : http://localhost:{PORT}")
    print("Ctrl+C pour arrêter.\n")
    app.run(port=PORT, debug=False)
