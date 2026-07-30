"""
rag_utils.py — Fonctions partagees pour le mini RAG iconographique (Ovide / Bibles)
=====================================================================================
Usage : from rag_utils import charger_siglip, charger_e5, charger_index, embed_texte, ...

Meme convention que gallica_utils.py (racine de notebooks/) : les fonctions de
chargement (charger_*) retournent les objets charges, les autres fonctions les
prennent en parametre explicite plutot que de dependre de variables globales —
chaque notebook/script reste libre de charger une seule fois et de reutiliser.
"""

import re
import unicodedata

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

NOM_MODELE_SIGLIP = "google/siglip-base-patch16-224"
NOM_MODELE_E5 = "intfloat/multilingual-e5-small"

# colonnes de faits exacts (ville/graveur/technique) : la similarite semantique
# seule confond des mots proches (ex. "Lyon" faisait remonter "Hippomene en
# Lion") — un mot de la requete qui matche mot-pour-mot une valeur de ces
# colonnes est un signal beaucoup plus fort qu'une similarite d'embedding.
COLONNES_FILTRABLES = ["ville", "graveur", "technique"]

COLS_COMMUNES = ["chemin", "source", "theme", "embedding", "embedding_metadonnees",
                  "url_page", "url_image"]
COLS_OVIDE_EXTRA = ["titre", "ville", "annee", "graveur", "technique",
                     "type_iconographique", "famille_iconographique",
                     "theme_precis", "description_planche"]


# ─────────────────────────────────────────────
# Chargement — modeles et index
# ─────────────────────────────────────────────

def charger_siglip(device):
    """Retourne (processor, model_siglip). SigLIP encode image et texte dans le
    meme espace 768D (pas de couche de projection separee comme CLIP)."""
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(NOM_MODELE_SIGLIP)
    model_siglip = AutoModel.from_pretrained(NOM_MODELE_SIGLIP).to(device).eval()
    return processor, model_siglip


def charger_e5():
    """Modele dedie a la similarite texte-texte (metadonnees/descriptions) :
    SigLIP seul donnait de mauvais resultats ici (ex. "Cadmus combat un
    serpent" faisait remonter un document Cain et Abel avant le bon document)
    car il n'est entraine que pour comparer image<->texte, jamais texte<->texte."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(NOM_MODELE_E5)


def charger_index(dossier_vector_db):
    """Charge et unifie les bases Bibles + Ovide (voir docs/mini_rag_iconographique.md).
    Retourne (index, X_index, X_index_meta)."""
    base_bibles = pd.read_pickle(dossier_vector_db / "bibles_siglip.pkl")
    base_bibles["source"] = "bible"

    base_ovide = pd.read_pickle(dossier_vector_db / "ovide_corpus_complet_siglip.pkl")
    base_ovide["source"] = "ovide"

    base_bibles = base_bibles.reindex(columns=COLS_COMMUNES + COLS_OVIDE_EXTRA)
    base_ovide = base_ovide.reindex(columns=COLS_COMMUNES + COLS_OVIDE_EXTRA)

    index = pd.concat([base_bibles, base_ovide], ignore_index=True)
    X_index = np.array(index["embedding"].tolist())
    X_index_meta = np.array(index["embedding_metadonnees"].tolist())
    return index, X_index, X_index_meta


# ─────────────────────────────────────────────
# Encodage requete (texte / image)
# ─────────────────────────────────────────────

def embed_texte(requete, processor, model_siglip, device):
    inputs = processor(text=[requete], padding="max_length", return_tensors="pt").to(device)
    import torch
    with torch.no_grad():
        vec = model_siglip.get_text_features(**inputs).pooler_output
    return vec.cpu().numpy()[0]


def embed_image(chemin_ou_image, processor, model_siglip, device):
    import torch
    img = chemin_ou_image if isinstance(chemin_ou_image, Image.Image) else Image.open(chemin_ou_image)
    img = img.convert("RGB").convert("L").convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        vec = model_siglip.get_image_features(**inputs).pooler_output
    return vec.cpu().numpy()[0]


# ─────────────────────────────────────────────
# Recherche
# ─────────────────────────────────────────────

def rechercher(vecteur_requete, index, X_index, k=5, source=None):
    """Recherche visuelle seule (SigLIP). Utilisee pour les requetes image : pas
    de texte a comparer a l'index metadonnees dans ce cas. `source` optionnel
    ("bible" / "ovide") restreint la recherche a un seul corpus."""
    sous_index = index if source is None else index[index["source"] == source]
    X = X_index if source is None else X_index[sous_index.index.to_numpy()]
    sims = cosine_similarity(vecteur_requete.reshape(1, -1), X)[0]
    ordre = sims.argsort()[::-1][:k]
    resultats = sous_index.iloc[ordre].copy()
    resultats["similarite"] = sims[ordre]
    return resultats.reset_index(drop=True)


def _normaliser(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.lower()


def masque_correspondance_exacte(requete_texte, index):
    """Pour chaque colonne filtrable, verifie si un mot significatif (>=4
    caracteres) de la valeur de cette colonne apparait comme mot entier dans la
    requete (insensible a la casse/accents). Retourne un masque booleen : au
    moins une correspondance exacte pour cette ligne."""
    q_norm = _normaliser(requete_texte)
    mots_requete = set(re.findall(r"\w+", q_norm))

    masque = np.zeros(len(index), dtype=bool)
    for col in COLONNES_FILTRABLES:
        valeurs = index[col]
        for i, val in enumerate(valeurs):
            if pd.isna(val) or masque[i]:
                continue
            mots_valeur = re.findall(r"\w+", _normaliser(val))
            if any(len(m) >= 4 and m in mots_requete for m in mots_valeur):
                masque[i] = True
    return masque


def rechercher_hybride(requete_texte, index, X_index, X_index_meta,
                        processor, model_siglip, modele_e5, device,
                        k=5, k_rrf=60):
    """Combine similarite visuelle (SigLIP), similarite metadonnees (e5) et
    correspondance exacte (ville/graveur/technique) par Reciprocal Rank Fusion
    — les modeles ont des echelles de score tres differentes (SigLIP
    texte-image ~0.1-0.2, e5 ~0.75-0.85), la RRF ne compare que des rangs donc
    insensible a ce probleme d'echelle. La correspondance exacte est traitee
    comme un 3e "classement" (rang 1 si match, dernier rang sinon) plutot que
    comme un filtre dur, pour rester tolerant si aucune ligne ne matche
    exactement. Ne s'applique qu'aux requetes texte : pour une image, on n'a
    pas de texte a comparer a l'index metadonnees (rechercher() visuel seul
    reste utilise pour les images)."""
    v_visuel = embed_texte(requete_texte, processor, model_siglip, device)
    v_meta = modele_e5.encode([f"query: {requete_texte}"], normalize_embeddings=True)[0]

    sims_visu = cosine_similarity(v_visuel.reshape(1, -1), X_index)[0]
    sims_meta = cosine_similarity(v_meta.reshape(1, -1), X_index_meta)[0]
    masque_exact = masque_correspondance_exacte(requete_texte, index)

    rangs_visu = (-sims_visu).argsort().argsort() + 1
    rangs_meta = (-sims_meta).argsort().argsort() + 1
    rang_exact = np.where(masque_exact, 1, len(index))
    score_rrf = 1 / (k_rrf + rangs_visu) + 1 / (k_rrf + rangs_meta) + 1 / (k_rrf + rang_exact)

    ordre = np.argsort(-score_rrf)[:k]
    resultats = index.iloc[ordre].copy()
    resultats["similarite"] = sims_visu[ordre]
    resultats["sim_metadonnees"] = sims_meta[ordre]
    resultats["correspondance_exacte"] = masque_exact[ordre]
    resultats["score_rrf"] = score_rrf[ordre]
    return resultats.reset_index(drop=True)


# ─────────────────────────────────────────────
# Affichage / contexte pour le LLM
# ─────────────────────────────────────────────

def libelle_resultat(row):
    """Etiquette d'affichage : le theme si connu, puis le theme precis
    (planche par planche, Solis/Salomon), sinon les metadonnees d'edition
    disponibles (graveur, annee) plutot que de reduire a un simple "sans theme"."""
    if pd.notna(row.theme):
        return str(row.theme)
    if pd.notna(getattr(row, "theme_precis", None)):
        return str(row.theme_precis)
    if row.source == "ovide":
        morceaux = []
        graveur = getattr(row, "graveur", None)
        if pd.notna(graveur):
            morceaux.append(str(graveur).split(",")[0].strip())
        annee = getattr(row, "annee", None)
        if pd.notna(annee):
            morceaux.append(str(annee))
        if morceaux:
            return ", ".join(morceaux)
    return "(sans theme)"


def contexte_depuis_resultats(resultats):
    """Met en forme les resultats en texte lisible par le LLM. Le score de
    similarite ne sert qu'au classement, jamais inclus ici : le donner au LLM
    le pousse a le reciter dans sa reponse malgre la consigne du prompt
    systeme de ne pas le faire — plus fiable de simplement ne pas le lui donner."""
    lignes = []
    for i, row in enumerate(resultats.itertuples(), 1):
        ligne = f"{i}. [{row.source}]"
        if pd.notna(row.theme):
            ligne += f" theme={row.theme}"
        if row.source == "ovide":
            extras = []
            for champ in ["titre", "ville", "annee", "graveur", "technique",
                          "type_iconographique", "famille_iconographique",
                          "theme_precis"]:
                valeur = getattr(row, champ, None)
                if pd.notna(valeur):
                    extras.append(f"{champ}={valeur}")
            description = getattr(row, "description_planche", None)
            if pd.notna(description):
                # planche a plusieurs episodes combines : tronquee pour ne pas
                # exploser le contexte envoye au LLM avec k illustrations
                texte = str(description)
                if len(texte) > 300:
                    texte = texte[:300] + "..."
                extras.append(f"description_planche={texte}")
            if extras:
                ligne += " (" + ", ".join(extras) + ")"
        lignes.append(ligne)
    return "\n".join(lignes)
