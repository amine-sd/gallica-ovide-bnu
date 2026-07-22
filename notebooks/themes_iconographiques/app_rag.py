"""Interface de chat (Gradio) pour tester le mini RAG iconographie Ovide / Bibles.

Reprend l'index + les fonctions de retrieval.ipynb et rag_generation.ipynb, exposees
dans une fenetre de chat façon ChatGPT : texte ou image en entree, recherche
vectorielle hybride (SigLIP + e5 + filtre exact), puis reponse redigee par un LLM
local (Ollama). Un menu "Options" permet de choisir le modele de generation :
vision (qwen2.5vl, voit reellement les images — par defaut) ou texte seul
(llama3.2, ne recoit que les metadonnees — utile pour comparer les deux approches).

Lancement :
    ./.venv/bin/python notebooks/themes_iconographiques/app_rag.py

Necessite Ollama demarre (`ollama serve`) avec les deux modeles deja telecharges
(`ollama pull qwen2.5vl`, `ollama pull llama3.2`). Avec share=True, un lien public
temporaire (72h) est imprime dans le terminal (gradio.live) — c'est ce lien qu'il
faut envoyer au tuteur.
"""

import base64
import re
import unicodedata
from io import BytesIO
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoProcessor

RACINE = Path(__file__).resolve().parent.parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", DEVICE)

# --- 1. Index unifie (identique a retrieval.ipynb) -------------------------------
DOSSIER_VECTOR_DB = RACINE / "data" / "vector_bases"

base_bibles = pd.read_pickle(DOSSIER_VECTOR_DB / "bibles_siglip.pkl")
base_bibles["source"] = "bible"

base_ovide = pd.read_pickle(DOSSIER_VECTOR_DB / "ovide_corpus_complet_siglip.pkl")
base_ovide["source"] = "ovide"

# ovide_corpus_complet_siglip.pkl (2191 illustrations, 28 editions) remplace
# ovide_bnu_corpus_siglip.pkl (73) : memes 73 themes connus conserves, ~2100 de
# plus sans theme individuel mais avec metadonnees d'edition (Synthese BNU_corpus.ods).
COLS_COMMUNES = ["chemin", "source", "theme", "embedding", "embedding_metadonnees",
                  "url_page", "url_image"]
COLS_OVIDE_EXTRA = ["titre", "ville", "annee", "graveur", "technique",
                     "type_iconographique", "famille_iconographique",
                     "theme_precis", "description_planche"]

base_bibles = base_bibles.reindex(columns=COLS_COMMUNES + COLS_OVIDE_EXTRA)
base_ovide = base_ovide.reindex(columns=COLS_COMMUNES + COLS_OVIDE_EXTRA)

index = pd.concat([base_bibles, base_ovide], ignore_index=True)
X_index = np.array(index["embedding"].tolist())
X_index_meta = np.array(index["embedding_metadonnees"].tolist())
print(f"Index charge : {len(index)} illustrations "
      f"({(index['source'] == 'bible').sum()} Bibles, {(index['source'] == 'ovide').sum()} Ovide)")

# --- 2. SigLIP complet (texte + image, meme espace) -------------------------------
NOM_MODELE_SIGLIP = "google/siglip-base-patch16-224"
processor = AutoProcessor.from_pretrained(NOM_MODELE_SIGLIP)
model_siglip = AutoModel.from_pretrained(NOM_MODELE_SIGLIP).to(DEVICE).eval()
print("SigLIP charge")

# modele dedie a la similarite texte-texte (metadonnees/descriptions) : SigLIP
# seul donnait de mauvais resultats ici (ex. "Cadmus combat un serpent" faisait
# remonter un document Cain et Abel avant le bon document Cadmus) car il n'est
# entraine que pour comparer image<->texte, jamais texte<->texte.
from sentence_transformers import SentenceTransformer

modele_e5 = SentenceTransformer("intfloat/multilingual-e5-small")
print("Modele e5 (metadonnees) charge")


def embed_texte(requete):
    inputs = processor(text=[requete], padding="max_length", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        vec = model_siglip.get_text_features(**inputs).pooler_output
    return vec.cpu().numpy()[0]


def embed_image(chemin_ou_image):
    img = chemin_ou_image if isinstance(chemin_ou_image, Image.Image) else Image.open(chemin_ou_image)
    img = img.convert("RGB").convert("L").convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        vec = model_siglip.get_image_features(**inputs).pooler_output
    return vec.cpu().numpy()[0]


def rechercher(vecteur_requete, k=5):
    sims = cosine_similarity(vecteur_requete.reshape(1, -1), X_index)[0]
    ordre = sims.argsort()[::-1][:k]
    resultats = index.iloc[ordre].copy()
    resultats["similarite"] = sims[ordre]
    return resultats.reset_index(drop=True)


# colonnes de faits exacts (ville/graveur/technique) : la similarite semantique
# seule confond des mots proches (ex. "Lyon" faisait remonter "Hippomene en
# Lion") — un mot de la requete qui matche mot-pour-mot une valeur de ces
# colonnes est un signal beaucoup plus fort qu'une similarite d'embedding.
COLONNES_FILTRABLES = ["ville", "graveur", "technique"]


def _normaliser(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.lower()


def masque_correspondance_exacte(requete_texte):
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


def rechercher_hybride(requete_texte, k=5, k_rrf=60):
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
    v_visuel = embed_texte(requete_texte)
    v_meta = modele_e5.encode([f"query: {requete_texte}"], normalize_embeddings=True)[0]

    sims_visu = cosine_similarity(v_visuel.reshape(1, -1), X_index)[0]
    sims_meta = cosine_similarity(v_meta.reshape(1, -1), X_index_meta)[0]
    masque_exact = masque_correspondance_exacte(requete_texte)

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


# --- 3. Generation (Ollama local) --------------------------------------------------
OLLAMA_URL = "http://localhost:11434"
MODELE_TEXTE = "llama3.2"
MODELE_VISION = "qwen2.5vl"  # bien plus fiable que llava:7b (pas de refus reflexe, vraie differenciation multi-images)

PROMPT_SYSTEME = """Tu es un expert en iconographie de la Renaissance europeenne (16e-17e s.),
specialise dans les editions illustrees des Metamorphoses d'Ovide et les Bibles
illustrees (MDZ/BSB).

Pour chaque illustration retrouvee par recherche vectorielle (SigLIP, similarite
cosinus), les champs disponibles varient : score de similarite toujours present,
theme precis pour une minorite seulement (la plupart des illustrations Ovide ne
sont pas cataloguees planche par planche), et pour les Ovide d'autres metadonnees
souvent presentes (titre de l'edition, ville, annee, graveur, technique, famille
iconographique). Le theme n'est qu'un champ parmi d'autres, pas un prerequis de
pertinence : juge chaque illustration a partir de TOUT ce qui est fourni pour
elle, jamais seulement de la presence ou de l'absence du champ theme — une
illustration sans theme mais avec un bon score de similarite et des metadonnees
cohérentes (meme famille iconographique, meme periode...) reste pertinente. Un
meme numero de famille iconographique entre deux illustrations signifie qu'elles
appartiennent au meme cycle de gravures (copie ou modele partage).

Tu n'as PAS vu les images : ne decris jamais un contenu visuel (scene,
personnage, composition, decor) que tu n'as pas recu par ecrit — ni en
affirmant qu'un personnage y figure, ni en affirmant qu'il en est absent.
Chaque illustration listee ci-dessous a SES PROPRES champs, distincts des
autres illustrations de la liste : ne reporte jamais un theme, un personnage
ou une metadonnee d'une illustration vers une autre de la meme liste, meme si
elles se ressemblent ou partagent le meme sujet de requete.

Attention : le Deluge biblique (Noe, l'arche) et le Deluge ovidien (Jupiter,
Deucalion, Pyrrha) sont deux recits differents — mais ne cite ces noms que
s'ils apparaissent litteralement dans les metadonnees ci-dessous, jamais de
memoire.

Reponds TOUJOURS en francais, quelle que soit la langue de la requete.

Consignes de reponse :
- Appuie-toi uniquement sur les informations fournies ci-dessous, tous champs
  confondus (pas seulement le theme).
- Utilise tous les champs informatifs disponibles dans ta reponse.
- Indique explicitement les numeros d'illustration qui appuient ta reponse.
- Recompte soigneusement avant de conclure — ne generalise pas sur une impression
  globale."""


def contexte_depuis_resultats(resultats):
    lignes = []
    for i, row in enumerate(resultats.itertuples(), 1):
        ligne = f"{i}. [{row.source}] similarite={row.similarite:.3f}"
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


#  Version courte volontairement : les premieres versions empilaient des
# consignes de prudence ("signale si aucune illustration ne correspond",
# "indique explicitement si ca correspond") qui poussaient llava:7b a ouvrir
# ses reponses par un rejet ("il n'y a aucune illustration qui correspond...")
# avant de decrire un contenu qui correspondait pourtant bien — la recherche
# vectorielle a deja fait le tri par similarite, le role du LLM ici est de
# decrire, pas de re-juger la pertinence.
# V5, retenue apres un test A/B/C/D/E sur plusieurs formulations (voir README) :
# - la regle de confiance metadonnee ("si le nom figure dans SA PROPRE
#   metadonnee, utilise-le sans hesiter ; sinon ne l'invente pas") corrige les
#   deux defauts complementaires observes avec la version precedente : les
#   identifications inventees sur des illustrations sans metadonnee (ex.
#   "probablement Diane" sans base) ET la sous-exploitation d'un theme_precis
#   pourtant fiable (le modele decrivait sans jamais nommer, meme quand le nom
#   etait fourni).
# - la consigne "chaque image a SA PROPRE metadonnee" reduit (sans l'eliminer)
#   la contamination inter-images : sur un lot de plusieurs illustrations dont
#   certaines n'ont aucun rapport avec le sujet de la requete, le modele tend a
#   leur attribuer le meme personnage que les autres images du lot. Teste sur
#   3 repetitions : nette amelioration sur une des deux images-pieges testees,
#   aucun effet mesurable sur l'autre — limite connue, voir README.
PROMPT_SYSTEME_VISION = """Tu es un expert en iconographie de la Renaissance europeenne (16e-17e s.).

Regarde reellement chaque image fournie et decris ce que tu vois (personnages,
actions, decor, composition). Appuie-toi aussi sur les metadonnees textuelles
fournies (theme si connu, titre, ville, graveur, technique) quand elles sont
disponibles, en complement de l'image.

Chaque image a SA PROPRE metadonnee, distincte des autres images de la reponse
et distincte du sujet de la requete de l'utilisateur : ne suppose jamais qu'un
nom de personnage (present dans la requete, ou releve sur une autre image de
la meme reponse) s'applique aussi a une image dont la metadonnee a elle ne le
mentionne pas.

Regle pour nommer un personnage mythologique ou biblique sur UNE image donnee :
si SA PROPRE metadonnee le nomme explicitement, utilise ce nom sans hesiter
(donnee de catalogue verifiee). Sinon, meme si ce nom apparait dans la requete
ou sur une autre image, decris la scene sans lui attribuer d'identite (ex. "un
personnage feminin armee d'un arc" plutot que d'affirmer "Diane").

Attention : le Deluge biblique (Noe, l'arche) et le Deluge ovidien (Jupiter,
Deucalion, Pyrrha) sont deux recits differents visuellement proches — verifie
sur l'image, ne devine pas a partir du titre seul.

Reponds TOUJOURS en francais, en 8-12 phrases maximum, en decrivant chaque
illustration avec son numero."""


# llava:7b refuse aleatoirement (~15-25% des appels, mesure empiriquement sur des
# gravures pourtant anodines) meme a temperature basse — pas un bug de notre cote,
# une instabilite du modele. Detection + retry automatique en filet de securite.
MARQUEURS_REFUS_VISION = [
    "desculpe", "não posso", "nao posso", "no puedo", "je ne peux pas",
    "i cannot", "i'm sorry", "unable to", "masquée", "masquee",
    "aucune image", "n'ai pas accès", "n'ai pas acces", "pas d'image fournie",
]


def _ressemble_a_un_refus(texte):
    t = texte.strip().lower()
    return len(t) < 60 or any(m in t[:250] for m in MARQUEURS_REFUS_VISION)


def generer_reponse_rag_vision(requete, resultats, max_essais=4):
    contexte = contexte_depuis_resultats(resultats)
    images_b64 = [image_vers_base64(chemin, taille_max=450) for chemin in resultats["chemin"]]
    # l'instruction visuelle doit venir AVANT les metadonnees techniques : un
    # message qui ouvre sur "recherche vectorielle / similarite cosinus" fait
    # deriver llava:7b en mode "je suis un modele de langage, pas d'accès aux
    # images" de facon quasi systematique (mesure : 4/4 refus) — juste reordonner
    # (regarder d'abord, metadonnees ensuite) fait remonter le taux de succes.
    message_utilisateur = f"""Regarde attentivement les {len(resultats)} images fournies, dans l'ordre, et
decris ce que tu vois sur chacune. Requete de l'utilisateur : {requete}

Metadonnees connues pour chaque illustration (a utiliser en complement de ce que
tu observes, pas a la place) :
{contexte}

Reponds en 8-12 phrases maximum, basees sur le contenu visuel que tu observes
reellement."""
    derniere_reponse = "[Ollama indisponible ou en erreur]"
    for essai in range(max_essais):
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODELE_VISION,
                "messages": [
                    {"role": "system", "content": PROMPT_SYSTEME_VISION},
                    {"role": "user", "content": message_utilisateur, "images": images_b64},
                ],
                "stream": False,
                # temperature basse : reduit (sans l'eliminer) la frequence des
                # refus aleatoires observee lors des tests
                "options": {"num_predict": 500, "temperature": 0.2},
            },
            timeout=180,
        )
        r.raise_for_status()
        derniere_reponse = r.json()["message"]["content"]
        if not _ressemble_a_un_refus(derniere_reponse):
            return derniere_reponse
    return derniere_reponse + (
        "\n\n_(Note : le modele vision a hesite/refuse plusieurs fois sur cette "
        "requete — reponse a prendre avec prudence.)_"
    )


def generer_reponse_rag(requete, resultats):
    contexte = contexte_depuis_resultats(resultats)
    message_utilisateur = f"""Requete : {requete}

Illustrations retrouvees par recherche vectorielle (SigLIP, similarite cosinus) :
{contexte}

Reponds en 8-12 phrases maximum."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODELE_TEXTE,
                "messages": [
                    {"role": "system", "content": PROMPT_SYSTEME},
                    {"role": "user", "content": message_utilisateur},
                ],
                "stream": False,
                # Ollama utilise le GPU (~265 tok/s mesure sur RTX 5090 apres passage a
                # l'install officielle, contre ~1,5 tok/s en CPU avec l'ancien snap
                # communautaire) — plus besoin de plafonner court.
                "options": {"num_predict": 500},
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        return (f"[Ollama indisponible ou en erreur : {e}]\n\n"
                f"Verifie que `ollama serve` tourne et que le modele `{MODELE_TEXTE}` "
                f"est telecharge (`ollama pull {MODELE_TEXTE}`).")


# --- 4. Vignettes en markdown (base64, pas de dependance a un serveur de fichiers) ---
def image_vers_base64(chemin, taille_max=200):
    img = Image.open(chemin).convert("RGB")
    img.thumbnail((taille_max, taille_max))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


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


def markdown_vignettes(resultats):
    cellules = []
    for i, row in enumerate(resultats.itertuples(), 1):
        b64 = image_vers_base64(row.chemin)
        legende = f"#{i} [{row.source}] {libelle_resultat(row)}"
        cellules.append(
            f"<div style='display:inline-block;text-align:center;margin:4px;'>"
            f"<img src='data:image/jpeg;base64,{b64}' width='140' "
            f"style='border-radius:4px;border:1px solid #ccc;'/><br>"
            f"<span style='font-size:0.8em;'>{legende}</span></div>"
        )
    return "".join(cellules)


# --- 5. Fonction de chat -----------------------------------------------------------
MODE_VISION = f"Vision — voit les images ({MODELE_VISION})"
MODE_TEXTE = f"Texte seul — metadonnees uniquement ({MODELE_TEXTE})"


def repondre(message, historique, mode_generation):
    texte = (message.get("text") or "").strip()
    fichiers = message.get("files") or []

    if fichiers:
        chemin_requete = fichiers[0]
        requete_desc = (
            f"une illustration envoyee par l'utilisateur ({Path(chemin_requete).name}), "
            f"theme inconnu a identifier a partir des voisins ci-dessous"
        )
        yield "Recherche des illustrations les plus proches dans la base..."
        # requete image : pas de texte a comparer a l'index metadonnees (e5),
        # recherche visuelle seule (SigLIP)
        resultats = rechercher(embed_image(chemin_requete), k=4)
    elif texte:
        requete_desc = f'"{texte}"'
        yield "Recherche des illustrations les plus proches dans la base..."
        # requete texte : recherche hybride (visuel SigLIP + metadonnees e5)
        resultats = rechercher_hybride(texte, k=4)
    else:
        yield "Tape une description (ex : *the creation of Eve*, plutot en anglais) ou envoie une image."
        return

    vignettes_md = markdown_vignettes(resultats)

    if mode_generation == MODE_TEXTE:
        yield f"{vignettes_md}\n\n_Redaction en cours ({MODELE_TEXTE}, sans les images)..._"
        reponse = generer_reponse_rag(requete_desc, resultats)
    else:
        yield f"{vignettes_md}\n\n_Analyse des illustrations en cours ({MODELE_VISION})..._"
        reponse = generer_reponse_rag_vision(requete_desc, resultats)

    yield f"{vignettes_md}\n\n---\n\n{reponse}"


demo = gr.ChatInterface(
    fn=repondre,
    multimodal=True,
    additional_inputs=[
        gr.Radio(
            [MODE_VISION, MODE_TEXTE],
            value=MODE_VISION,
            label="Modele de generation",
            info="Vision : voit reellement les images. Texte seul : ne recoit que les metadonnees (titre, theme, graveur...) — utile pour comparer les deux approches.",
        )
    ],
    additional_inputs_accordion=gr.Accordion(label="Options", open=False),
    title="Mini RAG — Iconographie Ovide / Bibles",
)

if __name__ == "__main__":
    # server_name="0.0.0.0" : accessible depuis d'autres machines du meme reseau
    # (utile si le lien public gradio.live est bloque par un pare-feu reseau).
    demo.queue().launch(share=True, server_name="0.0.0.0")
