"""Interface de chat (Gradio) pour tester le mini RAG iconographie Ovide / Bibles.

Reprend l'index + les fonctions de rag_utils.py, exposees dans une fenetre de
chat façon ChatGPT : texte ou image en entree, recherche vectorielle hybride
(SigLIP + e5 + filtre exact), puis reponse redigee par un LLM local (Ollama).
Un menu "Options" permet de choisir le modele de generation : vision
(qwen2.5vl, voit reellement les images — par defaut) ou texte seul (llama3.2,
ne recoit que les metadonnees — utile pour comparer les deux approches).

Lancement :
    ./.venv/bin/python notebooks/mini_rag_iconographique/app_rag.py

Necessite Ollama demarre (`ollama serve`) avec les deux modeles deja telecharges
(`ollama pull qwen2.5vl`, `ollama pull llama3.2`). Avec share=True, un lien public
temporaire (72h) est imprime dans le terminal (gradio.live) — c'est ce lien qu'il
faut envoyer au tuteur.
"""

import base64
from io import BytesIO
from pathlib import Path

import gradio as gr
import requests
import torch
from PIL import Image

from rag_utils import (
    charger_e5,
    charger_index,
    charger_siglip,
    contexte_depuis_resultats,
    embed_image,
    libelle_resultat,
    rechercher,
    rechercher_hybride,
)

RACINE = Path(__file__).resolve().parent.parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", DEVICE)

# --- 1. Index unifie (voir rag_utils.charger_index) ------------------------------
DOSSIER_VECTOR_DB = RACINE / "data" / "vector_bases"
index, X_index, X_index_meta = charger_index(DOSSIER_VECTOR_DB)
print(f"Index charge : {len(index)} illustrations "
      f"({(index['source'] == 'bible').sum()} Bibles, {(index['source'] == 'ovide').sum()} Ovide)")

# --- 2. Modeles d'encodage (SigLIP + e5, voir rag_utils.py) -----------------------
processor, model_siglip = charger_siglip(DEVICE)
print("SigLIP charge")
modele_e5 = charger_e5()
print("Modele e5 (metadonnees) charge")


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

Ne parle jamais de score de similarite, de confiance ou de seuil technique
dans ta reponse — ce sont des details internes de la recherche, pas des
informations a communiquer a l'utilisateur.

Reponds TOUJOURS en francais, quelle que soit la langue de la requete.

Consignes de reponse :
- Appuie-toi uniquement sur les informations fournies ci-dessous, tous champs
  confondus (pas seulement le theme).
- Utilise tous les champs informatifs disponibles dans ta reponse.
- Indique explicitement les numeros d'illustration qui appuient ta reponse.
- Recompte soigneusement avant de conclure — ne generalise pas sur une impression
  globale."""


# Version courte volontairement : empiler des consignes de prudence poussait
# llava:7b a ouvrir ses reponses par un rejet reflexe avant de decrire un
# contenu qui correspondait pourtant bien — la recherche vectorielle a deja
# fait le tri par similarite, le role du LLM ici est de decrire, pas de
# re-juger la pertinence.
#
# V5, retenue apres un test A/B/C/D/E sur plusieurs formulations : la regle de
# confiance metadonnee ("si le nom figure dans SA PROPRE metadonnee, utilise-le
# sans hesiter ; sinon ne l'invente pas") corrige a la fois les identifications
# inventees sans base et la sous-exploitation d'un theme_precis pourtant
# fiable ; la consigne "chaque image a SA PROPRE metadonnee" reduit (sans
# l'eliminer) la contamination inter-images. Historique complet des 5
# formulations testees et de leurs resultats : voir docs/mini_rag_iconographique.md.
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

Ne parle jamais de score de similarite, de confiance ou de seuil technique
dans ta reponse — ce sont des details internes de la recherche, pas des
informations a communiquer a l'utilisateur.

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
        vecteur = embed_image(chemin_requete, processor, model_siglip, DEVICE)
        resultats = rechercher(vecteur, index, X_index, k=4)
    elif texte:
        requete_desc = f'"{texte}"'
        yield "Recherche des illustrations les plus proches dans la base..."
        # requete texte : recherche hybride (visuel SigLIP + metadonnees e5)
        resultats = rechercher_hybride(
            texte, index, X_index, X_index_meta,
            processor, model_siglip, modele_e5, DEVICE, k=4,
        )
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
    # server_name="0.0.0.0" : accessible depuis d'autres machines du meme reseau.
    # Pas de share=True : le lien public gradio.live est bloque par le pare-feu
    # reseau ici (timeout a chaque lancement, pour rien) - usage local uniquement.
    demo.queue().launch(server_name="0.0.0.0")
