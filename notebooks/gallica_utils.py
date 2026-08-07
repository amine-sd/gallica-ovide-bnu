"""
utils.py — Fonctions partagées pour le projet Gallica Images / Ovide
=====================================================================
Projet  : Analyse illustrations Métamorphoses d'Ovide
Auteur  : Stage Bnu Strasbourg, avril 2026
Usage   : from utils import charger_yolo, segmenter_page, ...
"""

import os
import sys
import gc
import random
import shutil
import requests
import time

import torch
import torchvision.transforms as T
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import http.server
import threading
import webbrowser

from PIL import Image
from io import BytesIO
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────
# CONFIGURATION GLOBALE
# ─────────────────────────────────────────────

BASE_URL    = "https://galimages-search.bnf.fr"
ARK_SALOMON = "btv1b2200047r"
CLASSES     = ["bois", "cuivre"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM_PRED = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

TRANSFORM_VAL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────
# YOLO — Chargement et segmentation
# ─────────────────────────────────────────────

def charger_yolo(yolov5_repo="yolov5_repo"):
    import cv2
    cv2.imshow = lambda *args, **kwargs: None

    for mod in list(sys.modules.keys()):
        if "ultralytics" in mod or "yolov5" in mod:
            del sys.modules[mod]

    # Retirer le dossier notebooks/ du path pour éviter le conflit avec utils.py
    notebooks_path = os.path.dirname(os.path.abspath(__file__))
    if notebooks_path in sys.path:
        sys.path.remove(notebooks_path)

    # Ajouter yolov5_repo en premier
    yolov5_abs = os.path.abspath(yolov5_repo)
    if yolov5_abs in sys.path:
        sys.path.remove(yolov5_abs)
    sys.path.insert(0, yolov5_abs)

    chemin_modele = hf_hub_download(
        repo_id="seglinglin/Historical-Illustration-Extraction",
        filename="illustration_extraction.pt"
    )
    ckpt   = torch.load(chemin_modele, map_location="cpu", weights_only=False)
    modele = ckpt["model"].float().eval()

    # Remettre le dossier notebooks/ dans le path
    if notebooks_path not in sys.path:
        sys.path.append(notebooks_path)

    print(f"✓ YOLO chargé — classes : {modele.names}")
    return modele

def segmenter_page(chemin_image, prefixe, dossier_sortie,
                   modele_yolo, conf_thres=0.25):
    """
    Détecte et extrait les illustrations dans une page avec YOLO.
    Sauvegarde chaque illustration dans dossier_sortie.
    Retourne le nombre d'illustrations extraites.

    Paramètres
    ----------
    chemin_image  : str   — chemin vers la page JPG source
    prefixe       : str   — préfixe pour nommer les fichiers produits
    dossier_sortie: str   — dossier où sauvegarder les illustrations
    modele_yolo   : model — modèle YOLO chargé via charger_yolo()
    conf_thres    : float — seuil de confiance (défaut 0.25)
    """
    from utils.general import non_max_suppression

    os.makedirs(dossier_sortie, exist_ok=True)
    try:
        img_page     = Image.open(chemin_image).convert("RGB")
        img_w, img_h = img_page.size

        transform  = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        img_tensor = transform(img_page).unsqueeze(0)

        with torch.no_grad():
            res = modele_yolo(img_tensor)

        preds = non_max_suppression(res, conf_thres=conf_thres, iou_thres=0.45)

        nb = 0
        for j, det in enumerate(preds[0]):
            x1, y1, x2, y2, conf, _ = det.tolist()
            x1 = int(x1 * img_w / 640)
            x2 = int(x2 * img_w / 640)
            y1 = int(y1 * img_h / 640)
            y2 = int(y2 * img_h / 640)

            illustration = img_page.crop((x1, y1, x2, y2))
            nom = f"{dossier_sortie}/{prefixe}_det{j+1}_conf{round(conf, 2)}.jpg"
            illustration.save(nom)
            nb += 1

        return nb

    except Exception as e:
        print(f"  Erreur segmentation {prefixe} : {e}")
        return 0


def liberer_yolo(modele_yolo):
    """
    Libère la mémoire GPU occupée par le modèle YOLO.
    À appeler avant de charger ResNet50.

    """
    del modele_yolo
    torch.cuda.empty_cache()
    gc.collect()
    print("✓ Mémoire GPU libérée")
    return None


# ─────────────────────────────────────────────
# IIIF — Téléchargement de pages
# ─────────────────────────────────────────────

def telecharger_pages_iiif(manifest_url, dossier_sortie,
                            prefixe="page", pause=0.05):
    """
    Télécharge toutes les pages d'un livre depuis son manifest IIIF.
    Saute les pages déjà téléchargées (reprise possible).
    Retourne la liste des chemins locaux.

    Paramètres
    ----------
    manifest_url  : str   — URL du manifest IIIF (JSON)
    dossier_sortie: str   — dossier de destination
    prefixe       : str   — préfixe pour nommer les fichiers (défaut "page")
    pause         : float — délai entre requêtes en secondes (défaut 0.05)


    """
    os.makedirs(dossier_sortie, exist_ok=True)

    r        = requests.get(manifest_url, timeout=30)
    manifest = r.json()
    canvases = manifest["sequences"][0]["canvases"]
    print(f"Pages trouvées : {len(canvases)} — {manifest.get('label', '')[:60]}")

    chemins = []
    for i, canvas in enumerate(canvases):
        print(f"  {i+1}/{len(canvases)}...", end="\r")
        url    = canvas["images"][0]["resource"]["@id"]
        chemin = f"{dossier_sortie}/{prefixe}_{i+1:03d}.jpg"

        if os.path.exists(chemin):
            chemins.append(chemin)
            continue

        try:
            r   = requests.get(url, timeout=15)
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.save(chemin)
            chemins.append(chemin)
            time.sleep(pause)
        except Exception as e:
            print(f"\n  Erreur page {i+1} : {e}")

    print(f"\n✓ {len(chemins)} pages sauvegardées dans {dossier_sortie}")
    return chemins


def segmenter_corpus(pages_brutes, dossier_sortie,
                     modele_yolo, conf_thres=0.25):
    """
    Applique segmenter_page sur une liste de pages brutes.
    Retourne le nombre total d'illustrations extraites.

    Paramètres
    ----------
    pages_brutes  : list  — liste de chemins JPG
    dossier_sortie: str   — dossier de destination
    modele_yolo   : model — modèle YOLO chargé
    conf_thres    : float — seuil de confiance (défaut 0.25)
    """
    os.makedirs(dossier_sortie, exist_ok=True)
    total = 0

    for i, chemin in enumerate(pages_brutes):
        print(f"  {i+1}/{len(pages_brutes)}...", end="\r")
        prefixe = os.path.splitext(os.path.basename(chemin))[0]
        nb      = segmenter_page(chemin, prefixe, dossier_sortie,
                                 modele_yolo, conf_thres)
        total  += nb

    print(f"\n✓ {total} illustrations extraites dans {dossier_sortie}")
    return total


# ─────────────────────────────────────────────
# DATASET — Split et organisation
# ─────────────────────────────────────────────

def rassembler_illustrations(dossier_seg, dossier_dataset):
    os.makedirs(f"{dossier_dataset}/bois",   exist_ok=True)
    os.makedirs(f"{dossier_dataset}/cuivre", exist_ok=True)

    for dossier in os.listdir(dossier_seg):
        chemin = f"{dossier_seg}/{dossier}"
        if not os.path.isdir(chemin):
            continue

        if dossier.startswith("bois"):
            classe = "bois"
        elif dossier.startswith("cuivre") and dossier != "cuivre_pdf":
            classe = "cuivre"
        elif dossier == "cuivre_pdf":
            for edition in os.listdir(chemin):
                sous = f"{chemin}/{edition}"
                if os.path.isdir(sous):
                    for f in os.listdir(sous):
                        if f.endswith(".jpg"):
                            shutil.copy(f"{sous}/{f}",
                                        f"{dossier_dataset}/cuivre/{edition}_{f}")
            continue
        else:
            continue

        for f in os.listdir(chemin):
            if f.endswith(".jpg"):
                shutil.copy(f"{chemin}/{f}",
                            f"{dossier_dataset}/{classe}/{dossier}_{f}")

def split_dataset(dossier_dataset, classe,
                  ratio_train=0.7, ratio_val=0.15):
    """
    Divise les images d'une classe en train/val/test.
    Crée les dossiers dataset/train|val|test/classe/.

    Paramètres
    ----------
    dossier_dataset : str   — chemin vers dataset/
    classe          : str   — "bois" ou "cuivre"
    ratio_train     : float — proportion train (défaut 0.70)
    ratio_val       : float — proportion val   (défaut 0.15)
    """
    for split in ["train", "val", "test"]:
        os.makedirs(f"{dossier_dataset}/{split}/{classe}", exist_ok=True)

    images = [f for f in os.listdir(f"{dossier_dataset}/{classe}")
              if f.endswith(".jpg")]
    random.shuffle(images)

    n       = len(images)
    n_train = int(n * ratio_train)
    n_val   = int(n * ratio_val)

    splits = {
        "train": images[:n_train],
        "val"  : images[n_train:n_train + n_val],
        "test" : images[n_train + n_val:]
    }

    for split, fichiers in splits.items():
        for f in fichiers:
            shutil.copy(f"{dossier_dataset}/{classe}/{f}",
                        f"{dossier_dataset}/{split}/{classe}/{f}")
        print(f"  {split}/{classe} : {len(fichiers)} images")


def stats_illustrations(dossier_seg):
    """
    Affiche un récapitulatif du nombre d'illustrations par source.

    Paramètres
    ----------
    dossier_seg : str — chemin vers illustrations_segmentees/
    """
    print("Illustrations segmentées :\n")
    nb_bois = nb_cuivre = 0

    print("── BOIS ──")
    for d in sorted(os.listdir(dossier_seg)):
        chemin = f"{dossier_seg}/{d}"
        if os.path.isdir(chemin) and "bois" in d:
            n = len([f for f in os.listdir(chemin) if f.endswith(".jpg")])
            print(f"  {d:50s} : {n}")
            nb_bois += n

    print("\n── CUIVRE ──")
    for d in sorted(os.listdir(dossier_seg)):
        chemin = f"{dossier_seg}/{d}"
        if os.path.isdir(chemin) and "cuivre" in d:
            if d == "cuivre_pdf":
                for edition in sorted(os.listdir(chemin)):
                    sous = f"{chemin}/{edition}"
                    if os.path.isdir(sous):
                        n = len([f for f in os.listdir(sous) if f.endswith(".jpg")])
                        print(f"  {edition[:50]:50s} : {n}")
                        nb_cuivre += n
            else:
                n = len([f for f in os.listdir(chemin) if f.endswith(".jpg")])
                print(f"  {d:50s} : {n}")
                nb_cuivre += n

    print(f"\n{'─'*60}")
    print(f"  Total bois   : {nb_bois}")
    print(f"  Total cuivre : {nb_cuivre}")
    print(f"  TOTAL        : {nb_bois + nb_cuivre}")


# ─────────────────────────────────────────────
# RESNET50 — Chargement et prédiction
# ─────────────────────────────────────────────

def charger_resnet(chemin_pth, device=None):
    """
    Charge un modèle ResNet50 fine-tuné bois/cuivre depuis un fichier .pth.
    Retourne le modèle en mode eval, prêt pour la prédiction.

    Paramètres
    ----------
    chemin_pth : str            — chemin vers le fichier .pth
    device     : torch.device   — CPU ou CUDA (auto-détecté si None)

    """
    if device is None:
        device = DEVICE

    modele = models.resnet50(weights=None)
    modele.fc = nn.Linear(2048, 2)
    modele.load_state_dict(torch.load(chemin_pth, map_location=device))
    modele = modele.to(device)
    modele.eval()
    print(f"✓ ResNet50 chargé : {chemin_pth}")
    return modele


def predire_technique(url, modele, device=None):
    """
    Prédit la technique de gravure (bois ou cuivre) d'une illustration
    accessible via une URL Gallica.
    Retourne (classe, confiance).

    Paramètres
    ----------
    url    : str   — URL de l'illustration Gallica
    modele : model — ResNet50 chargé via charger_resnet()
    device : torch.device — CPU ou CUDA (auto-détecté si None)


    """
    if device is None:
        device = DEVICE
    try:
        r          = requests.get(url, timeout=10)
        img        = Image.open(BytesIO(r.content)).convert("RGB")
        img_tensor = TRANSFORM_PRED(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = modele(img_tensor)
            probs   = torch.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)
        return CLASSES[pred.item()], round(probs[0][pred.item()].item(), 3)
    except Exception:
        return "inconnu", 0.0


def appliquer_modele_df(df, modele, col_classe, col_conf, device=None):
    """
    Applique predire_technique sur toutes les lignes d'un dataframe
    et ajoute deux colonnes : col_classe et col_conf.

    Paramètres
    ----------
    df        : DataFrame — doit contenir une colonne "link"
    modele    : model     — ResNet50 chargé
    col_classe: str       — nom de la colonne pour la classe (ex: "technique_ia_v3")
    col_conf  : str       — nom de la colonne pour la confiance
    device    : torch.device


    """
    if device is None:
        device = DEVICE

    classes, confs = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"  {i+1}/{len(df)}...", end="\r")
        classe, conf = predire_technique(row["link"], modele, device)
        classes.append(classe)
        confs.append(conf)

    df[col_classe] = classes
    df[col_conf]   = confs
    print(f"\n✓ {col_classe} — {df[col_classe].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────
# TABLEAU HTML
# ─────────────────────────────────────────────
    

def generer_tableau_html(df, nom_fichier, colonnes_ia=None, port=8085):
    """
    Génère un tableau HTML filtrable depuis un dataframe de résultats
    de similarité et ouvre un serveur local pour le visualiser.

    Paramètres
    ----------
    df            : DataFrame — résultats de similarité
    nom_fichier   : str       — nom du fichier HTML produit
    colonnes_ia   : list      — liste de versions IA à inclure,
                                ex: ["v1", "v2", "v3"] ou None pour aucune
    port          : int       — port du serveur local (défaut 8085)

    """
    if colonnes_ia is None:
        colonnes_ia = []

    subtitle = (f"{len(df)} illustrations similaires trouvées dans Gallica "
                f"pour {df['salomon_page'].nunique()} gravures de Salomon")

    # ── Filtres dynamiques IA ───────────────────────────────
    filtres_ia_html = ""
    for v in colonnes_ia:
        filtres_ia_html += f"""
        <div class="filtre-groupe">
            <label>Technique IA {v}</label>
            <select id="f_technique_ia_{v}" onchange="filtrer()">
                <option value="">Toutes</option>
                <option value="bois">bois</option>
                <option value="cuivre">cuivre</option>
            </select>
        </div>"""

    # ── En-têtes IA ─────────────────────────────────────────
    headers_ia = ""
    for v in colonnes_ia:
        headers_ia += f"<th>Technique IA {v}</th><th>Conf. {v}</th>"

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Résultats similarité — Bernard Salomon 1557</title>
    <style>
        body {{ font-family: Georgia, serif; background: #f8f5f0; margin: 0; padding: 20px; }}
        h1 {{ text-align: center; color: #2c2c2c; font-size: 1.4em; margin-bottom: 5px; }}
        .subtitle {{ text-align: center; color: #666; font-size: 0.85em; margin-bottom: 20px; }}
        .filtres {{ background: white; border-radius: 8px; padding: 15px;
                   margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                   display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; }}
        .filtre-groupe {{ display: flex; flex-direction: column; gap: 4px; }}
        .filtre-groupe label {{ font-size: 0.8em; color: #666; }}
        .filtre-groupe select, .filtre-groupe input {{
            padding: 6px 8px; border: 1px solid #ccc; border-radius: 4px;
            font-size: 0.85em; font-family: Georgia, serif; }}
        .btn-reset {{ padding: 6px 14px; background: #2c2c2c; color: white;
                     border: none; border-radius: 4px; cursor: pointer;
                     font-size: 0.85em; font-family: Georgia, serif; }}
        .btn-reset:hover {{ background: #555; }}
        .compteur {{ font-size: 0.85em; color: #888; align-self: center; }}
        table {{ width: 100%; border-collapse: collapse; background: white;
                border-radius: 8px; overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        th {{ background: #2c2c2c; color: white; padding: 10px 12px;
             text-align: left; font-size: 0.85em; font-weight: normal; }}
        td {{ padding: 8px 12px; font-size: 0.8em; border-bottom: 1px solid #eee;
             vertical-align: middle; }}
        tr:hover td {{ background: #faf8f5; }}
        td img {{ width: 60px; height: 75px; object-fit: cover;
                 border: 1px solid #ddd; border-radius: 3px; }}
        .score-high {{ color: #2e7d32; font-weight: bold; }}
        .score-mid  {{ color: #f57c00; font-weight: bold; }}
        .tag-bois    {{ background: #e8f5e9; color: #2e7d32; padding: 2px 8px;
                       border-radius: 10px; font-size: 0.8em; }}
        .tag-cuivre  {{ background: #fff3e0; color: #e65100; padding: 2px 8px;
                       border-radius: 10px; font-size: 0.8em; }}
        .tag-inconnu {{ background: #f5f5f5; color: #888; padding: 2px 8px;
                       border-radius: 10px; font-size: 0.8em; }}
        a {{ color: #2c2c2c; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .hidden {{ display: none; }}
    </style>
</head>
<body>
    <h1>Résultats de similarité — Illustrations Bernard Salomon (1557)</h1>
    <p class="subtitle">{subtitle}<br>
        Filtrez par page, technique, genre, score ou technique IA
    </p>
    <div class="filtres">
        <div class="filtre-groupe">
            <label>Page Salomon</label>
            <input type="number" id="f_page" placeholder="ex: 5" min="1" max="181" oninput="filtrer()">
        </div>
        <div class="filtre-groupe">
            <label>Technique</label>
            <select id="f_technique" onchange="filtrer()">
                <option value="">Toutes</option>
                <option value="estampe">estampe</option>
                <option value="photographie">photographie</option>
                <option value="dessin">dessin</option>
            </select>
        </div>
        <div class="filtre-groupe">
            <label>Mode chromatique</label>
            <select id="f_chroma" onchange="filtrer()">
                <option value="">Tous</option>
                <option value="nb">nb</option>
                <option value="monochrome">monochrome</option>
                <option value="couleur">couleur</option>
            </select>
        </div>
        <div class="filtre-groupe">
            <label>Score minimum</label>
            <input type="number" id="f_score" placeholder="ex: 0.90" min="0" max="1"
                   step="0.01" oninput="filtrer()">
        </div>
        <div class="filtre-groupe">
            <label>Genre</label>
            <input type="text" id="f_genre" placeholder="ex: Scènes" oninput="filtrer()">
        </div>
        {filtres_ia_html}
        <button class="btn-reset" onclick="reset()">Réinitialiser</button>
        <span class="compteur" id="compteur"></span>
    </div>
    <table id="tableau">
        <thead>
            <tr>
                <th>Image</th><th>Page Salomon</th><th>Score</th>
                <th>Titre</th><th>Auteur</th><th>Date</th>
                <th>Technique</th><th>Genre</th><th>Palette</th>
                <th>Mode</th><th>Corpus</th>
                {headers_ia}
            </tr>
        </thead>
        <tbody id="tbody">
"""

    def safe(val, maxlen=None):
        s = str(val) if pd.notna(val) else ""
        return s[:maxlen] if maxlen else s

    for _, row in df.iterrows():
        score = row["score"]
        sc    = "score-high" if score >= 0.93 else "score-mid"
        link  = safe(row["link"])
        titre = safe(row["titre"], 60) + ("..." if len(safe(row["titre"])) > 60 else "")

        # data-attributes AI dynamiques
        data_ia = ""
        for v in colonnes_ia:
            col = f"technique_ia_{v}"
            val = row.get(col, "inconnu")
            data_ia += f' data-technique_ia_{v}="{safe(val)}"'

        cells_ia = ""
        for v in colonnes_ia:
            col_c = f"technique_ia_{v}"
            col_f = f"technique_ia_conf_{v}"
            tech  = safe(row.get(col_c, "inconnu"))
            conf  = safe(row.get(col_f, ""))
            tag   = f"tag-{tech}" if tech in ["bois", "cuivre"] else "tag-inconnu"
            cells_ia += f'<td><span class="{tag}">{tech}</span></td><td>{conf}</td>'

        html += f"""
        <tr data-page="{row['salomon_page']}" data-technique="{safe(row['technique'])}"
            data-chroma="{safe(row['chromatic_mode'])}" data-score="{score}"
            data-genre="{safe(row['genre']).lower()}"{data_ia}>
            <td><a href="{link}" target="_blank">
                <img src="{link}" alt="illustration" onerror="this.style.display='none'">
            </a></td>
            <td>{row['salomon_page']}</td>
            <td class="{sc}">{score}</td>
            <td><a href="{link}" target="_blank">{titre}</a></td>
            <td>{safe(row['auteur'], 40)}</td>
            <td>{safe(row['date'], 4)}</td>
            <td>{safe(row['technique'])}</td>
            <td>{safe(row['genre'])}</td>
            <td>{safe(row['palette'])}</td>
            <td>{safe(row['chromatic_mode'])}</td>
            <td>{safe(row['corpus'], 50)}</td>
            {cells_ia}
        </tr>"""

    # ── JavaScript filtrage ──────────────────────────────────
    js_vars = "\n".join(
        [f'var technique_ia_{v} = document.getElementById("f_technique_ia_{v}").value;'
         for v in colonnes_ia]
    )
    js_checks = "\n".join(
        [f'if (technique_ia_{v} && row.dataset.technique_ia_{v} !== technique_ia_{v}) ok = false;'
         for v in colonnes_ia]
    )
    js_reset = "\n".join(
        [f'document.getElementById("f_technique_ia_{v}").value = "";'
         for v in colonnes_ia]
    )

    html += f"""
        </tbody>
    </table>
    <script>
    function filtrer() {{
        var page      = document.getElementById("f_page").value.trim();
        var technique = document.getElementById("f_technique").value;
        var chroma    = document.getElementById("f_chroma").value;
        var scoreMin  = parseFloat(document.getElementById("f_score").value) || 0;
        var genre     = document.getElementById("f_genre").value.toLowerCase().trim();
        {js_vars}
        var rows    = document.querySelectorAll("#tbody tr");
        var visible = 0;
        rows.forEach(function(row) {{
            var ok = true;
            if (page      && row.dataset.page      !== page)              ok = false;
            if (technique && row.dataset.technique !== technique)          ok = false;
            if (chroma    && row.dataset.chroma    !== chroma)             ok = false;
            if (scoreMin  && parseFloat(row.dataset.score) < scoreMin)    ok = false;
            if (genre     && !row.dataset.genre.includes(genre))          ok = false;
            {js_checks}
            row.classList.toggle("hidden", !ok);
            if (ok) visible++;
        }});
        document.getElementById("compteur").textContent =
            visible + " résultat" + (visible > 1 ? "s" : "") +
            " affiché" + (visible > 1 ? "s" : "");
    }}
    function reset() {{
        document.getElementById("f_page").value      = "";
        document.getElementById("f_technique").value = "";
        document.getElementById("f_chroma").value    = "";
        document.getElementById("f_score").value     = "";
        document.getElementById("f_genre").value     = "";
        {js_reset}
        filtrer();
    }}
    filtrer();
    </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(nom_fichier) or ".", exist_ok=True)
    with open(nom_fichier, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✓ Tableau généré : {nom_fichier}")

    def _serveur():
        dossier = os.path.dirname(os.path.abspath(nom_fichier)) or "."
        os.chdir(dossier)
        handler = http.server.SimpleHTTPRequestHandler
        handler.log_message = lambda *args: None
        with http.server.HTTPServer(("", port), handler) as httpd:
            httpd.serve_forever()

    t = threading.Thread(target=_serveur, daemon=True)
    t.start()
    url = f"http://localhost:{port}/{os.path.basename(nom_fichier)}"
    print(f"✓ URL : {url}")
    webbrowser.open(url)
