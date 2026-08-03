#!/usr/bin/env python3
"""
Atelier de regroupement des illustrations de Bibles — pour Céline (v3).

Source de vérité = le dossier  data/bibles_mdz/classes_celine/  :
chaque sous-dossier = une classe (nom = nom du dossier), avec ses images.

Au démarrage :
  - si classes_celine/ existe et n'est pas vide -> on REPREND ce découpage
  - sinon -> clustering initial depuis regroupement/ pour proposer un départ

Fonctions :
  - "Regrouper le reste" : K-Means (n classes) sur les images NON validées
  - Valider une classe : la fige pour la session (le re-clustering ne la touche plus)
  - déplacer une image (icône + menu), supprimer une image
  - fusionner une classe entière dans une autre (menu, comme pour déplacer une image),
    ou la supprimer définitivement — deux boutons distincts, pas de confusion possible
  - voir en grand + navigation (flèches / clavier), numéro de position sur chaque vignette
  - supprimer une image depuis la vue agrandie (icône ou touche Suppr), sans repasser
    par les vignettes — reste en grand sur l'image suivante
  - sélection multiple (case sur chaque vignette) + suppression en lot (bouton ou
    touche Suppr) — n'importe où dans la grille, toutes classes confondues
  - menu "Classement" : change l'ordre d'affichage des classes (numéro, nom, taille,
    validées/non validées d'abord) — affichage seulement, ne change rien aux données
  - "Enregistrer" : réécrit classes_celine/ avec l'organisation courante
        => c'est À LA FOIS la sauvegarde et le point de reprise de la prochaine session

Les images viennent toujours de regroupement/ (intact) ; classes_celine/ ne stocke que
des copies rangées.

Lancement :  python atelier_regroupement.py   puis  http://localhost:8050
"""

import os, shutil
import numpy as np
from flask import Flask, request, jsonify, send_file, abort

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC    = os.path.join(RACINE, "data", "bibles_mdz", "regroupement")      # images de référence (intact)
SORTIE = os.path.join(RACINE, "data", "bibles_mdz", "classes_celine")    # source de vérité (lue + réécrite)
PORT   = 8050

app = Flask(__name__)
ETAT = {"images": [], "X": None, "id2idx": {}}


def preparer():
    """Liste les images de regroupement/ et calcule les embeddings CLIP (une fois)."""
    import torch, open_clip
    from PIL import Image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)

    images = []
    for bsb in sorted(os.listdir(SRC)):
        d = os.path.join(SRC, bsb)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and "_flip" not in f and not f.startswith("_tmp_"):
                images.append({"id": f"{bsb}/{f}", "bsb": bsb, "nom": f,
                               "chemin": os.path.join(d, f)})
    print(f"{len(images)} illustrations dans regroupement/")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.to(device).eval()

    print("Calcul des embeddings CLIP…")
    vecs = []
    for k, img in enumerate(images, 1):
        print(f"  {k}/{len(images)}", end="\r")
        im = Image.open(img["chemin"]).convert("RGB").convert("L").convert("RGB")
        x = preprocess(im).unsqueeze(0).to(device)
        with torch.no_grad():
            vecs.append(model.encode_image(x).cpu().numpy()[0])
    print("\nEmbeddings prêts.")
    ETAT["images"] = images
    ETAT["X"] = np.array(vecs)
    ETAT["id2idx"] = {im["id"]: i for i, im in enumerate(images)}


def lire_classes_celine():
    """Lit classes_celine/ et renvoie l'organisation reprise, ou None si vide.
    Renvoie une liste d'images {id, classe (int), valide=False} + noms {idx: nom}.
    Le rattachement se fait par NOM DE FICHIER (basename), car classes_celine
    contient des copies."""
    if not os.path.isdir(SORTIE):
        return None
    sous = [d for d in sorted(os.listdir(SORTIE)) if os.path.isdir(os.path.join(SORTIE, d))]
    if not sous:
        return None

    # index basename -> id complet (depuis regroupement)
    base2id = {}
    for im in ETAT["images"]:
        base2id.setdefault(os.path.basename(im["chemin"]), im["id"])

    images_etat, noms = [], {}
    validees = []
    deja = set()
    for ci, nom_dossier in enumerate(sous):
        noms[ci] = nom_dossier
        dchem = os.path.join(SORTIE, nom_dossier)
        if os.path.exists(os.path.join(dchem, ".valide")):
            validees.append(ci)
        for f in sorted(os.listdir(dchem)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                ident = base2id.get(f)
                if ident and ident not in deja:
                    images_etat.append({"id": ident, "classe": ci, "valide": False})
                    deja.add(ident)

    # images de regroupement/ absentes de classes_celine -> non classées (classe null)
    for im in ETAT["images"]:
        if im["id"] not in deja:
            images_etat.append({"id": im["id"], "classe": None, "valide": False})

    return {"images": images_etat, "noms": noms, "prochaine": len(sous), "validees": validees}


def clusteriser_reste(n, ids_reste):
    from sklearn.cluster import KMeans
    indices = [ETAT["id2idx"][i] for i in ids_reste if i in ETAT["id2idx"]]
    if not indices:
        return {}
    n = max(1, min(n, len(indices)))
    labels = KMeans(n_clusters=n, random_state=42, n_init=10).fit_predict(ETAT["X"][indices])
    return {ids_reste[k]: int(labels[k]) for k in range(len(ids_reste))}


@app.route("/")
def index():
    return PAGE


@app.route("/img/<path:ident>")
def img(ident):
    for im in ETAT["images"]:
        if im["id"] == ident:
            return send_file(im["chemin"])
    abort(404)


@app.route("/init")
def init():
    repris = lire_classes_celine()
    tous = [{"id": im["id"], "bsb": im["bsb"]} for im in ETAT["images"]]
    return jsonify(images=tous, repris=repris)


@app.route("/clusteriser", methods=["POST"])
def route_clusteriser():
    data = request.get_json()
    n = int(data.get("n", 12))
    mapping = clusteriser_reste(n, data.get("ids", []))
    return jsonify(ok=True, mapping=mapping)


@app.route("/enregistrer", methods=["POST"])
def enregistrer():
    """Réécrit classes_celine/ avec l'organisation courante (= sauvegarde + reprise).
    Préserve le marqueur .valide des classes validées listées."""
    data = request.get_json()
    orga = data.get("organisation", {})
    validees = set(data.get("validees", []))   # noms de classes (dossiers) validées
    try:
        if os.path.exists(SORTIE):
            shutil.rmtree(SORTIE)
        os.makedirs(SORTIE, exist_ok=True)
        chemin_par_id = {im["id"]: im["chemin"] for im in ETAT["images"]}
        total = 0
        for nom_classe, ids in orga.items():
            safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in nom_classe).strip() or "sans_nom"
            d = os.path.join(SORTIE, safe)
            os.makedirs(d, exist_ok=True)
            if nom_classe in validees:
                open(os.path.join(d, ".valide"), "w").close()   # marqueur
            for ident in ids:
                src = chemin_par_id.get(ident)
                if src and os.path.isfile(src):
                    shutil.copy2(src, os.path.join(d, os.path.basename(src)))
                    total += 1
        return jsonify(ok=True, total=total, dossiers=len(orga))
    except Exception as e:
        return jsonify(ok=False, erreur=str(e))


PAGE = r"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>Atelier de regroupement — Bibles</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; padding: 16px; background: #faf9f7; color: #2c2c2a; }
  h1 { font-size: 20px; font-weight: 500; }
  .barre { position: sticky; top: 0; background: #faf9f7; padding: 10px 0; display: flex; gap: 10px;
           align-items: center; flex-wrap: wrap; border-bottom: 1px solid #ddd; margin-bottom: 14px; z-index: 30; }
  input[type=number] { width: 60px; font-size: 15px; padding: 5px; }
  input[type=text] { font-size: 14px; padding: 4px 6px; border: 1px solid #bbb; border-radius: 6px; }
  button { font-size: 14px; padding: 7px 12px; border: 1px solid #888; background: #fff; border-radius: 8px; cursor: pointer; }
  button:hover { background: #f0efe8; }
  button.primary { border-color: #1d6e56; color: #0f6e56; font-weight: 500; }
  button.danger { border-color: #c0392b; color: #c0392b; }
  .classe { border: 1px solid #ddd; border-radius: 10px; margin-bottom: 18px; background: #fff; }
  .classe.valide { border-color: #1d9e75; box-shadow: 0 0 0 2px #d4f0e5 inset; }
  .classe-tete { display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-bottom: 1px solid #eee;
                 position: sticky; top: 56px; background: #fff; z-index: 10; border-radius: 10px 10px 0 0; flex-wrap: wrap; }
  .classe.valide .classe-tete { background: #f0faf6; }
  .classe-tete .n { font-size: 13px; color: #888; }
  .badge-v { font-size: 12px; color: #0f6e56; border: 1px solid #1d9e75; border-radius: 10px; padding: 1px 8px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; padding: 12px; }
  .cell { position: relative; border: 1px solid #ccc; border-radius: 6px; overflow: hidden; background: #fff; min-height: 90px; }
  .cell img { width: 100%; display: block; }
  .cell .num { position: absolute; bottom: 2px; right: 4px; background: rgba(0,0,0,0.6); color: #fff;
               font-size: 11px; padding: 0 5px; border-radius: 8px; }
  .cell .oeil, .cell .move, .cell .x { position: absolute; width: 22px; height: 22px; border-radius: 50%;
               color: #fff; border: none; font-size: 12px; cursor: pointer; display: block;
               opacity: 0.55; transition: opacity .1s; }
  .cell .oeil { top: 2px; left: 2px; background: #185fa5; font-size: 11px; }
  .cell .move { bottom: 2px; left: 2px; background: #0f6e56; }
  .cell .x    { top: 2px; right: 2px; background: #c0392b; }
  .cell:hover .oeil, .cell:hover .move, .cell:hover .x { opacity: 1; }
  .cell .sel { position: absolute; top: 2px; left: 50%; transform: translateX(-50%);
               width: 17px; height: 17px; cursor: pointer; z-index: 5; }
  .cell.selectionnee { outline: 3px solid #185fa5; outline-offset: -3px; }
  .selection-bar { display: none; align-items: center; gap: 8px; }
  select#tri { font-size: 14px; padding: 6px 8px; border: 1px solid #bbb; border-radius: 6px; }
  .hint { font-size: 13px; color: #777; margin: 4px 0 12px; }
  #msg { font-size: 14px; color: #0f6e56; }
  .loupe-fond { position: fixed; inset: 0; background: rgba(0,0,0,0.8); display: none;
                align-items: center; justify-content: center; z-index: 100; }
  .loupe-fond img { max-width: 84%; max-height: 92%; }
  .loupe-fleche { position: absolute; top: 50%; transform: translateY(-50%); width: 52px; height: 52px;
                  border-radius: 50%; background: rgba(255,255,255,0.9); border: none; font-size: 26px; cursor: pointer; }
  #fg { left: 20px; } #fd { right: 20px; }
  .loupe-fermer { position: absolute; top: 16px; right: 20px; width: 40px; height: 40px; border-radius: 50%;
                  background: rgba(255,255,255,0.9); border: none; font-size: 20px; cursor: pointer; }
  .loupe-suppr { position: absolute; top: 16px; left: 20px; width: 40px; height: 40px; border-radius: 50%;
                 background: rgba(192,57,43,0.9); color: #fff; border: none; font-size: 18px; cursor: pointer; }
  .loupe-cpt { position: absolute; bottom: 18px; left: 50%; transform: translateX(-50%); color: #fff;
               font-size: 14px; background: rgba(0,0,0,0.5); padding: 4px 12px; border-radius: 12px; }
  .menu-fond { position: fixed; inset: 0; background: rgba(0,0,0,0.4); display: none;
               align-items: center; justify-content: center; z-index: 90; }
  .menu { background: #fff; border-radius: 10px; padding: 16px; max-width: 360px; max-height: 70vh; overflow-y: auto; }
  .menu h3 { margin: 0 0 10px; font-size: 15px; font-weight: 500; }
  .menu button { display: block; width: 100%; text-align: left; margin-bottom: 6px; }
</style></head><body>
<h1>Atelier de regroupement des illustrations</h1>
<div class="barre">
  <label>Classes (pour le reste) : <input type="number" id="n" value="12" min="1" max="60"></label>
  <button onclick="regrouper()">Regrouper le reste</button>
  <span style="border-left:1px solid #ccc;height:24px"></span>
  <label>Classement : <select id="tri" onchange="render()">
    <option value="numero">Ordre (numero)</option>
    <option value="nom">Nom (A→Z)</option>
    <option value="taille_desc">Nombre d'images (plus → moins)</option>
    <option value="taille_asc">Nombre d'images (moins → plus)</option>
    <option value="non_validees">Non validees d'abord</option>
    <option value="validees">Validees d'abord</option>
  </select></label>
  <span style="border-left:1px solid #ccc;height:24px"></span>
  <button class="primary" onclick="enregistrer()">Enregistrer</button>
  <span id="msg"></span>
  <span class="selection-bar" id="selection-bar">
    <span id="sel-cpt"></span>
    <button class="danger" onclick="supprimerSelection()">Supprimer la selection</button>
    <button onclick="viderSelection()">Annuler la selection</button>
  </span>
</div>
<p class="hint">L'atelier reprend ce qui est dans classes_celine/. Survole une image : oeil = voir,
fleches = deplacer, croix = supprimer. En grand (oeil), la poubelle (ou la touche Suppr)
supprime l'image sans refermer la vue agrandie — elle passe directement a la suivante.
La case au-dessus de chaque vignette selectionne l'image pour une suppression en lot
(bouton "Supprimer la selection" dans la barre, ou touche Suppr). Le menu Classement
change juste l'ordre d'affichage des classes, il ne change rien aux donnees.
Sur une classe : "Fusionner..." deplace toutes ses images dans une autre classe (menu au
choix) ; "Supprimer la classe" supprime definitivement ses images — deux boutons distincts.
Valider fige une classe (le regroupement ne la touche plus).
Enregistrer reecrit classes_celine/ : c'est ta sauvegarde et le point de reprise.</p>
<div id="classes"></div>

<div class="loupe-fond" id="loupe">
  <button class="loupe-suppr" onclick="supprimerDepuisLoupe()" title="Supprimer cette image">&#128465;</button>
  <button class="loupe-fermer" onclick="fermerLoupe()">&times;</button>
  <button class="loupe-fleche" id="fg" onclick="naviguer(-1)">&#8249;</button>
  <img src="">
  <button class="loupe-fleche" id="fd" onclick="naviguer(1)">&#8250;</button>
  <div class="loupe-cpt" id="loupe-cpt"></div>
</div>
<div class="menu-fond" id="menu-fond" onclick="if(event.target===this)this.style.display='none'">
  <div class="menu" id="menu"></div>
</div>

<script>
let images = [];          // {id, classe (int|null), valide}
let nomsClasses = {};
let classesValidees = new Set();
let prochaineClasse = 0;
let selection = new Set();   // ids selectionnes pour suppression en lot

async function init() {
  const r = await fetch("/init");
  const d = await r.json();
  if (d.repris) {
    images = d.repris.images;
    nomsClasses = d.repris.noms || {};
    prochaineClasse = d.repris.prochaine || 0;
    classesValidees = new Set(d.repris.validees || []);
    render();
    document.getElementById("msg").textContent = "Repris depuis classes_celine/";
    // s'il reste des images non classées, on peut les regrouper
    if (images.some(im => im.classe === null)) {
      document.getElementById("msg").textContent += " — des images non classées : clique « Regrouper le reste »";
    }
  } else {
    images = d.images.map(im => ({id: im.id, classe: null, valide: false}));
    await regrouper();
  }
}

function idsNonValides() {
  // Une image est "à re-clusteriser" si sa classe n'est PAS validée
  return images.filter(im => !classesValidees.has(im.classe)).map(im => im.id);
}

async function regrouper() {
  const n = parseInt(document.getElementById("n").value) || 12;
  const ids = idsNonValides();
  if (!ids.length) { alert("Toutes les classes sont validees, rien a regrouper."); return; }
  document.getElementById("msg").textContent = "Calcul...";
  const r = await fetch("/clusteriser", {method:"POST", headers:{"Content-Type":"application/json"},
                        body: JSON.stringify({n, ids})});
  const res = await r.json();
  const base = (classesValidees.size ? Math.max(...classesValidees) : -1) + 1;
  images.forEach(im => {
    if (!classesValidees.has(im.classe) && res.mapping[im.id] !== undefined) im.classe = base + res.mapping[im.id];
  });
  prochaineClasse = Math.max(prochaineClasse, base + n);
  render();
  document.getElementById("msg").textContent = n + " classes proposees (hors validees)";
}

function classesActuelles() {
  const m = {};
  images.forEach(im => { if (im.classe !== null) (m[im.classe] = m[im.classe] || []).push(im); });
  return m;
}

function ordreClasses(m) {
  const cls = Object.keys(m);
  const tri = document.getElementById("tri") ? document.getElementById("tri").value : "numero";
  const taille = cl => m[cl].length;
  const nom = cl => (nomsClasses[cl] || ("classe " + cl)).toLowerCase();
  const estValidee = cl => classesValidees.has(parseInt(cl)) ? 1 : 0;
  if (tri === "nom") cls.sort((a, b) => nom(a).localeCompare(nom(b)));
  else if (tri === "taille_desc") cls.sort((a, b) => taille(b) - taille(a));
  else if (tri === "taille_asc") cls.sort((a, b) => taille(a) - taille(b));
  else if (tri === "non_validees") cls.sort((a, b) => estValidee(a) - estValidee(b));
  else if (tri === "validees") cls.sort((a, b) => estValidee(b) - estValidee(a));
  else cls.sort((a, b) => a - b);   // "numero" (defaut) : ordre d'apparition des classes
  return cls;
}

function render() {
  const cont = document.getElementById("classes");
  cont.innerHTML = "";
  const m = classesActuelles();
  ordreClasses(m).forEach(cl => {
    const valide = classesValidees.has(parseInt(cl));
    const div = document.createElement("div");
    div.className = "classe" + (valide ? " valide" : "");
    const nomVal = nomsClasses[cl] || ("classe " + cl);
    div.innerHTML = '<div class="classe-tete">'
      + '<input type="text" value="' + nomVal + '" onchange="nomsClasses[' + cl + ']=this.value">'
      + '<span class="n">' + m[cl].length + ' images</span>'
      + (valide ? '<span class="badge-v">validee</span><button onclick="devalider(' + cl + ')">Devalider</button>'
                : '<button class="primary" onclick="valider(' + cl + ')">Valider</button>'
                  + '<button onclick="fusionnerClasse(' + cl + ')">Fusionner...</button>'
                  + '<button class="danger" onclick="supprimerClasse(' + cl + ')">Supprimer la classe</button>')
      + '</div><div class="grid"></div>';
    const grid = div.querySelector(".grid");
    m[cl].forEach((im, i) => {
      const c = document.createElement("div");
      c.className = "cell" + (selection.has(im.id) ? " selectionnee" : "");
      c.innerHTML = '<img loading="lazy" src="/img/' + encodeURIComponent(im.id) + '">'
        + '<input type="checkbox" class="sel" title="Selectionner" onclick="toggleSelection(\'' + im.id + '\')"'
          + (selection.has(im.id) ? ' checked' : '') + '>'
        + '<button class="oeil" onclick="voir(\'' + im.id + '\')">&#128065;</button>'
        + (valide ? '' : '<button class="move" onclick="ouvrirMenu(\'' + im.id + '\')">&#8596;</button>'
                       + '<button class="x" onclick="supprimerImage(\'' + im.id + '\')">&times;</button>')
        + '<span class="num">' + (i+1) + '</span>';
      grid.appendChild(c);
    });
    cont.appendChild(div);
  });
  majBarreSelection();
}

function toggleSelection(id) {
  if (selection.has(id)) selection.delete(id); else selection.add(id);
  render();
}

function majBarreSelection() {
  const bar = document.getElementById("selection-bar");
  if (selection.size) {
    bar.style.display = "inline-flex";
    document.getElementById("sel-cpt").textContent = selection.size + " selectionnee(s)";
  } else {
    bar.style.display = "none";
  }
}

function supprimerSelection() {
  if (!selection.size) return;
  if (!confirm("Supprimer " + selection.size + " illustration(s) selectionnee(s) ?")) return;
  images = images.filter(im => !selection.has(im.id));
  selection.clear();
  render();
}

function viderSelection() { selection.clear(); render(); }

async function valider(cl) {
  classesValidees.add(parseInt(cl));
  render();
  await enregistrer();   // grave aussitôt sur le disque (avec marqueur .valide)
  document.getElementById("msg").textContent = "Classe validee et enregistree";
}
function devalider(cl) { classesValidees.delete(parseInt(cl)); render(); }

function supprimerImage(id) { images = images.filter(im => im.id !== id); selection.delete(id); render(); }

function supprimerClasse(cl) {
  const n = images.filter(im => im.classe == cl).length;
  if (!confirm("Supprimer definitivement cette classe et ses " + n + " illustrations ? (utilise \"Fusionner...\" pour les deplacer au lieu de les supprimer)")) return;
  images = images.filter(im => im.classe != cl);
  render();
}

function fusionnerClasse(cl) {
  const m = classesActuelles();
  const cibles = Object.keys(m).filter(c => c != cl);
  if (!cibles.length) { alert("Aucune autre classe disponible pour fusionner."); return; }
  let html = "<h3>Fusionner cette classe dans...</h3>";
  cibles.sort((a, b) => a - b).forEach(c => {
    const tag = classesValidees.has(parseInt(c)) ? " (validee)" : "";
    html += '<button onclick="fusionnerVers(' + cl + ', ' + c + ')">'
          + (nomsClasses[c] || ("classe " + c)) + tag + ' <span style="color:#888">(' + m[c].length + ')</span></button>';
  });
  html += '<button onclick="document.getElementById(\'menu-fond\').style.display=\'none\'" style="margin-top:8px;color:#888">Annuler</button>';
  document.getElementById("menu").innerHTML = html;
  document.getElementById("menu-fond").style.display = "flex";
}

function fusionnerVers(clSource, clCible) {
  clCible = parseInt(clCible);
  images.forEach(im => { if (im.classe == clSource) im.classe = clCible; });
  document.getElementById("menu-fond").style.display = "none";
  render();
}

function ouvrirMenu(id) {
  const im = images.find(x => x.id === id);
  if (!im) return;
  const m = classesActuelles();
  let html = "<h3>Deplacer vers...</h3>";
  Object.keys(m).sort((a,b)=>a-b).forEach(cl => {
    if (parseInt(cl) === im.classe) return;
    const tag = classesValidees.has(parseInt(cl)) ? " (validee)" : "";
    html += '<button onclick="deplacerVers(\'' + id + '\', ' + cl + ')">'
          + (nomsClasses[cl]||("classe "+cl)) + tag + ' <span style="color:#888">(' + m[cl].length + ')</span></button>';
  });
  html += '<button onclick="deplacerVers(\'' + id + '\', ' + prochaineClasse + ')" style="border-color:#0f6e56;color:#0f6e56">+ nouvelle classe</button>';
  html += '<button onclick="document.getElementById(\'menu-fond\').style.display=\'none\'" style="margin-top:8px;color:#888">Annuler</button>';
  document.getElementById("menu").innerHTML = html;
  document.getElementById("menu-fond").style.display = "flex";
}

function deplacerVers(id, cl) {
  const im = images.find(x => x.id === id);
  cl = parseInt(cl);
  if (im) im.classe = cl;
  if (cl >= prochaineClasse) prochaineClasse = cl + 1;
  document.getElementById("menu-fond").style.display = "none";
  render();
}

let loupeListe = [], loupeIdx = 0;
function voir(id) {
  const im = images.find(x => x.id === id);
  if (!im) return;
  loupeListe = images.filter(x => x.classe === im.classe).map(x => x.id);
  loupeIdx = loupeListe.indexOf(id);
  afficherLoupe();
  document.getElementById("loupe").style.display = "flex";
}
function afficherLoupe() {
  document.querySelector("#loupe img").src = "/img/" + encodeURIComponent(loupeListe[loupeIdx]);
  document.getElementById("loupe-cpt").textContent = (loupeIdx+1) + " / " + loupeListe.length;
}
function naviguer(s) { loupeIdx = (loupeIdx + s + loupeListe.length) % loupeListe.length; afficherLoupe(); }
function fermerLoupe() { document.getElementById("loupe").style.display = "none"; }

function supprimerDepuisLoupe() {
  // Supprime l'image affichee en grand sans revenir a la grille : reste dans la
  // loupe sur l'image suivante de la meme classe, ou ferme si c'etait la derniere.
  const id = loupeListe[loupeIdx];
  if (id === undefined) return;
  images = images.filter(im => im.id !== id);
  selection.delete(id);
  loupeListe = loupeListe.filter(x => x !== id);
  render();
  if (!loupeListe.length) { fermerLoupe(); return; }
  loupeIdx = loupeIdx % loupeListe.length;
  afficherLoupe();
}

document.addEventListener("keydown", e => {
  if (document.getElementById("loupe").style.display === "flex") {
    if (e.key === "ArrowLeft") naviguer(-1);
    if (e.key === "ArrowRight") naviguer(1);
    if (e.key === "Escape") fermerLoupe();
    if (e.key === "Delete") supprimerDepuisLoupe();
    return;
  }
  // en dehors de la loupe : Suppr agit sur la selection en lot (pas si on tape dans un champ)
  if (e.key === "Delete" && selection.size && document.activeElement.tagName !== "INPUT") {
    supprimerSelection();
  }
});

async function enregistrer() {
  const m = classesActuelles();
  const orga = {};
  const validees = [];
  Object.keys(m).forEach(cl => {
    const nom = nomsClasses[cl] || ("classe_"+cl);
    orga[nom] = m[cl].map(im=>im.id);
    if (classesValidees.has(parseInt(cl))) validees.push(nom);
  });
  document.getElementById("msg").textContent = "Enregistrement...";
  const r = await fetch("/enregistrer", {method:"POST", headers:{"Content-Type":"application/json"},
                        body: JSON.stringify({organisation: orga, validees})});
  const res = await r.json();
  document.getElementById("msg").textContent = res.ok ?
    ("Enregistre : " + res.total + " images dans " + res.dossiers + " classes (classes_celine/)") :
    ("Erreur : " + res.erreur);
}

init();
</script></body></html>"""


if __name__ == "__main__":
    print("=" * 55)
    print("  Atelier de regroupement des Bibles (v3)")
    print("=" * 55)
    print("Images  :", SRC)
    print("Classes :", SORTIE)
    preparer()
    print(f"\nOuvre dans ton navigateur : http://localhost:{PORT}")
    print("Ctrl+C pour arreter.\n")
    app.run(port=PORT, debug=False)
