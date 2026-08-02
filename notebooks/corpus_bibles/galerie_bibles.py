#!/usr/bin/env python3
"""
Galerie de tri des illustrations de Bibles — avec suppression réelle.

Lance un mini-serveur local. Dans le navigateur, pour chaque Bible :
  - cocher les illustrations à supprimer, puis « Supprimer la sélection »
  - ou « Tout supprimer cette Bible » si aucune illustration ne convient
La suppression agit sur les SOURCES (segmentees/), après une sauvegarde automatique.

Lancement :
    cd <dossier où est ce script>
    python galerie_bibles.py
Puis ouvrir l'adresse affichée (http://localhost:8000) dans un navigateur.
"""

import os
import shutil
from flask import Flask, send_file, request, jsonify, abort

# ─────────────────────────────────────────────────────────
# Configuration — adapter RACINE si besoin
# ─────────────────────────────────────────────────────────
RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOSSIER_BIBLES = os.path.join(RACINE, "data", "bibles_mdz", "segmentees")
BACKUP = os.path.join(RACINE, "data", "bibles_mdz", "segmentees_backup")
PORT = 8000

app = Flask(__name__)


def sauvegarde_initiale():
    """Copie de sécurité complète avant toute suppression (une seule fois)."""
    if not os.path.exists(BACKUP):
        print("Sauvegarde de sécurité en cours (une seule fois)…")
        shutil.copytree(DOSSIER_BIBLES, BACKUP)
        print(f"  ✓ Sauvegarde : {BACKUP}")
    else:
        print(f"Sauvegarde déjà présente : {BACKUP}")


def lister_bibles():
    """Renvoie {bsb_id: [noms de fichiers]} pour les Bibles non vides."""
    data = {}
    for bsb_id in sorted(os.listdir(DOSSIER_BIBLES)):
        d = os.path.join(DOSSIER_BIBLES, bsb_id)
        if not os.path.isdir(d):
            continue
        imgs = sorted([f for f in os.listdir(d)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))
                       and "_flip" not in f and not f.startswith("_tmp_")])
        if imgs:
            data[bsb_id] = imgs
    return data


@app.route("/")
def index():
    bibles = lister_bibles()
    total = sum(len(v) for v in bibles.values())
    blocs = []
    for bsb_id, imgs in bibles.items():
        cellules = "".join(
            f'<div class="cell" data-bsb="{bsb_id}" data-nom="{nom}" onclick="toggle(this)">'
            f'<img loading="lazy" src="/img/{bsb_id}/{nom}">'
            f'<span class="check">&#10007;</span></div>'
            for nom in imgs
        )
        blocs.append(
            f'<section class="bible" id="sec-{bsb_id}">'
            f'<div class="entete">'
            f'<h2>{bsb_id} <small>({len(imgs)} illustrations)</small></h2>'
            f'<div class="actions">'
            f'<button onclick="supprimerSelection(\'{bsb_id}\')">Supprimer la sélection</button>'
            f'<button class="danger" onclick="supprimerTout(\'{bsb_id}\')">Tout supprimer cette Bible</button>'
            f'</div></div>'
            f'<div class="grid">{cellules}</div>'
            f'</section>'
        )

    return f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>Galerie Bibles — tri et suppression</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 20px; background: #faf9f7; color: #2c2c2a; }}
  h1 {{ font-size: 22px; font-weight: 500; }}
  .info {{ font-size: 14px; color: #666; margin-bottom: 20px; }}
  .bible {{ margin-bottom: 32px; border-bottom: 2px solid #e5e3da; padding-bottom: 20px; }}
  .entete {{ position: sticky; top: 0; background: #faf9f7; padding: 10px 0; display: flex;
             justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; z-index: 5; }}
  h2 {{ font-size: 17px; font-weight: 500; margin: 0; }} h2 small {{ color: #888; font-weight: 400; }}
  .actions {{ display: flex; gap: 8px; }}
  button {{ font-size: 14px; padding: 7px 14px; border: 1px solid #888; background: #fff;
            border-radius: 8px; cursor: pointer; }}
  button:hover {{ background: #f0efe8; }}
  button.danger {{ border-color: #c0392b; color: #c0392b; }}
  button.danger:hover {{ background: #fdecea; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; margin-top: 12px; }}
  .cell {{ position: relative; border: 1px solid #ccc; border-radius: 8px; overflow: hidden;
           cursor: pointer; background: #fff; min-height: 110px; }}
  .cell img {{ width: 100%; display: block; }}
  .cell.sel {{ outline: 3px solid #c0392b; opacity: 0.55; }}
  .check {{ position: absolute; top: 4px; right: 4px; width: 24px; height: 24px; border-radius: 50%;
            background: #c0392b; color: #fff; display: none; align-items: center; justify-content: center; font-size: 14px; }}
  .cell.sel .check {{ display: flex; }}
</style></head><body>
<h1>Galerie des illustrations de Bibles</h1>
<p class="info">{total} illustrations · {len(bibles)} Bibles · clique une image pour la marquer (rouge),
puis « Supprimer la sélection ». La suppression est <b>définitive</b> (sauvegarde faite au démarrage).</p>
{"".join(blocs)}
<script>
function toggle(cell) {{ cell.classList.toggle("sel"); }}

async function supprimerSelection(bsb) {{
  const sel = [...document.querySelectorAll(`#sec-${{bsb}} .cell.sel`)];
  if (!sel.length) {{ alert("Aucune image sélectionnée pour " + bsb); return; }}
  if (!confirm(`Supprimer ${{sel.length}} illustration(s) de ${{bsb}} ? Action définitive.`)) return;
  const noms = sel.map(c => c.dataset.nom);
  const r = await fetch("/supprimer", {{
    method: "POST", headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{bsb: bsb, noms: noms}})
  }});
  const res = await r.json();
  if (res.ok) {{ sel.forEach(c => c.remove()); }}
  else {{ alert("Erreur : " + res.erreur); }}
}}

async function supprimerTout(bsb) {{
  if (!confirm(`Supprimer TOUTES les illustrations de ${{bsb}} ? Action définitive.`)) return;
  const r = await fetch("/supprimer_tout", {{
    method: "POST", headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify({{bsb: bsb}})
  }});
  const res = await r.json();
  if (res.ok) {{ document.getElementById("sec-" + bsb).remove(); }}
  else {{ alert("Erreur : " + res.erreur); }}
}}
</script></body></html>"""


@app.route("/img/<bsb>/<path:nom>")
def image(bsb, nom):
    chemin = os.path.join(DOSSIER_BIBLES, bsb, nom)
    if not os.path.isfile(chemin):
        abort(404)
    return send_file(chemin)


@app.route("/supprimer", methods=["POST"])
def supprimer():
    data = request.get_json()
    bsb, noms = data.get("bsb"), data.get("noms", [])
    try:
        n = 0
        for nom in noms:
            chemin = os.path.join(DOSSIER_BIBLES, bsb, nom)
            if os.path.isfile(chemin):
                os.remove(chemin)
                n += 1
        print(f"  {bsb} : {n} supprimées")
        return jsonify(ok=True, supprimes=n)
    except Exception as e:
        return jsonify(ok=False, erreur=str(e))


@app.route("/supprimer_tout", methods=["POST"])
def supprimer_tout():
    data = request.get_json()
    bsb = data.get("bsb")
    try:
        d = os.path.join(DOSSIER_BIBLES, bsb)
        n = 0
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                os.remove(os.path.join(d, f))
                n += 1
        print(f"  {bsb} : TOUT supprimé ({n} fichiers)")
        return jsonify(ok=True, supprimes=n)
    except Exception as e:
        return jsonify(ok=False, erreur=str(e))


if __name__ == "__main__":
    print("=" * 55)
    print("  Galerie de tri des Bibles")
    print("=" * 55)
    print("Dossier :", DOSSIER_BIBLES)
    sauvegarde_initiale()
    print(f"\nOuvre dans ton navigateur : http://localhost:{PORT}")
    print("Ctrl+C pour arrêter le serveur.\n")
    app.run(port=PORT, debug=False)
