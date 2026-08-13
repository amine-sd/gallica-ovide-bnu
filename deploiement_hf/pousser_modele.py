#!/usr/bin/env python3
"""
Publie le classifieur bois/cuivre sur le Hugging Face Hub.

Convertit les state_dict PyTorch (.pth) en dépôt Hub standard : poids au format
safetensors + config.json + carte de modèle.

Les deux versions retenues à l'issue du stage sont publiées dans le même dépôt :
  - v4.1.1 sur la branche `main`  — meilleure précision, version par défaut
  - v4.0.0 sur la branche `v4.0.0` — meilleur rappel, pour un pré-tri relu par un humain

Prérequis :
    ./.venv/bin/hf auth login        (token avec droit "write")

Lancement, depuis la racine du dépôt :
    ./.venv/bin/python deploiement_hf/pousser_modele.py
"""

import os
import sys

import torch
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modele_hf import ClassifieurBoisCuivre, LABELS

# ──────────────────────────────────────────────────────
COMPTE = "Spakalao"                    # identifiant Hugging Face
NOM    = "resnet50-gravure-bois-cuivre"
PRIVE  = True                          # dépôt privé — basculer en public depuis les
                                       # réglages du dépôt une fois la relecture faite

# version -> branche cible (None = branche par défaut `main`)
VERSIONS = {
    "v4.1.1": None,
    "v4.0.0": "v4.0.0",
}
# ──────────────────────────────────────────────────────

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHEMIN_CARTE = os.path.join(RACINE, "deploiement_hf", "carte_modele.md")
REPO_ID = f"{COMPTE}/{NOM}"


def chemin_poids(version):
    return os.path.join(RACINE, "modeles", "bois_cuivre",
                        f"resnet50_bois_cuivre_{version}.pth")


def charger(version):
    """Reconstruit le modèle et y charge les poids entraînés.

    Les clés du state_dict torchvision (conv1, layer1..4, fc) correspondent à celles
    de l'attribut `.net` — d'où le load_state_dict sur `.net` plutôt que sur le modèle.
    """
    modele = ClassifieurBoisCuivre(num_classes=len(LABELS))
    modele.net.load_state_dict(torch.load(chemin_poids(version), map_location="cpu"))
    modele.eval()

    # garde-fou : une passe avant doit sortir un logit par classe
    with torch.no_grad():
        sortie = modele(torch.zeros(1, 3, 224, 224))
    assert sortie.shape == (1, len(LABELS)), f"sortie inattendue : {sortie.shape}"

    return modele


def main():
    if COMPTE == "TON_COMPTE":
        sys.exit("Renseigne d'abord COMPTE en haut de ce fichier.")

    for version in VERSIONS:
        if not os.path.exists(chemin_poids(version)):
            sys.exit(f"Poids introuvables : {chemin_poids(version)}")

    api = HfApi()

    for version, branche in VERSIONS.items():
        modele = charger(version)
        print(f"✓ {version} chargé et vérifié")

        # une branche doit exister avant qu'on puisse y committer
        if branche is not None:
            api.create_branch(repo_id=REPO_ID, branch=branche, exist_ok=True)

        modele.push_to_hub(
            REPO_ID,
            private=PRIVE,
            branch=branche,
            commit_message=f"Classifieur bois/cuivre {version}",
        )
        print(f"  → publié sur « {branche or 'main'} »")

    # carte de modèle : sur la branche par défaut uniquement
    with open(CHEMIN_CARTE, encoding="utf-8") as f:
        carte = f.read().replace("REPO_ID", REPO_ID)

    api.upload_file(
        path_or_fileobj=carte.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Carte de modèle",
    )
    print("✓ Carte de modèle publiée")
    print(f"\n→ https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
