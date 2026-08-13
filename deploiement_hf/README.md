# Publication du classifieur bois/cuivre sur Hugging Face

Fichiers ayant servi à publier le modèle ResNet50 bois/cuivre sur le Hub, et permettant de
le republier ou de le mettre à jour.

**Dépôt publié** : [`Spakalao/resnet50-gravure-bois-cuivre`](https://huggingface.co/Spakalao/resnet50-gravure-bois-cuivre)

| Fichier | Rôle |
|---|---|
| `modele_hf.py` | Classe `ClassifieurBoisCuivre` — enveloppe le ResNet50 avec `PyTorchModelHubMixin` pour que `from_pretrained()` fonctionne |
| `pousser_modele.py` | Convertit les `.pth` en safetensors et publie poids + config + carte de modèle |
| `carte_modele.md` | Carte de modèle (devient le `README.md` du dépôt Hub) |

## Marche à suivre

**1. S'authentifier** — token avec droit *write* depuis
<https://huggingface.co/settings/tokens>, dans un terminal WSL (le `.venv` est un
environnement Linux) :

```bash
./.venv/bin/hf auth login
```

**2. Publier** (depuis la racine du dépôt) :

```bash
./.venv/bin/python deploiement_hf/pousser_modele.py
```

Le script publie **les deux versions retenues** dans le même dépôt — v4.1.1 sur `main`
(chargée par défaut), v4.0.0 sur une branche `v4.0.0` — puis remplace la carte de modèle
générée automatiquement par `carte_modele.md`. Pour chacune, il recharge les poids et
vérifie la passe avant avant d'envoyer quoi que ce soit.

## Notes

- Les deux versions retenues cohabitent dans un dépôt unique via le mécanisme de branches
  du Hub : `from_pretrained(REPO_ID)` charge v4.1.1, et
  `from_pretrained(REPO_ID, revision="v4.0.0")` la variante orientée rappel. Le
  dictionnaire `VERSIONS` en haut de `pousser_modele.py` pilote cette correspondance.
- Toute personne rechargeant le modèle depuis le Hub a besoin de la définition de classe de
  `modele_hf.py` — elle est reproduite dans la carte de modèle pour cette raison.
- `PRIVE = True` dans `pousser_modele.py` : le dépôt est privé. Le passage en public se fait
  depuis les réglages du dépôt sur le Hub, pas depuis ce script.
- **Pas de démo en ligne** : les Spaces Gradio requièrent un abonnement PRO (seuls les
  Spaces statiques sont gratuits). Le code de démonstration a été retiré.
