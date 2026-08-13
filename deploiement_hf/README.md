# Publication du classifieur bois/cuivre sur Hugging Face

Fichiers ayant servi à publier le modèle ResNet50 bois/cuivre sur le Hub, et permettant de
le republier ou de le mettre à jour.

**Dépôt publié** : [`Spakalao/resnet50-gravure-bois-cuivre`](https://huggingface.co/Spakalao/resnet50-gravure-bois-cuivre)

| Fichier | Rôle |
|---|---|
| `modele_hf.py` | Classe `ClassifieurBoisCuivre` — enveloppe le ResNet50 avec `PyTorchModelHubMixin` pour que `from_pretrained()` fonctionne |
| `pousser_modele.py` | Convertit les `.pth` en safetensors et publie poids + config + carte de modèle |
| `carte_modele.md` | Carte de modèle (devient le `README.md` du dépôt Hub) |

