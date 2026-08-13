"""
Définition du classifieur bois/cuivre au format Hugging Face Hub.

Cette classe enveloppe le ResNet50 fine-tuné du stage pour lui ajouter
`push_to_hub()` / `from_pretrained()` via PyTorchModelHubMixin. Les poids d'origine
(`modeles/bois_cuivre/*.pth`) sont des state_dict torchvision : ils se chargent dans
l'attribut `.net`, dont les noms de couches sont identiques (conv1, layer1..4, fc).

Ce fichier est importé par `pousser_modele.py` (publication). Toute personne qui veut
recharger le modèle depuis le Hub a besoin de cette définition de classe — elle est
reproduite dans la carte de modèle pour cette raison.
"""

import torch.nn as nn
from torchvision import models
from huggingface_hub import PyTorchModelHubMixin

# Ordre imposé par l'entraînement (cf. CLASSES dans notebooks/gallica_utils.py)
LABELS = ["bois", "cuivre"]


class ClassifieurBoisCuivre(
    nn.Module,
    PyTorchModelHubMixin,
    tags=[
        "image-classification",
        "resnet",
        "digital-humanities",
        "livre-ancien",
        "histoire-du-livre",
        "gravure",
        "illustration-ovidienne",
    ],
    license="cc-by-4.0",
):
    """ResNet50 binaire : gravure sur bois (0) vs gravure sur cuivre (1)."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = models.resnet50(weights=None)
        self.net.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.net(x)
