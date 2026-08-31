---
license: mit
library_name: pytorch
pipeline_tag: image-classification
language:
  - fr
tags:
  - image-classification
  - resnet
  - digital-humanities
  - livre-ancien
  - histoire-du-livre
  - gravure
  - illustration-ovidienne
---

# Classifieur bois / cuivre — gravures de livres anciens

Prédit si une illustration de livre ancien est une **gravure sur bois** ou une **gravure sur
cuivre**, à partir de l'image seule, sans métadonnées ni texte. Éditions imprimées entre le
16<sup>e</sup> et le 18<sup>e</sup> siècle.

Modèle produit dans le cadre du projet *« Ovide en images à la BNU : de l'imprimé ancien à
la donnée »*, Bibliothèque nationale et universitaire de Strasbourg.

## Utilisation

```python
import torch, torch.nn as nn
from torchvision import models, transforms
from huggingface_hub import PyTorchModelHubMixin
from PIL import Image

LABELS = ["bois", "cuivre"]

class ClassifieurBoisCuivre(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = models.resnet50(weights=None)
        self.net.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.net(x)

modele = ClassifieurBoisCuivre.from_pretrained("REPO_ID").eval()

transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image = Image.open("une_illustration_decoupee.jpg").convert("RGB")
with torch.no_grad():
    probabilites = torch.softmax(modele(transformation(image).unsqueeze(0)), dim=1)[0]

print(LABELS[probabilites.argmax()], round(probabilites.max().item(), 3))
```

- **Entrée** : image RGB redimensionnée en 224×224, normalisation ImageNet
- **Sortie** : 2 logits — indice `0 = bois`, `1 = cuivre`

## Versions

**Deux versions sont publiées**, à égalité sur le jeu de test interne : elles y classent
correctement 98,3 % des illustrations, mais se trompent dans des directions opposées. La
v4.1.1, chargée par défaut, est équilibrée entre les deux techniques : 98,4 % de rappel sur
le bois, 98,3 % sur le cuivre. La v4.0.0 pousse le rappel sur le cuivre à 99,4 %, mais
retombe à 95,1 % sur le bois.

```python
# variante orientée rappel
modele = ClassifieurBoisCuivre.from_pretrained("REPO_ID", revision="v4.0.0")
```

## Limites connues

- **Le corpus « bois » ne repose que sur 4 sources visuellement indépendantes** (l'une des
  cinq éditions disponibles étant une copie en miroir d'une autre). La capacité du modèle à
  généraliser à de la gravure sur bois totalement extérieure à ce corpus **reste à
  démontrer** — c'est la limite la plus importante de ce modèle.

- Aucune évaluation n'a été conduite sur des techniques de gravure autres que le bois
  et le cuivre : le modèle attribuera nécessairement l'une des deux classes, même face à une
  lithographie ou une photographie.

## Citation et contexte

Modèle développé lors d'un stage d'ingénieur (ENSICAEN, parcours ISIA) à la Bibliothèque
nationale et universitaire de Strasbourg, dans le cadre de la résidence de recherche de
Céline Bohnert (CRIMEL, Université de Reims Champagne-Ardenne).
