---
license: cc-by-4.0
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

ResNet50 fine-tuné distinguant la **gravure sur bois** de la **gravure sur
cuivre**, sur des illustrations d'éditions imprimées anciennes
(16<sup>e</sup>–18<sup>e</sup> siècle).

Modèle produit dans le cadre du projet *« Ovide en images à la BNU : de l'imprimé ancien à
la donnée »*, Bibliothèque nationale et universitaire de Strasbourg.

## ⚠️ À lire avant d'utiliser le modèle

**Ce modèle attend une illustration déjà découpée, pas une page de livre.** Dans les éditions
anciennes, une gravure n'occupe souvent qu'une fraction de la page, le reste étant du texte,
des marges et des ornements. Lui envoyer une page entière revient à lui faire classer une
mise en page plutôt qu'un motif gravé, et dégrade les résultats.

Une étape de segmentation préalable est donc indispensable. Le projet utilise
[`seglinglin/Historical-Illustration-Extraction`](https://huggingface.co/seglinglin/Historical-Illustration-Extraction)
(YOLOv5, seuil de confiance 0,25) pour isoler les illustrations avant classification.

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

## Performances

Évaluation **face aux annotations manuelles d'une spécialiste du livre ancien**, sur un
échantillon indépendant des données d'entraînement — et non sur le seul jeu de test interne
(voir « Méthodologie » ci-dessous pour comprendre pourquoi cette distinction est décisive).

| Version | Précision (cuivre) | Rappel (cuivre) | F1 | Rappel (bois) |
|---|---|---|---|---|
| **v4.1.1** — branche `main` | **84,7 %** | 94,4 % | 0,893 | **98,4 %** |
| v4.0.0 — branche `v4.0.0` | 81,3 % | **99,4 %** | 0,894 | 95,1 % |

**Les deux versions sont publiées**, car elles correspondent à deux compromis également
défendables et non à une meilleure et une moins bonne — leurs F1 sont indiscernables.

- **v4.1.1** (chargée par défaut) se trompe rarement quand elle annonce un cuivre, et tient
  mieux la classe *bois*, la plus fragile du corpus. À privilégier pour un enrichissement
  automatique de catalogue, et comme choix par défaut si l'usage n'est pas fixé.
- **v4.0.0** ne rate presque aucun cuivre, au prix de davantage de fausses alertes. À
  privilégier pour un **pré-tri destiné à être relu par un humain**, où manquer une planche
  coûte plus cher qu'en signaler une à tort.

```python
# variante orientée rappel
modele = ClassifieurBoisCuivre.from_pretrained("REPO_ID", revision="v4.0.0")
```

Sur le jeu de test interne (partage par édition) : 98,3 % pour les deux versions.

## Méthodologie — pourquoi le protocole d'évaluation compte ici

Une version antérieure du modèle atteignait un score interne excellent qui s'est révélé
trompeur : le partage apprentissage/validation/test se faisait par tirage aléatoire
d'**images individuelles**. Des illustrations d'un même ouvrage pouvaient donc se retrouver
simultanément en apprentissage et en test, et le modèle apprenait à reconnaître le grain du
papier ou le style de numérisation d'une édition plutôt que la technique de gravure — une
**fuite de données** invisible au score interne.

La version publiée ici corrige ce biais : le partage se fait **par édition entière**, jamais
par image. Le corpus a par ailleurs été nettoyé de ses doublons (tirages successifs d'une
même plaque, scans recolorés), les classes rééquilibrées par échantillonnage pondéré, et la
graine aléatoire fixée pour rendre les variantes comparables entre elles.

**Données d'entraînement** : 27 éditions, 1 629 illustrations (1 169 / 227 / 233 en
apprentissage / validation / test), issues d'éditions numérisées par Gallica (BnF), la
Bayerische Staatsbibliothek (BSB/MDZ) et la Biblioteca Digital Ovidiana.

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
