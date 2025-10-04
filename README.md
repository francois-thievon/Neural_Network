<div align="center">

# 🧠 Neural Network (NumPy)

Perceptron logistique et petit MLP codés from scratch, avec visualisations interactives dans un notebook Jupyter.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](#prerequis) 
[![NumPy](https://img.shields.io/badge/NumPy-✅-013243?logo=numpy&logoColor=white)](#prerequis)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-📈-11557c)](#visualisations)
[![Scikit‑learn](https://img.shields.io/badge/scikit--learn-🧪-f89939?logo=scikitlearn&logoColor=white)](#datasets)

🎯 Fichier principal: <strong><a href="Neural_network.ipynb">Neural_network.ipynb</a></strong>

</div>

---

## Sommaire

- [Aperçu](#aperçu)
- [Fonctionnalités](#fonctionnalités)
- [Démo rapide](#démo-rapide)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [API dans le notebook](#api-dans-le-notebook)
- [Datasets](#datasets)
- [Visualisations](#visualisations)
- [Détails algorithmiques](#détails-algorithmiques)
- [Structure du projet](#structure-du-projet)
- [Notes](#notes)

## Aperçu

Ce projet didactique montre comment implémenter:
- un perceptron logistique binaire en NumPy, avec frontière de décision + courbe de loss en direct;
- un petit réseau de neurones (MLP) en empilant le neurone précédent, avec forward/backward et update des poids.

Tout est regroupé et expliqué dans le notebook: [Neural_network.ipynb](Neural_network.ipynb).

## Fonctionnalités

- ✍️ Implémentation “from scratch” (NumPy pur)
- 🔁 Propagation avant/arrière + mise à jour des poids
- 📉 Visualisation live: frontière 2D et courbe de loss
- 🧪 Jeux de données jouets (scikit‑learn + générateurs maison)
- 🧰 API minimaliste pour entraîner, prédire et tracer

## Démo rapide

1) Installer les dépendances nécessaires.
2) Ouvrir le notebook.
3) Lancer les cellules “Exemples”.

Résultat: une frontière de décision qui évolue pendant l’entraînement et une courbe de loss en temps réel. ✨

## Prérequis

- Python 3.11+
- Bibliothèques: numpy, matplotlib, scikit-learn, tqdm, plotly, IPython

## Installation

```bash
pip install numpy matplotlib scikit-learn tqdm plotly ipython
```

## Utilisation

- Ouvrir [Neural_network.ipynb](Neural_network.ipynb) dans VS Code/Jupyter.
- Exécuter les cellules de haut en bas.
- Ajuster `architecture`, `n_iter`, `learning_rate` et les générateurs de données selon vos besoins.

## API dans le notebook

Classes principales définies dans [Neural_network.ipynb](Neural_network.ipynb):

- Perceptron logistique — classe `artificial_neuron`
  - Initialisation/utilitaires: `init_params`, `_sigmoid`
  - Forward: `forward`
  - Backward:
    - Couches cachées: `backward` (chaîne de gradient via sigmoïde)
    - Sortie BCE+sigmoïde: `backward_output` avec dZ = y_hat − y
  - Mise à jour: `update_params`
  - API pratique: `predict_proba`, `predict`, `log_loss`, `fit`

- Réseau de neurones (MLP) — classe `neural_network`
  - Construction des couches depuis `architecture` (ex. `[2, 8, 8, 1]`)
  - Initialisation: `_init_all_weights`
  - Forward sur toutes les couches: `_forward_all` → `forward`/`predict_proba`
  - Perte: `_bce_loss`
  - Entraînement: `fit` (forward → backward sortie → backward cachées → update)
  - Prédiction binaire: `predict`

## Datasets

- Scikit‑learn: `make_blobs`, `make_moons`, `make_circles`
- Maison: `make_xor`, `make_offset_circles`, `make_donut_vs_arcs`

## Visualisations

- Graphe 1: frontière de décision 2D (isolignes + nuage de points)
- Graphe 2: courbe de loss en temps réel
- Bonus: surface sigmoïde 3D (Plotly) apprise par le perceptron

## Détails algorithmiques

- Sigmoïde: σ(z) = 1 / (1 + e^(−z))
- BCE (log loss binaire):
  L(y,ŷ) = −(1/m) Σ [ y log(ŷ) + (1−y) log(1−ŷ) ]
- Sortie BCE+sigmoïde: gradient simplifié dZ = ŷ − y, puis
  dW = (1/m) Xᵀ dZ,  db = mean(dZ),  dA_prev = dZ Wᵀ
- Couches cachées: dZ = dA · σ(A) · (1 − σ(A))

## Structure du projet

```
.
├── Neural_network.ipynb   # Notebook principal (tout le code et les explications)
└── README.md              # Vous êtes ici
```

## Notes

- La dernière couche contient 1 neurone (classification binaire, BCE).
- Initialisation des poids ~ 1/√n_in pour garder des activations stables.
- Le rythme d’affichage se règle via `plot_interval`.
