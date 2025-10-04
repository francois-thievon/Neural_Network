# Neural_network.ipynb — Perceptron logistique et réseau de neurones profond en NumPy

Ce notebook construit pas à pas:
- un perceptron logistique binaire “from scratch” en NumPy, avec affichage interactif de la frontière de décision et de la courbe de loss;
- un petit réseau de neurones profond (MLP) en empilant le neurone précédent, avec propagation avant/arrière, mise à jour des poids et visualisations en direct.

Fichier principal:
- [Neural_network.ipynb](Neural_network.ipynb)

## Contenu et API

Classes principales définies dans [Neural_network.ipynb](Neural_network.ipynb):
- Perceptron logistique:
  - [`artificial_neuron`](Neural_network.ipynb)
    - Initialisation et utilitaires: [`artificial_neuron.init_params`](Neural_network.ipynb), [`artificial_neuron._sigmoid`](Neural_network.ipynb)
    - Propagation avant: [`artificial_neuron.forward`](Neural_network.ipynb)
    - Rétropropagation:
      - Couches cachées: [`artificial_neuron.backward`](Neural_network.ipynb) (chaîne de gradient via la sigmoïde)
      - Couche de sortie BCE+sigmoïde: [`artificial_neuron.backward_output`](Neural_network.ipynb) avec $dZ=\hat y - y$
    - Mise à jour: [`artificial_neuron.update_params`](Neural_network.ipynb)
    - API “simple” pour le perceptron: [`artificial_neuron.predict_proba`](Neural_network.ipynb), [`artificial_neuron.predict`](Neural_network.ipynb), [`artificial_neuron.log_loss`](Neural_network.ipynb), [`artificial_neuron.fit`](Neural_network.ipynb)

- Réseau de neurones (MLP) basé sur la classe ci‑dessus:
  - [`neural_network`](Neural_network.ipynb)
    - Construction des couches: liste de listes de neurones selon `architecture` (ex. `[2, 8, 8, 1]`)
    - Initialisation des poids: [`neural_network._init_all_weights`](Neural_network.ipynb)
    - Propagation avant sur toutes les couches: [`neural_network._forward_all`](Neural_network.ipynb), exposée via [`neural_network.forward`](Neural_network.ipynb) et [`neural_network.predict_proba`](Neural_network.ipynb)
    - Perte BCE: [`neural_network._bce_loss`](Neural_network.ipynb)
    - Entraînement (forward → backward sortie → backward cachées → update): [`neural_network.fit`](Neural_network.ipynb)
    - Prédiction binaire: [`neural_network.predict`](Neural_network.ipynb)

Datasets et démos:
- Scikit-learn: `make_blobs`, `make_moons`, `make_circles`
- Générateurs maison: `make_xor`, `make_offset_circles`, `make_donut_vs_arcs`
- Visualisations: Matplotlib (frontière + loss), Plotly 3D pour la surface sigmoïde

## Détails algorithmiques

- Activation sigmoïde: $\sigma(z)=\frac{1}{1+e^{-z}}$
- Perte logistique binaire (BCE):
  $$
  \mathcal{L}(y,\hat y)=-\frac{1}{m}\sum_{i=1}^{m}\big[y_i\log(\hat y_i)+(1-y_i)\log(1-\hat y_i)\big]
  $$
- Sortie (BCE+sigmoïde): le gradient simplifie à $dZ=\hat y - y$, puis
  $$
  dW=\frac{1}{m}X^\top dZ,\quad db=\mathrm{mean}(dZ),\quad dA_{\text{prev}}=dZ\,W^\top
  $$
- Couches cachées: rétroprop via la dérivée de la sigmoïde $dZ=dA\cdot \sigma(A)\cdot(1-\sigma(A))$

## Visualisations “live”

Pendant [`artificial_neuron.fit`](Neural_network.ipynb) et [`neural_network.fit`](Neural_network.ipynb):
- Graphe 1: frontière de décision 2D, isolignes et nuage de points coloré par classe
- Graphe 2: courbe de loss en temps réel
- Une figure Plotly 3D montre aussi la surface $\sigma(W_0 x_0 + W_1 x_1 + b)$ apprise par le perceptron

## Exemples fournis

- Perceptron sur `make_blobs` (binaire), avec affichage au fil de l’entraînement
- MLP sur:
  - `make_moons` et `make_circles`
  - Blobs séparables
  - Jeux non linéaires maison: XOR, cercles décalés, anneau vs arcs

Chaque exemple crée les données, instancie le modèle avec une architecture et des hyperparamètres adaptés, puis appelle `.fit(..., plot_live=True)` pour visualiser l’apprentissage.

## Prérequis

- Python 3.11+
- Bibliothèques: numpy, matplotlib, scikit-learn, tqdm, plotly, IPython

Installation rapide:
```bash
pip install numpy matplotlib scikit-learn tqdm plotly ipython
```

## Utilisation

- Ouvrir [Neural_network.ipynb](Neural_network.ipynb) dans VS Code/Jupyter
- Exécuter les cellules de haut en bas
- Adapter `architecture`, `n_iter`, `learning_rate` et les générateurs de données selon vos besoins

## Notes

- La dernière couche est contrainte à 1 neurone (classification binaire BCE).
- Les poids sont initialisés via une échelle $1/\sqrt{n_{in}}$ dans chaque [`artificial_neuron.init_params`](Neural_network.ipynb) pour des activations stables.
- Le pas de visualisation se contrôle via `plot_interval`.
