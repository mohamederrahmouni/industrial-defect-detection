# Détection Automatisée des Défauts sur Pièces Industrielles (Impellers)

Projet de Deep Learning visant à automatiser le contrôle qualité de pièces de fonderie (*impellers*) à partir d'images, en classant chaque pièce comme **Normale (OK)** ou **Défectueuse**.

## 📌 Description du projet

Dans une chaîne de production industrielle, le contrôle visuel manuel des pièces coulées est lent, coûteux et sujet à l'erreur humaine. Ce projet propose une solution de **vision par ordinateur** basée sur des réseaux de neurones convolutifs pour automatiser cette inspection.

À partir d'un jeu de données d'images d'impellers (pièces de fonderie), plusieurs modèles de Deep Learning ont été entraînés, comparés puis déployés dans une application web interactive permettant de tester la classification en temps réel.

Le dataset utilisé (`Data/casting_dataset`) contient deux classes :
- `ok_front` : pièces normales
- `def_front` : pièces défectueuses

## 🎯 Objectifs

- Construire et comparer plusieurs architectures de classification d'images (MLP, CNN, Transfer Learning).
- Analyser et corriger les problèmes de sur-apprentissage (overfitting) via la régularisation et l'augmentation de données.
- Évaluer les modèles avec des métriques robustes (accuracy, précision, rappel, F1-score, matrice de confusion, courbe ROC).
- Déployer le meilleur modèle dans une application **Streamlit** simple et utilisable par un non-spécialiste.

## 🛠️ Technologies utilisées

- **Python 3**
- **TensorFlow / Keras** : construction et entraînement des modèles CNN et EfficientNetB0
- **Scikit-learn** : métriques d'évaluation (classification report, matrice de confusion, ROC/AUC)
- **NumPy, Matplotlib, Seaborn** : manipulation des données et visualisation
- **Streamlit** : application web
- **Google Colab** : environnement d'entraînement

## 🧠 Modèles de Deep Learning

Le notebook (`notebook.ipynb`) explore et compare trois approches, de la plus simple à la plus performante :

### 1. MLP 
Un perceptron multicouche simple utilisé comme référence de base, entraîné sur les images aplaties.

### 2. CNN (réseau de neurones convolutif)
Un CNN construit et entraîné en deux variantes :
- **Sans augmentation de données** : architecture régularisée pour limiter le sur-apprentissage.
- **Avec augmentation de données** : application de transformations (rotation, flip, zoom, etc.) pour améliorer la généralisation.

Les courbes d'apprentissage des deux variantes sont disponibles dans `Courbes/CNN.png` et `Courbes/CNN augmenté.png`.

### 3. EfficientNetB0 (Transfer Learning)
Un modèle pré-entraîné **EfficientNetB0** (ImageNet), réutilisé par **transfer learning** puis affiné (**fine-tuning**) sur le jeu de données d'impellers. C'est généralement le modèle offrant les meilleures performances, comme le montre la comparaison finale MLP vs CNN vs Transfer Learning dans le notebook.

Les courbes d'entraînement de ce modèle sont visibles dans `Courbes/EfficientNetB0.png`.

## 📂 Structure du projet

```
industrial-defect-detection/
├── Data/
│   └── casting_dataset/
│       ├── ok_front/        # Images de pièces normales
│       └── def_front/       # Images de pièces défectueuses
├── Courbes/                 # Courbes d'apprentissage (CNN, CNN augmenté, EfficientNetB0)
├── notebook.ipynb           # Entraînement, évaluation et comparaison des modèles
├── app.py                   # Application Streamlit
└── README.md
```

> ⚠️ Le dossier `models/` (contenant `model_cnn_aug.keras` et `efficientnet_model.keras`) attendu par `app.py` n'est pas inclus dans ce dépôt : il doit être généré en exécutant le notebook, ou ajouté manuellement si vous disposez déjà des modèles entraînés.

## ⚙️ Installation

1. **Cloner le dépôt**

```bash
git clone https://github.com/mohamederrahmouni/industrial-defect-detection.git
cd industrial-defect-detection
```

2. **Créer un environnement virtuel (recommandé)**

```bash
python -m venv venv
source venv/bin/activate   # Sur Windows : venv\Scripts\activate
```

3. **Installer les dépendances**

```bash
pip install streamlit streamlit-option-menu tensorflow numpy pillow matplotlib seaborn scikit-learn
```

## ▶️ Exécution du projet

### Entraîner / explorer les modèles

Ouvrir `notebook.ipynb` dans Jupyter ou Google Colab et exécuter les cellules dans l'ordre pour :
- charger et prétraiter les données,
- entraîner les modèles MLP, CNN et EfficientNetB0,
- visualiser les courbes d'apprentissage et les métriques,
- exporter les modèles entraînés (`.keras`) dans un dossier `models/`.

### Lancer l'application Streamlit

Une fois les modèles disponibles dans `models/model_cnn_aug.keras` et `models/efficientnet_model.keras` :

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur (par défaut sur `http://localhost:8501`).

## 🖥️ Présentation de l'application Streamlit

L'application propose un menu latéral avec trois sections :

- **CNN Model** : importez une image d'impeller (jpg/jpeg/png) et obtenez sa classification (Normale ou Défectueuse) par le modèle CNN, avec le score de probabilité associé.
- **EfficientNet B0 Model** : même fonctionnement, mais avec le modèle EfficientNetB0 pour une prédiction potentiellement plus précise.
- **Performances des modèles** : affiche les courbes d'apprentissage des trois configurations (CNN sans augmentation, CNN avec augmentation, EfficientNetB0) pour comparer visuellement leur comportement (overfitting, convergence, etc.).

Chaque image importée est automatiquement redimensionnée (224×224), normalisée puis passée au modèle sélectionné, qui retourne la classe prédite ainsi que la probabilité associée.

## 📈 Résultats

Les métriques détaillées (accuracy, précision, rappel, F1-score, matrices de confusion, courbes ROC) pour chaque modèle sont disponibles dans le notebook, section *"Comparaison globale — MLP vs CNN vs Transfer Learning"*.
