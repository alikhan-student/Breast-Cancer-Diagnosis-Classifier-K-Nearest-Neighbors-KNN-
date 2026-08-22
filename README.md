# Breast Cancer Diagnosis Classifier — K-Nearest Neighbors (KNN)

A machine learning project that classifies breast tumors as **Malignant (M)** or **Benign (B)** using the **K-Nearest Neighbors (KNN)** algorithm, trained on the Wisconsin Breast Cancer Diagnostic dataset.

## 📖 Overview

Breast cancer diagnosis relies heavily on accurately classifying tumor characteristics as cancerous or non-cancerous. This project builds a KNN classifier that uses 30 numerical features — computed from digitized images of breast mass cell nuclei (e.g. radius, texture, perimeter, area, smoothness) — to predict whether a tumor is malignant or benign.

The notebook walks through the full ML workflow: loading and exploring the data, preprocessing with feature scaling, training a baseline model, tuning the K hyperparameter, visualizing results, and plotting the decision boundary for interpretability.

## ✨ Features

- **Data preprocessing** — features standardized using `StandardScaler` for distance-based KNN performance
- **Train/test split** — 80/20 split with a fixed random state for reproducibility
- **Baseline model** — KNN classifier with `k=5`, achieving **~94.7% accuracy**
- **Hyperparameter tuning** — accuracy evaluated across `k=1` to `k=20` to identify the optimal number of neighbors
- **Multi-format visualizations** — pie chart, bar chart, scatter plot, and line chart comparing accuracy across all tested K values
- **Decision boundary visualization** — a 2D plot (using `radius_mean` and `texture_mean`) showing how KNN partitions the feature space between malignant and benign classes

## 🗂️ Dataset

- **Source:** Wisconsin Breast Cancer Diagnostic dataset (`breast-cancer.csv`)
- **Target variable:** `diagnosis` — `M` (Malignant) or `B` (Benign)
- **Features:** 30 numerical features describing cell nuclei characteristics (mean, standard error, and "worst"/largest values for radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension)

> **Note:** Place `breast-cancer.csv` in the project root directory before running the notebook.

## 🛠️ Tech Stack

- **Python 3**
- **NumPy** — numerical operations
- **Pandas** — data loading and manipulation
- **Matplotlib** — data visualization
- **Scikit-learn** — `KNeighborsClassifier`, `train_test_split`, `StandardScaler`, `accuracy_score`

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Running the Project
1. Clone this repository
2. Place `breast-cancer.csv` in the project directory
3. Open and run `Untitled5.ipynb` in Jupyter Notebook / JupyterLab

```bash
jupyter notebook Untitled5.ipynb
```

## 📊 Results

| K Value | Accuracy |
|---------|----------|
| 5       | ~94.7%   |

The model was evaluated across K values from 1 to 20 to find the optimal balance between bias and variance. Results are visualized in four chart formats for easy comparison, and the final decision boundary plot (using `radius_mean` vs `texture_mean` at `k=9`) illustrates how the classifier separates the two diagnosis classes in feature space.

## 📁 Project Structure
```
├── Untitled5.ipynb      # Main notebook — data loading, preprocessing, training, evaluation
├── breast-cancer.csv    # Dataset (not included — add your own copy)
└── README.md
```

## 🔮 Future Improvements
- Add cross-validation for more robust K selection
- Compare KNN against other classifiers (Logistic Regression, SVM, Random Forest)
- Add a confusion matrix and classification report (precision, recall, F1-score)
- Perform feature selection / dimensionality reduction (e.g. PCA) before classification
- Deploy as a simple web app for interactive predictions

## 📄 License
This project is open source and available for educational use.

## 👤 Author
Waqas Ali Khan