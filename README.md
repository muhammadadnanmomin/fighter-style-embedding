# Fighter Style Discovery using Autoencoders (Unsupervised Learning)

## Project Overview

This project applies unsupervised deep learning to discover latent fighting styles from fighter performance statistics.
Instead of using labels (e.g., striker/grappler), the model learns representations automatically using a deep autoencoder, followed by clustering and visualization.

#### The goal is to demonstrate:

- Representation learning

- Unsupervised clustering

- Interpretability of learned embeddings

- End-to-end ML workflow

## Problem Statement

Can we discover meaningful fighter styles without predefined labels, using only historical performance statistics?

This project answers that by:

- Learning low-dimensional embeddings from fighter stats

- Clustering fighters based on learned representations

- Interpreting the resulting groups as distinct fighting styles

## Methodology
### 1. Data Preparation

- Cleaned raw fighter statistics

- Converted physical and performance attributes to numeric form

- Handled missing values using domain-aware median imputation

- Standardized features for neural network training

### 2. Representation Learning (Autoencoder)

- Built a deep autoencoder in PyTorch

- Encoder compresses fighter stats into an 8-dimensional latent space

- Decoder reconstructs original inputs

- Trained using reconstruction loss (MSE)

### 3. Unsupervised Clustering

- Applied KMeans on learned embeddings

- Selected optimal clusters using the elbow method

- Identified 4 distinct latent fighting styles

### 4. Visualization

- Used PCA for linear sanity checks

- Used UMAP for nonlinear structure visualization

- Produced a 2D fighter style map

## Discovered Fighter Styles

| Cluster   | Style                              |
| --------- | ---------------------------------- |
| Cluster 0 | All-Rounders                       |
| Cluster 1 | High-Volume Strikers               |
| Cluster 2 | Low-Engagement Fighters            |
| Cluster 3 | Grappling / Submission Specialists |

These styles emerged without labels, purely from data.

## Key Results

- Autoencoder learned meaningful latent representations

- Clear separation of fighter styles in UMAP space

- Statistical profiling validates each cluster’s interpretation

Tech Stack

- Python

- Pandas, NumPy

- Scikit-learn

- PyTorch

- UMAP

- Matplotlib, Seaborn

## Repository Structure
```
fighter-style-embedding/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_feature_scaling_tensor_prep.ipynb
│   ├── 03_autoencoder_training.ipynb
│   └── 04_clustering_and_visualization.ipynb
│
├── models/
│   ├── autoencoder.pth
│   └── scaler.pkl
│
├── requirements.txt
└── README.md
```

## ⚠️ Disclaimer

This project is for educational and research purposes only.
No scraped data is distributed with this repository.

## Future Work

- Fighter similarity search using embeddings

- P4P-style ranking system

- Interactive Streamlit visualization

- Temporal style evolution analysis
