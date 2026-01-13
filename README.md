# Fighter Style Discovery using Autoencoders
Unsupervised Representation Learning | PyTorch | Clustering | UMAP

## Problem Statement

Most fighter classification systems rely on hand-crafted labels such as “striker” or “grappler.”
But what if these labels are incomplete, biased, or oversimplified?

This project explores a more fundamental question:

> Can we discover meaningful fighting styles directly from data — without using any labels?

Using historical performance statistics, this system learns latent representations of fighters via a deep autoencoder and then clusters these embeddings to uncover natural groupings.

## Why This Project Matters

This project demonstrates:

- Representation learning
- Unsupervised pattern discovery
- Deep learning on tabular data
- Interpretability of embeddings
- End-to-end ML workflows

These techniques are widely used in:
- Recommender systems
- User segmentation
- Anomaly detection
- Market clustering
- Behavioral profiling

## Dataset

The dataset contains historical fighter performance statistics, including:

- Striking volume
- Accuracy metrics
- Takedown attempts
- Submission activity
- Physical attributes

All features were converted to numeric form and standardized for neural network training.

## Methodology
#### 1. Data Preparation

- Cleaned raw fighter statistics
- Converted categorical and physical attributes to numeric format
- Handled missing values using domain-aware median imputation
- Standardized all features for stable neural network training

#### 2. Representation Learning (Autoencoder)

A deep autoencoder was trained to learn compact, informative embeddings.

- Implemented using PyTorch
- Encoder compresses fighter stats into an 8-dimensional latent space
- Decoder reconstructs original input
- Trained using Mean Squared Error (MSE) loss

This forces the network to learn the most informative structure in the data.

#### 3. Unsupervised Clustering

Once embeddings were learned:

- Applied KMeans clustering on the latent space
- Used the elbow method to determine optimal number of clusters
- Identified 4 distinct latent fighting styles

#### 4. Visualization

To interpret the learned structure:

- Used PCA for linear sanity checks
- Used UMAP for nonlinear manifold visualization
- Generated a 2D “fighter style map”

## Discovered Fighter Styles

| Cluster | Interpreted Style                  |
| ------- | ---------------------------------- |
| 0       | All-Rounders                       |
| 1       | High-Volume Strikers               |
| 2       | Low-Engagement Fighters            |
| 3       | Grappling / Submission Specialists |

These styles emerged purely from data, without any labels.

## Key Results

- Autoencoder learned meaningful low-dimensional embeddings
- Clear cluster separation observed in UMAP space
- Statistical profiling validated cluster interpretations
- Demonstrated that unsupervised deep learning can extract semantic structure from tabular sports data

## Tech Stack

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

## Key Learnings

- How autoencoders learn compressed representations
- Why representation learning is powerful for unsupervised tasks
- Trade-offs between linear vs nonlinear dimensionality reduction
- Interpreting black-box embeddings using visualization
- Unsupervised clustering validation techniques

## ⚠️ Disclaimer

This project is for educational and research purposes only.
No scraped or proprietary data is distributed with this repository.

## Future Work

- Fighter similarity search using embeddings
- P4P-style ranking using latent features
- Interactive Streamlit visualization
- Temporal evolution of fighter styles
- Transfer learning across promotions

## Author

Adnan Momin

LinkedIn: https://www.linkedin.com/in/adnanmomin/

GitHub: https://github.com/muhammadadnanmomin
