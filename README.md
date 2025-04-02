# EXXA GSoC 2025 Test Submission

## Overview
This submission implements three key components for the EXXA project:
1. Unsupervised Clustering (EXXA2)
2. Transit Curve Classifier (EXXA3)
3. Autoencoder (EXXA4)

## Results Summary

### Clustering (EXXA2)
- Best silhouette score: 0.433 with 2 clusters
- Data preprocessing: Removed 91 outliers, retained 7 components explaining 95% variance
- Visualization: PCA plots and cluster visualizations available in `EXXA_Results/clustering/`

### Transit Classifier (EXXA3)
- Test AUC: 0.993
- Test Average Precision: 0.994
- Validation Accuracy: 96.00%
- Training stopped at epoch 27 with early stopping
- ROC and PR curves available in `EXXA_Results/metrics/`

### Autoencoder (EXXA4)
- Final Training Loss: 0.0113
- Final Validation Loss: 0.0150
- Reconstructed images available in `EXXA_Results/autoencoder/`

## Visualizations

### Training Progress
![Autoencoder Training History](EXXA_Results/training_progress/autoencoder_training_history.png)
*Autoencoder training and validation loss over epochs*

![Classifier Training History](EXXA_Results/training_progress/classifier_training_history.png)
*Classifier training and validation metrics over epochs*

### Performance Metrics
![ROC Curve](EXXA_Results/metrics/roc_curve.png)
*Receiver Operating Characteristic (ROC) curve for the classifier*

![PR Curve](EXXA_Results/metrics/pr_curve.png)
*Precision-Recall (PR) curve for the classifier*

### Clustering Results
![PCA Visualization](EXXA_Results/clustering/pca_visualization.png)
*PCA visualization of the clustered data*

![Cluster Visualization](EXXA_Results/clustering/cluster_visualization.png)
*Visual representation of the identified clusters*

### Autoencoder Results
![Reconstruction Examples](EXXA_Results/autoencoder/reconstruction_examples.png)
*Original vs reconstructed images from the autoencoder*

## Directory Structure
```
EXXA_Results/
├── checkpoints/
│   ├── best_autoencoder.pth
│   ├── best_classifier.pth
│   ├── autoencoder_final.pth
│   └── classifier_final.pth
├── training_progress/
│   ├── autoencoder_training_history.png
│   └── classifier_training_history.png
├── metrics/
│   ├── roc_curve.png
│   └── pr_curve.png
├── clustering/
│   ├── pca_visualization.png
│   └── cluster_visualization.png
└── autoencoder/
    └── reconstruction_examples.png
```

## Dependencies
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- astropy
- pytorch_msssim

## Installation
1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the main script:
```bash
python exxa2.py
```

## Implementation Details

### Clustering (EXXA2)
- Implemented KMeans clustering
- Used silhouette score for evaluation
- Preprocessing includes outlier removal and PCA
- Visualizations include PCA plots and cluster examples

### Transit Classifier (EXXA3)
- Attention-based neural network
- BCEWithLogitsLoss for training
- Early stopping with patience=5
- Comprehensive evaluation metrics

### Autoencoder (EXXA4)
- Convolutional autoencoder architecture
- MSE loss for training
- Latent space visualization
- Reconstruction quality assessment

## Contact
For any questions about this submission, please contact the author. 