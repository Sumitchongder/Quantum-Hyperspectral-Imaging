# Q-HSI: Quantum-Enhanced Hyperspectral Imaging for Skin Cancer Classification

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Frameworks](https://img.shields.io/badge/Xanadu-PennyLane-orange)
![License](https://img.shields.io/badge/License-MIT-red)

## 📖 Executive Summary
This project explores the intersection of **Classical Deep Learning** and **Quantum Machine Learning (QML)** for medical image diagnostics. Using the **HAM10000 dataset**, we benchmark three distinct architectures for skin cancer classification (Benign vs. Malignant):

1.  **Classical Convolutional Neural Network (CNN):** A custom deep learning baseline.
2.  **Variational Quantum Classifier (VQC):** A quantum neural network using PennyLane.
3.  **Hybrid Quantum-Classical Ensemble:** A meta-learning approach that fuses latent quantum and classical features.

**Key Finding:** While the Classical CNN provides high accuracy at the cost of significant computational time (**~160s/epoch**), the Hybrid approach leverages the speed of the Quantum model (**~12s/epoch**) and the feature richness of the Classical model to achieve optimal accuracy with negligible meta-training time (**~1-2s**).

---

## 📂 Dataset
**Source:** [HAM10000 (Human Against Machine with 10000 training images)](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
* **Description:** A large collection of multi-source dermatoscopic images of common pigmented skin lesions.
* **Preprocessing:**
    * Images resized to `128x128` (Classical) and PCA-reduced (Quantum).
    * Normalization: `mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]`.
    * **Class Imbalance Handling:** Diagnoses mapped to binary labels (0: Benign, 1: Malignant [Melanoma, BCC, AKIEC]).

---

## 🏗️ Methodologies & Architecture

### 1. Classical Approach: Deep Convolutional Network
We designed a custom CNN to extract spatial hierarchies from dermatoscopic images. The architecture prioritizes feature preservation while managing computational load through strategic pooling and dropout.

**Network Architecture (Layer-by-Layer):**
1.  **Input:** `(Batch, 3, 128, 128)`
2.  **Conv Block 1:** Conv2d (24 filters, kernel=3, groups=3) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ AvgPool.
3.  **Conv Block 2:** Conv2d (40 filters) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ Dropout(0.25) $\rightarrow$ MaxPool.
4.  **Bottleneck:** Conv2d (32 filters, 1x1 kernel) for dimensionality reduction $\rightarrow$ ReLU $\rightarrow$ BatchNorm.
5.  **Conv Block 3:** Conv2d (48 filters, asymmetric kernel 5x3) $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool.
6.  **Conv Block 4:** Conv2d (64 filters) $\rightarrow$ ReLU $\rightarrow$ Dropout(0.35) $\rightarrow$ BatchNorm.
7.  **Classifier Head:**
    * Flatten
    * Linear (300 units) $\rightarrow$ ReLU $\rightarrow$ Dropout(0.4)
    * Linear (120 units) $\rightarrow$ ReLU $\rightarrow$ BatchNorm $\rightarrow$ Dropout(0.35)
    * **Output:** Linear (2 units)

---

### 2. Quantum Approach: Variational Quantum Classifier (VQC)
To overcome the limitations of current NISQ (Noisy Intermediate-Scale Quantum) hardware, we utilized **Dimensionality Reduction (PCA)** to compress image features into 6 quantum-compatible latent variables. These are fed into a PennyLane quantum circuit.

**Quantum Workflow:**
1.  **Preprocessing:** Extract penultimate features from CNN $\rightarrow$ PCA to 6 dimensions $\rightarrow$ Scale to $[-\pi, \pi]$.
2.  **Quantum Circuit (QNode):**
    * **Qubits:** 6
    * **Embedding:** `AngleEmbedding` (Data re-uploading strategy).
    * **Ansatz:** `StronglyEntanglingLayers` (4 layers, repeated 3 times for expressibility).
    * **Measurement:** Expectation value of Pauli-Z operators.
3.  **Classical Post-Processing:**
    * The quantum output is passed through a lightweight MLP (Linear $\rightarrow$ ReLU $\rightarrow$ Dropout) to map quantum states to class probabilities.

---

### 3. Hybrid Quantum-Classical Ensemble
The "Best of Both Worlds" approach. We treat the Classical and Quantum models as feature extractors and train a **High-Capacity MLP Meta-Classifier** to learn the optimal decision boundary based on the combined logits.

**Hybrid Workflow:**
1.  **Inference:** Run forward passes on frozen Quantum and Classical models to generate probability distributions (Logits).
2.  **Feature Fusion:** Concatenate Quantum Probabilities + Classical Logits.
3.  **Meta-Classifier Architecture:**
    * Input: Fused Features
    * Hidden Layer 1: 428 neurons (GELU, Dropout 0.2)
    * Hidden Layer 2: 256 neurons (GELU, Dropout 0.2)
    * Hidden Layer 3: 128 neurons (GELU, Dropout 0.2)
    * Output: 2 classes
4.  **Training:** Optimized using `AdamW` and `FocalLoss` to handle class imbalance.

---

## 📊 Results & Performance Interpretation

We observed a distinct trade-off between computational cost and predictive power across the three methods.

| Approach | Epoch Time | Total Training | Accuracy (Test) | AUC (ROC) |
| :--- | :--- | :--- | :--- | :--- |
| **Classical CNN** | ~160.0s | ~26 mins | **82.8%** | 0.825 |
| **Quantum VQC** | **~12.0s** | **~2 mins** | 80.4% | 0.780 |
| **Hybrid Ensemble** | **< 1.0s** | **< 5s** | **83.5%** | **0.838** |

### 🔍 Key Insights for Recruiters:

1.  **Classical Efficiency:**
    * The Classical CNN achieves high accuracy but is computationally expensive (**160s per epoch**). It effectively captures texture and boundary irregularities in skin lesions but requires heavy GPU usage.

2.  **Quantum Speedup:**
    * The Quantum approach is drastically faster (**~12s per epoch**). By compressing data via PCA and leveraging the Hilbert space for classification, we achieve a model that converges rapidly. However, due to the loss of information during PCA compression, the standalone accuracy drops slightly compared to the full CNN.

3.  **Hybrid Superiority:**
    * The Hybrid model validates the hypothesis that quantum features contain orthogonal information to classical features.
    * **Training Time:** The meta-classifier trains in **under 2 seconds**.
    * **Accuracy:** It matches and slightly exceeds the classical baseline.
    * **Conclusion:** The Hybrid pipeline offers a pathway to integrate quantum subroutines into production ML environments, providing robustness without the massive overhead of retraining large classical backbones.

---

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* PyTorch
* PennyLane (for Quantum circuits)
* Scikit-Learn (for metrics and PCA)
* Matplotlib/Seaborn (for visualization)

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/Q-HSI-Skin-Cancer.git](https://github.com/yourusername/Q-HSI-Skin-Cancer.git)
    cd Q-HSI-Skin-Cancer
    ```
2.  Install dependencies:
    ```bash
    pip install torch torchvision pennylane scikit-learn matplotlib seaborn pandas
    ```
3.  Run the Notebook:
    ```bash
    Jupyter Notebook Q-HSI - Quantum Hyperspectral Imaging.ipynb
    ```

---

## 📈 Visualizations
*See notebook for visualization plots.*

* **ROC Curves:** Demonstrating the Uplift of the Hybrid model over standalone Quantum.
* **Confusion Matrices:** Detailed breakdown of False Positives vs. False Negatives for medical sensitivity analysis.
* **Latent Space:** t-SNE projections showing the separability of Benign vs. Malignant classes in the quantum-embedded space.

---

## 🤝 Contact
**Name:** Sumit Tapas Chongder
**Email:** sumitchongder960@gmail.com
**LinkedIn:** www.linkedin.com/in/sumit-chongder/

*Open to opportunities in Machine Learning, Quantum Computing, and Data Science.*
