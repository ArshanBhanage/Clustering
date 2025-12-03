---

# 📘 **Clustering Assignment – README**

## 👤 **Author:** *Arshan Bhanage*

## 🧪 **Assignment:** Clustering Algorithms (a–i)

---

# 📖 **Overview**

This assignment demonstrates a wide range of **unsupervised learning** techniques across multiple data modalities, including:

* Tabular data
* Time series
* Text documents
* Images
* Audio

Each section (a–i) includes its own **Google Colab notebook** with:

* Clean, well-documented code
* Visualizations
* Clustering quality metrics
* Short interpretation and conclusions

All experiments use **public datasets** (HuggingFace, tslearn, torchaudio, torchvision) and modern embedding models including **Sentence Transformers** and **Meta ImageBind**.

---

# 📂 **Notebook Summary**

Below is a brief summary of all nine notebooks included in this assignment.

---

## **(a) K-Means Clustering from Scratch**

* **Dataset:** Palmer Penguins (HuggingFace)
* **Methods:**

	* Manual implementation of K-Means (no sklearn)
	* Distance calculation, centroid updates, convergence
* **Evaluation:** Silhouette score, inertia
* **Visualization:** PCA-based 2D cluster plot

---

## **(b) Hierarchical Clustering**

* **Dataset:** Palmer Penguins
* **Methods:**

	* Agglomerative clustering (Ward, Complete, Average, Single)
	* Dendrogram using SciPy
* **Evaluation:** Silhouette score, ARI
* **Visualization:** PCA 2D plots

---

## **(c) Gaussian Mixture Models**

* **Dataset:** Palmer Penguins
* **Methods:**

	* `sklearn.mixture.GaussianMixture`
	* Soft cluster probabilities
* **Model Selection:** BIC & AIC
* **Evaluation:** Silhouette score, ARI
* **Visualization:** PCA plots

---

## **(d) DBSCAN Clustering (PyCaret)**

* **Dataset:** Palmer Penguins
* **Framework:** PyCaret clustering module
* **Methods:**

	* DBSCAN model creation
	* Cluster assignment & comparison with K-Means
* **Evaluation:** Silhouette score, ARI
* **Visualization:** PCA cluster scatterplots

---

## **(e) Anomaly Detection Using PyOD**

* **Dataset:** Palmer Penguins (numeric features)
* **Method:** PyOD IsolationForest
* **Synthetic anomalies** injected into dataset
* **Evaluation:**

	* Confusion matrix
	* Precision, recall, F1
	* ROC-AUC
* **Visualization:** PCA anomaly maps

---

## **(f) Time-Series Clustering Using a Pretrained Model**

* **Dataset:** ECG200 (UCR Archive via tslearn)
* **Method:**

	* Train an autoencoder to learn latent embeddings
	* Cluster embeddings with K-Means
* **Evaluation:** Silhouette score, ARI
* **Visualization:**

	* Reconstruction loss curves
	* Latent space in PCA

---

## **(g) Document Clustering Using LLM Embeddings**

* **Dataset:** AG News (HuggingFace)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Methods:**

	* Embed all documents
	* K-Means clustering
	* Inspect sample texts per cluster
* **Evaluation:** Silhouette score, ARI
* **Visualization:** PCA document embedding map

---

## **(h) Image Clustering with ImageBind Embeddings**

* **Dataset:** CIFAR-10 subset (1,000 images)
* **Embeddings:** Meta AI **ImageBind** (Vision encoder)
* **Methods:**

	* Embed images using `imagebind_huge`
	* Cluster using K-Means
* **Evaluation:** Silhouette score
* **Visualization:**

	* PCA embedding scatterplot
	* Sample images per cluster

---

## **(i) Audio Clustering with ImageBind Embeddings**

* **Dataset:** Speech Commands (using `torchaudio.datasets.SPEECHCOMMANDS`)
* **Embeddings:** Meta AI **ImageBind** (Audio encoder)
* **Methods:**

	* Load `.wav` files
	* Extract audio embeddings
	* K-Means clustering
* **Evaluation:** Silhouette score, ARI
* **Visualization:** PCA scatterplot of audio embeddings
* **Analysis:** Cluster → true word distribution

---

# 📊 **Clustering Quality Metrics Used**

Each notebook includes at least one of the following (usually several):

* **Silhouette Score**
* **Davies–Bouldin Index**
* **Calinski–Harabasz Score**
* **Adjusted Rand Index (ARI)**
* **Inertia / SSE**
* **BIC / AIC** (for GMM)
* **Confusion Matrix & ROC-AUC** (for anomaly detection)

---

# 📦 **Dependencies**

The full assignment uses:

* `numpy`, `pandas`, `matplotlib`
* `sklearn`
* `datasets` (HuggingFace)
* `tslearn`
* `sentence-transformers`
* `pycaret`
* `pyod`
* `torch`, `torchaudio`, `torchvision`
* **ImageBind (Meta AI)** — installed via GitHub

All notebooks run in **Google Colab** with no local setup required.

---

# 🧠 **How to Run**

1. Open each notebook in Google Colab
2. Run **all cells** (top to bottom)
3. Review the final clustering results and discussion
4. Submit your notebook links as required by the assignment

---

# ✅ **Conclusion**

This assignment covers **the full spectrum of clustering** across diverse data types using both classical and modern machine learning techniques. It demonstrates:

* How clustering behaves on different modalities
* How embeddings dramatically improve cluster quality
* How to evaluate clustering results rigorously
* How to build multimodal pipelines using state-of-the-art models

Each notebook stands alone, but together they show a complete understanding of modern unsupervised learning.

---
