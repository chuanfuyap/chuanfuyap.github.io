---
title: "t-SNE vs UMAP vs PCA"
published: true
tags: python dimension-reduction pca tsne umap
sidebar:
  title: "Skip Buttons"
  nav: dim-red
description: "Comparison of commonly used dimension reduction methods, ~10 min read"
---

tldr summarytable;

| Feature                    | PCA    | t-SNE              | UMAP               |
| -------------------------- | ------ | ------------------ | ------------------ |
| Type                       | Linear | Non-linear         | Non-linear         |
| Preserves Local Structure  | No     | Yes                | Yes                |
| Preserves Global Structure | Yes    | No                 | Somewhat           |
| Runtime Performance        | Fast   | Slow               | Moderate           |
| Interpretability           | High   | Low                | Low                |
| Use in ML Pipelines        | Yes**    | No*  | No* |
| Supports projection of new data          | Yes    | No                 | Yes                 |

* t-SNE and UMAP are typically used for visualization rather than as preprocessing steps in machine learning pipelines.

** PCA is commonly used for noise reduction, clustering, and as input to supervised models as part of machine learning pipelines. 

# Introduction
In molecular biology—which spans genetics, transcriptomics, proteomics, metabolomics, and other -omics fields—we often work with high-dimensional data. Dimensionality reduction methods are commonly employed to reduce the number of features while retaining the most essential information. Most such methods fall into one of two broad categories: matrix factorisation (e.g. PCA) or graph layout-based techniques (e.g. t-SNE and UMAP).

Below, I provide a brief overview of the three most widely used dimensionality reduction techniques—PCA, t-SNE, and UMAP—and highlight key differences between them.

PCA is widely used to represent population structure in genetic datasets, while t-SNE and UMAP are more commonly applied to single-cell RNA sequencing data, where capturing local structure and cluster separation is important. Although UMAP has also been proposed for representing population structure in genetic data, it is less commonly adopted for this purpose and, in my view, less preferable due to its lower interpretability compared to PCA.

<a class="anchor" id="pca"></a>

# Principal Component Analysis (PCA)
Principal Component Analysis (PCA) is a linear dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible. It is one of the oldest and most widely used methods for dimensionality reduction.

## How PCA works
Principal Component Analysis (PCA) is a linear dimensionality reduction technique that transforms high-dimensional data into a new coordinate system such that the greatest variance in the data lies along the first axis (the first principal component), the second greatest variance along the second axis, and so on. This is achieved by computing the eigenvectors (also called principal component directions or loadings) and eigenvalues (which quantify the variance explained by each component) of the data’s covariance matrix. The PC scores are the coordinates of each sample in the new reduced space after projection.

PCA can also be computed via Singular Value Decomposition (SVD) of the centred data matrix, which decomposes it into orthogonal components that represent directions of maximum variance. Alternatively, for large or incomplete datasets, iterative algorithms such as NIPALS (Nonlinear Iterative Partial Least Squares) can be used to estimate the principal components one at a time.

Key parameters include n_components, which determines how many principal components to retain. For visualisation, this is typically set to 2 or 3; for feature reduction, it can be set to retain a fixed number of components or to capture a target proportion of the total variance (e.g. 95%). Other considerations include standardisation of the input data—PCA assumes that each feature is on the same scale, so preprocessing with mean-centering and scaling to unit variance (e.g. using z-scores) is typically essential.

## Key Rules when using PCA
1. **PCA assumes linearity**. It only captures linear correlations. If your data lies on a non-linear manifold, PCA may not reveal meaningful structure.
2. **The direction of a component is not unique**. Eigenvectors can be flipped (i.e. multiplied by -1) and the result is still valid. So don’t over-interpret directionality in the plots.
3. **Explained variance is useful, but not everything**. Retaining 95% variance doesn't guarantee biological or conceptual interpretability. Low-variance PCs may still carry meaningful signal.
4. **Distances and angles are meaningful**. Unlike t-SNE or UMAP, PCA preserves global geometry, so Euclidean distances and directions in PCA space do carry information.

## Compared to other methods
Unlike non-linear methods like t-SNE or UMAP, PCA preserves global linear structure, making it particularly useful for understanding overall variance patterns or for downstream tasks like clustering and regression. PCA is deterministic and fast to compute, especially on moderately sized datasets, and the resulting components are directly interpretable in terms of how they relate to the original features. PCA can be useful beyond visualisation. You can use PCA-reduced features as inputs to downstream models (e.g. clustering, regression, classification). The resulting PC loadings from fitting the model can be used to project new data onto the reduced dimension space (latent space), which is not possible with t-SNE or UMAP. This makes PCA a versatile tool for both exploratory data analysis and as a preprocessing step in machine learning pipelines.

<a class="anchor" id="tsne"></a>

# t-Distributed Stochastic Neighbor Embedding (t-SNE)
t-SNE is explained in detail [here](https://distill.pub/2016/misread-tsne/). t-SNE is a non-linear dimensionality reduction technique that is particularly effective for visualising high-dimensional data in lower dimensions (typically 2D or 3D). 

## How t-SNE works
It is based on the idea of converting high-dimensional Euclidean distances between points into conditional probabilities that represent pairwise similarities, and then minimising the Kullback–Leibler (KL) divergence between these probability distributions in the lower-dimensional space. 

It proceeds in two main stages. First, it measures pairwise similarities in the high‑dimensional space: each point’s neighbourhood is modelled as a conditional probability distribution whose bandwidth is chosen so that the perplexity (an information‑theoretic proxy for the effective number of neighbours) matches a user‑supplied value. Second, it optimises a low‑dimensional map by placing points so that an analogous distribution—now based on a heavy‑tailed Student’s t‑distribution—matches the original similarities as closely as possible. The mismatch between the two sets of probabilities is quantified with the Kullback–Leibler (KL) divergence, which t‑SNE minimises via gradient descent.

Key parameters include `perplexity` (controls the local/global balance), `learning_rate`, and `early_exaggeration` (temporarily amplifies attractive forces to sharpen cluster separation at the start of optimisation). Lower perplexity values emphasise very local structure, while higher values allow somewhat broader relationships to influence the embedding—though t‑SNE remains primarily a local‑structure method.

## Key Rules when using t-SNE
1. **Hyperparameters really matter**, you need run it several times with different `perplexity` to find the best fit for your data. 
2. **Cluster sizes in a UMAP plot mean nothing**, they are meaningless. 
3. **Distances between clusters might not mean anything**, this is due to using local distances when constructing the graph.
4. **Random noise doesn’t always look random.**, spurious clustering can occur. 
5. **You can see some shapes, sometimes**.
6. **For topology, you may need more than one plot**, again refer to point number 1. Make many plots with different perplexity values. 

## Compared to other methods
Relative to UMAP, t‑SNE excels at revealing tight local clusters, often separating sub‑populations that other techniques blur together. Its main trade‑off is a weaker preservation of global structure: inter‑cluster distances and cluster sizes are not trustworthy, so the overall geometry of the map can be misleading. t‑SNE is also computationally heavier (slower) than UMAP, especially on large datasets, and the embedding can vary noticeably across runs unless the random seed and parameters are fixed.

Compared with PCA, t‑SNE is non‑linear, making it far better for exploring complex manifolds in data such as single‑cell RNA‑seq. However, like UMAP, it warps the original space, so axis labels (“t‑SNE1”, “t‑SNE2”, etc.) and raw distances have no direct interpretative meaning. Unlike PCA, t-SNE cannot be used to project new dataset onto the reduced dimension space (latent space). This lack of interpretability, inability to project new data, combined with its stochastic nature and sensitivity to hyper‑parameters, means t‑SNE is used almost exclusively for visualisation, not as a feature‑engineering step in machine‑learning pipelines.

<a class="anchor" id="umap"></a>

# Unified Manifold Approximation and Projection (UMAP)
UMAP is clearly explained in this [blog post](https://pair-code.github.io/understanding-umap/). It is a non-linear dimensionality reduction technique that is particularly effective for visualizing high-dimensional data in lower dimensions (typically 2D or 3D). UMAP is based on manifold learning and graph theory, and it aims to preserve both local and some global structure of the data.

## How UMAP works
In its simplest sense, the UMAP algorithm consists of two steps: construction of a graph in high dimensions followed by an optimization step to find the most similar graph in lower dimensions. UMAP essentially constructs a weighted graph from the high dimensional data, with edge strength representing how “close” a given point is to another, then projects this graph down to a lower dimensionality. UMAP connects more and more neighboring points when constructing the graph representation of the high-dimensional data, which leads to a projection that more accurately reflects the global structure of the data. At very low values, any notion of global structure is almost completely lost. 

Key parameters when running UMAP include `min_dist`, which controls how tightly points are clustered in the low-dimensional space, and `n_neighbors`, which adjusts the balance between local detail and global structure—higher values preserve more of the broad data relationships. Additionally, UMAP supports various distance metrics (e.g. `'euclidean'`, `'cosine'`), which can be chosen to better suit the data type and structure. As `min_dist` increases, UMAP tends to "spread out" the projected points, leading to decreased clustering and less emphasis on local relationships

## Key Rules when using UMAP
Basically same as t-SNE. 

1. **Hyperparameters really matter**, you need run UMAP several times with different parameters to find the best fit for your data. 
2. **Cluster sizes in a UMAP plot mean nothing**, they are meaningless, just like t-SNE's clusters. 
3. **Distances between clusters might not mean anything**, this is due to using local distances when constructing the graph.
4. **Random noise doesn’t always look random.**, spurious clustering can occur. 
5. **You may need more than one plot**, refer to point number 1. 

## Compared to other methods
The biggest difference between the output of UMAP when compared with t-SNE is this balance between local and global structure - UMAP is often better at preserving global structure in the final projection. This means that the inter-cluster relations are potentially more meaningful than in t-SNE. While UMAP is also a stochastic algorithm, it's striking how similar the resulting projections are from run to run and with different parameters. This is due, again, to UMAP's increased emphasis on global structure in comparison to t-SNE. Comparisons have found that UMAP runs faster than t-SNE, especially on larger datasets, while still producing similar or better quality visualizations. This makes UMAP a popular choice for visualizing high-dimensional data.

However, it's important to note that, because UMAP and t-SNE both necessarily warp the high-dimensional shape of the data when projecting to lower dimensions, any given axis or distance in lower dimensions still isn’t directly interpretable in the way of techniques such as PCA. This is reflected in the axis labels of UMAP plots, which are often just labeled as "UMAP1", "UMAP2", etc., without any specific meaning attached to them. UMAP allows for projecting new dataset onto the latent space just like PCA. The combination of lack of interpretability and stochasticity in the output of UMAP means that it is not suitable for use in machine learning pipelines, where interpretability and reproducibility are important, and it is primarily used for visualization purposes.


<a class="anchor" id="examples"></a>

# Visual Examples 
There are already many blog posts and articles that presents visualisation of these methods (such as the above links), so I will not repeat them here. Instead, I will give you links to what the great scikit-learn authors have prepared, which covers the less popular methods as well.
- [Decomposition methodds (which includes PCA)](https://scikit-learn.org/stable/modules/decomposition.html#decompositions)
- [Manifold Learning Methods](https://scikit-learn.org/stable/modules/manifold.html#manifold)
- [Comparison of Manifold Learning methods](https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py)
- [Manifold Learning methods on a severed sphere](https://scikit-learn.org/stable/auto_examples/manifold/plot_manifold_sphere.html#sphx-glr-auto-examples-manifold-plot-manifold-sphere-py)
- [Manifold learning on handwritten digits](https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py)
- [Multi-dimensional scaling (MDS)](https://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html#sphx-glr-auto-examples-manifold-plot-mds-py)
- [Swiss Roll And Swiss-Hole Reduction](https://scikit-learn.org/stable/auto_examples/manifold/plot_swissroll.html#sphx-glr-auto-examples-manifold-plot-swissroll-py)
- [t-SNE: The effect of various perplexity values on the shape](https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py)
