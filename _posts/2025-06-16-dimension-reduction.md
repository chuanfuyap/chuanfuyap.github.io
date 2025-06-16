---
title: "t-SNE vs UMAP vs PCA"
published: false
tags: python dimension-reduction pca tsne umap
sidebar:
  title: "Skip Buttons"
  nav: dim-red
description: "comparison of commonly used dimension reduction methods"
---

tldr table;

| Feature                    | PCA    | t-SNE              | UMAP               |
| -------------------------- | ------ | ------------------ | ------------------ |
| Type                       | Linear | Non-linear         | Non-linear         |
| Preserves Local Structure  | No     | Yes                | Yes                |
| Preserves Global Structure | Yes    | No                 | Somewhat           |
| Runtime Performance        | Fast   | Slow               | Moderate           |
| Interpretability           | High   | Low                | Low                |
| Use in ML Pipelines        | Yes    | No*  | No* |

* t-SNE and UMAP are typically used for visualization rather than as preprocessing steps in machine learning pipelines.

# Introduction



<a class="anchor" id="pca"></a>

# Principal Component Analysis (PCA)

<a class="anchor" id="tsne"></a>

# t-Distributed Stochastic Neighbor Embedding (t-SNE)

<a class="anchor" id="umap"></a>

# Unified Manifold Approximation and Projection (UMAP)

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
