# 🌪️ PatchGum  
**Sparse Gumbel-Guided Subsequence Selection for Efficient Long-Horizon Time Series Forecasting**

> PatchGum introduces a **Gumbel-guided sparse attention framework** that dynamically selects the most informative temporal patches, achieving efficient and accurate long-horizon forecasting across volatile and periodic time series.

---

## 🧠 Overview

Traditional Transformer-based forecasting models process all subsequences uniformly, often leading to **redundant computation** and **overfitting** on short-term noise.  
PatchGum tackles this by using a **dynamic Gumbel-Softmax Top-k mechanism** to adaptively focus on critical subsequences, combining **Full-Attention modeling** and **lightweight decomposition** for efficient hybrid computation.

<p align="center">
  <img src="assets/patchgum_architecture.png" width="80%" alt="PatchGum Framework"/>
</p>

---

## 🔍 Key Features

- ⚡ **Dynamic Subsequence Selection**  
  Learnable Gumbel-Softmax Top-k sampling highlights the most informative temporal patches.

- 🧩 **Dual-Branch Hybrid Modeling**  
  - **Full Branch:** Deep attention modeling for high-importance patches.  
  - **Lite Branch:** Lightweight decomposition for low-importance ones.

- 💡 **Sparse & Scalable**  
  Reduces attention complexity while maintaining strong temporal representation.

- 🌏 **Generalizable**  
  Performs effectively on both volatile (e.g., wind power) and periodic (e.g., ETT) datasets.

---

## 🏗️ Repository Structure

