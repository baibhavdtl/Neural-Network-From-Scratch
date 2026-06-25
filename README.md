# Neural Network From Scratch (NNFS)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-darkgreen.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, fully custom Deep Learning pipeline built entirely from scratch using **NumPy** and **Python**. This project implements the fundamental mathematics of neural networks—including forward propagation, backpropagation, activation derivatives, and loss gradients—without relying on high-level frameworks like TensorFlow, Keras, or PyTorch.

---

## Key Features

* **Pure NumPy Implementation:** Core matrix operations and linear algebra handled entirely via NumPy.
* **Complete Backpropagation:** Manual calculation of exact gradients for weights, biases, and activation states.
* **Non-Linear Activations:** Custom implementation of **ReLU** for hidden layers and **Softmax** for stable probability distributions.
* **Loss Evaluation:** Fully implemented **Categorical Cross-Entropy Loss** for robust multi-class classification evaluation.

---

## Model Architecture

The network is configured to classify non-linear, complex data on a 2D plane:

| Layer | Type | Configuration / Output Dimensions | Activation |
| :--- | :--- | :--- | :--- |
| **Input** | Data Points | 2 Features $(x, y)$ | None |
| **Hidden Layer 1** | Fully Connected | 64 Neurons | ReLU |
| **Output Layer** | Fully Connected | 3 Neurons (Classes) | Softmax |

---

## Dataset: Spiral Data

This project utilizes the classic **NNFS spiral dataset**, a highly non-linear benchmark perfect for testing a network's capacity to learn complex decision boundaries.

* **Total Samples:** 300 (100 samples per class)
* **Classes:** 3 distinct intertwined spirals

---

## Code Structure & Implementation

The repository organizes core neural network components into clean, modular Python classes:

### 1. `Layer_Dense`
Manages the weights ($W$) and biases ($b$). Handles the fundamental linear transformation:
$$Z = X \cdot W + b$$
It computes gradients for both parameters ($\partial L / \partial W$, $\partial L / \partial b$) and passes the gradient down to the previous layer.

### 2. `Activation_ReLU`
Applies element-wise rectification: $\max(0, x)$. Its backward pass selectively zeros out gradients where the input forward state was $\le 0$.

### 3. `Activation_Softmax`
Normalizes raw network outputs (logits) into a stable probability distribution. Features a dynamic mathematical optimization utilizing the **Jacobian matrix** during backpropagation.

### 4. `Loss_CategoricalCrossEntropy`
Quantifies network performance by evaluating the negative log-likelihood of the correct class confidences, paired with standard clipping to prevent numerical instability ($\log(0)$ errors).

---

## Getting Started

### Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install numpy nnfs
