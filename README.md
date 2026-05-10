# CNS Project – Novel Work using Co-Teaching Technique

## Project Description

This project is based on/modifies parts of the original RAPIER repository:
https://github.com/XXnormal/RAPIER

Used for educational/research purposes with additional modifications.

In this work, we implemented a novel approach using the **Co-Teaching Technique** to improve intrusion detection performance on noisy and imbalanced datasets.

Only the files modified and used for the novel work have been uploaded in this repository.

---

# Novel Work – Co-Teaching

## Co-Teaching Technique

1. Two MLP neural network models are trained simultaneously.
2. Each model selects low-loss (reliable) samples during training.
3. The selected clean samples are exchanged between both models.
4. Noisy or incorrect samples are gradually ignored using a forget-rate strategy.
5. This improves robustness, reduces overfitting, and enhances generalization.
6. Co-teaching helps improve intrusion detection performance on noisy and imbalanced datasets.

---

# Advantages

1. Better handling of noisy labels
2. Improved attack detection capability
3. Higher recall and model stability
4. More generalized learning compared to single-model training

---

# Dataset Used

We used the **CICIDS Dataset** in PCAP format for experimentation and evaluation.

---

# Performance Before Novel Work

| Metric    | Value              |
| --------- | ------------------ |
| Recall    | 0.7599999999848    |
| Precision | 0.7900207900043654 |
| F1 Score  | 0.7747196687883238 |
| Accuracy  | 0.959818181816436  |

---

# Performance After Novel Work

| Metric    | Value              |
| --------- | ------------------ |
| Recall    | 0.9399999999812    |
| Precision | 0.6216931216848982 |
| F1 Score  | 0.7484076385079009 |
| Accuracy  | 0.9425454545437408 |

---

# How to Run the Project

Run the following command:

```bash
python main.py
```

---

# Research Objective

The objective of this project is to improve malicious traffic/intrusion detection by applying robust learning techniques capable of handling noisy and imbalanced network traffic datasets.

---

# Acknowledgement

This work is inspired by and based on the original RAPIER repository developed by the original authors. Additional modifications and experimental work were performed for educational and research purposes.
