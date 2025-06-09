# NYCU-WMN
NYCU 無線多媒體網路 Wireless Multimedia Networks 2024 Fall Semester
---

## FC-AE for Wireless Throughput Prediction

This project applies a **two-stage training** strategy using a Fully-Connected Autoencoder (FC-AE) to predict wireless throughput. It includes:

- **Lab 1: Pretraining Phase**
- **Lab 2: Fine-tuning Phase**

---

## Lab Descriptions

### Lab 1: Pretraining Phase

In this stage, we train a Fully-Connected Autoencoder (FC-AE) model using a larger and more general dataset. The goal is to enable the model to learn meaningful feature representations and capture complex nonlinear relationships between input features and the output. After training, the model's generalized weights are saved for downstream tasks.

### Lab 2: Fine-tuning Phase

This stage performs fine-tuning on domain-specific data (e.g., a specific RU site). Selected layers such as encoders or residual blocks are frozen to retain generalized knowledge, while only the task-specific layers (e.g., decoders) are updated. Data augmentation techniques such as **Gaussian noise** and **random masking** are applied to enhance robustness, and **normalized MSE** is used as the loss metric to handle varying scales in the target values.

---
