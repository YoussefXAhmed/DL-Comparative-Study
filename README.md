# Deep Learning Comparative Study

## Project Overview
This project compares:

1. DL Feature Extraction + Traditional ML Classifier (SVM)
2. End-to-End Deep Learning Classification

using pre-trained CNN architectures.

---

## Datasets
- Chest X-Ray Pneumonia Dataset
- Brain MRI Tumor Dataset

---

## Models Used
- ResNet50
- EfficientNet-B0

---

## Implemented Approaches

### Approach 1
Pre-trained CNN as feature extractor + SVM classifier.

### Approach 2
End-to-End CNN fine-tuning and classification.

---

## Files

- main_resnet50_chestxray.py
- bonus_efficientnet_chestxray.py
- bonus_brainmri_dataset.py

---

## Requirements

```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy
