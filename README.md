# CAD Project - Monkeypox Detection

## Overview

This project involves the application of **Machine Learning** and **Deep Learning** techniques for Monkeypox detection using image data. The goal is to preprocess and analyze images, extract features, and classify them as either Monkeypox or Non-Monkeypox.

### Project Team
- **Aliaa Alaa El-Din**
- **Neama Refaat**
- **Shada Gamal**

### Supervised By
Dr. Rana Hossam Eldeen

---

## Dataset

The dataset, containing labeled Monkeypox and Non-Monkeypox images, was sourced from [Kaggle](https://www.kaggle.com). The images were structured and preprocessed into a tabular format using a CSV file with columns for labels, image IDs, and image paths.

---

## Workflow

### Machine Learning Approach

#### 1. Data Preprocessing
- Imported essential libraries: `Numpy`, `Pandas`, `Matplotlib`, `TensorFlow`, `OpenCV`, `Skimage`, `Pywt`, and more.
- Organized and cleaned the dataset, ensuring all images were consistent in shape and resolution.
- Normalized images to a range of [0,1] using MinMax normalization.

#### 2. Feature Extraction
- **Shape Features**:
  - Canny Edge Detection
  - Hough Circle Detection

- **Texture Features**:
  - Gray Level Co-Occurrence Matrix (GLCM)
  - Gray Level Run Length Matrix (GLRLM)

- **Color Features**:
  - Color Moments
  - Histograms

- **Wavelet Features**:
  - Haar wavelet transform coefficients

- **SURF Features**:
  - Keypoint detection and feature extraction using the KAZE algorithm.

#### 3. Feature Selection
- Applied methods such as Forward Selection, Backward Elimination, Recursive Feature Elimination (RFE), and Exhaustive Search.
- Filtered the most relevant features, improving accuracy from **65%** to **74%** using Support Vector Machines (SVM).

---

### Deep Learning Approach

#### Convolutional Neural Network (CNN)
- Developed a small CNN architecture to classify augmented data.
- Observed overfitting: **Training Accuracy: 99%**, **Validation Accuracy: 57%**.

---

## Results and Observations

- **Machine Learning**:
  - Best accuracy achieved: **74%** with texture features.
  - Combining texture, color, and shape features resulted in slightly lower performance (72%).

- **Deep Learning**:
  - Overfitting was a significant challenge, indicating a need for further optimization.

---

## Requirements

Install the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow`
- `opencv-python`
- `scikit-image`
- `scipy`
- `pywt`
- `seaborn`

---
