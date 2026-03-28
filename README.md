# LeatherVision ->  Real vs Fake Leather Classifier

A Computer Vision command-line tool that classifies whether a given leather image is **Real or Fake** using texture-based analysis and Machine Learning (SVM).
Built as part of the **Bring Your Own Project (BYOP)** component for the flipped course at VIT Bhopal University.

---

## What This Project Does

Leather textures have distinct microscopic patterns. Real leather tends to have **irregular, natural grain**, while fake leather shows **uniform, repeated patterns**.

This tool analyzes those differences using 3 key steps:

1. **Preprocessing** — Resize and convert the image to grayscale
2. **LBP (Local Binary Patterns)** — Extracts texture features from pixel neighborhoods
3. **SVM Classification** — Uses a trained Support Vector Machine to classify the texture as Real or Fake

The result is a fast and lightweight texture-based classifier that works directly from the command line.

---

## Computer Vision & ML Concepts Covered

| Concept                      | Where Used                              |
| ---------------------------- | --------------------------------------- |
| Grayscale Conversion         | Preprocessing before feature extraction |
| Local Binary Patterns (LBP)  | Texture feature extraction              |
| Histogram Normalization      | Feature scaling for classification      |
| Support Vector Machine (SVM) | Binary classification model             |
| Train-Test Split             | Model evaluation                        |
| NumPy Arrays                 | Feature storage and manipulation        |

---

## Requirements

* Python **3.8 or higher**
* pip (comes with Python)
* Terminal / Command Prompt

---

## Environment Setup

### Step 1 — Check Python installation

```bash
python --version
```

---

### Step 2 — Place your project files

Make sure you have:

```
leather-classifier/
├── leather_classification.py
├── requirements.txt
└── dataset/
    ├── real/
    └── fake/
```

---

### Step 3 — Create virtual environment (recommended)

**Mac / Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

---

### Step 4 — Install dependencies

```bash
pip install opencv-python numpy scikit-learn joblib
```

---

## Dataset Structure

Your dataset must follow this format:

```
dataset/
├── real/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── fake/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

* `real/` → images of genuine leather
* `fake/` → images of synthetic leather

---

## Running the Project

### 1. Train the Model

```bash
python leather_classification.py train --dataset dataset
```

### Expected Output

```
[INFO] Loading dataset...
[INFO] Training model...

[RESULTS]
Accuracy: 0.85
...classification report...

[INFO] Model saved to leather_model.pkl
```

---

### 2. Predict an Image

```bash
python leather_classification.py predict --image test.jpg
```

### Expected Output

```
[RESULT]
Prediction : Real Leather
Confidence : 0.92
```

---

## How It Works Internally

### 1. LBP Feature Extraction

* Compares each pixel with its 8 neighbors
* Generates a binary pattern
* Converts pattern into decimal value
* Builds a histogram of texture patterns

### 2. Feature Vector

* 256-length normalized histogram
* Represents texture distribution

### 3. SVM Model

* Linear kernel used
* Learns boundary between real and fake textures

---

## Project Structure

```
leather-classifier/
├── leather_classification.py   # Main script
├── leather_model.pkl           # Saved trained model
├── dataset/                    # Training data
└── README.md                   # Documentation
```

---

## Key Functions

**`extract_lbp()`**
→ Extracts texture features using Local Binary Patterns

**`load_data()`**
→ Loads images and assigns labels

**`train()`**
→ Trains SVM model and saves it

**`predict()`**
→ Classifies a new image

---

## Troubleshooting

**Model not found error**

```
[ERROR] Train model first!
```

→ Run training before prediction

---

**Invalid image path**

```
[ERROR] Invalid image path
```

→ Check file name and location

---

**Low accuracy**
→ Improve dataset quality:

* Add more images
* Ensure balanced real vs fake samples
* Use higher resolution images

---

**Slow performance**
→ LBP is pixel-wise — reduce image size or dataset size

---

## How to Verify on Google Colab

```python
# Install dependencies
!pip install opencv-python-headless numpy scikit-learn joblib

# Upload files
from google.colab import files
uploaded = files.upload()

# Train
!python leather_classification.py train --dataset dataset

# Predict
!python leather_classification.py predict --image test.jpg
```

---

## Author

**Name:** Raunak Sachdeva 

**Registration No:** 23BAI11296

**Course:** Computer Vision

**Institution:** VIT Bhopal University, Madhya Pradesh — 466114

**Submission:** BYOP — VITyarthi Portal

**Date:** March 2026
