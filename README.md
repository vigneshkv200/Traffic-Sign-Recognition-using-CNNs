# Traffic Sign Recognition using CNNs (GTSRB Dataset)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Deep Learning project that classifies physical traffic sign images into 43 distinct categories using a custom **Convolutional Neural Network (CNN)** built with TensorFlow/Keras. This pipeline achieves a high-performance **97.15% test accuracy** on the competitive **German Traffic Sign Recognition Benchmark (GTSRB)**.

---

## 📌 Key Learning Objectives & Achievements
* **ROI Extraction & Preprocessing**: Applied ground-truth bounding box crop configurations (`Roi.X1`, `Roi.Y1`, `Roi.X2`, `Roi.Y2`) using OpenCV to eliminate background clutter (roads, sky, trees) and optimize learning boundaries.
* **CNN Design**: Structured a multi-stage feature extraction hierarchy using stacked 2D Convolution, MaxPooling, Dropout, and Dense layers.
* **Generalization**: Controlled overfitting through Dropout regularizations and EarlyStopping callbacks, achieving near-perfect convergence.
* **Model Diagnostics**: Analyzed errors using confusion matrices, training history curves, and visually plotting misclassified samples to comprehend edge cases (extreme shadows, motion blur).

---

## 🛠️ Tech Stack
* **Framework**: TensorFlow / Keras
* **Data Processing**: NumPy, Pandas, OpenCV (`cv2`)
* **Visualization**: Matplotlib, Seaborn
* **Dataset**: GTSRB (German Traffic Sign Recognition Benchmark)

---

## 📐 Model Architecture & Flow

Our custom Sequential CNN pipeline transforms raw pixel dimensions `(32, 32, 3)` into class probabilities via abstract spatial hierarchy:

```
Input (32x32x3) 
   ├── Conv2D (32 filters, 3x3) -> Relu 
   ├── MaxPooling2D (2x2)
   │
   ├── Conv2D (64 filters, 3x3) -> Relu
   ├── MaxPooling2D (2x2)
   │
   ├── Conv2D (128 filters, 3x3) -> Relu
   ├── MaxPooling2D (2x2)
   │
   ├── Flatten (1D Vector: 512)
   ├── Dense (256 Neurons) -> Relu
   ├── Dropout (Rate: 0.5)
   └── Dense (43 Output Nodes) -> Softmax
```

---

## 📈 Results and Metrics
* **Training Accuracy**: `99.39%` (Loss: `0.0207`)
* **Test Accuracy**: `97.15%` (Loss: `0.1452`)
* **Total Samples**: Over 50,000 images.
* **Misclassified Test Items**: Only `360` out of `12,630` evaluations.

---

## 📂 Folder Structure
```
├── dataset/                 # GTSRB Raw & CSV Data
│   ├── Train.csv            # Annotation paths and ROI bounding coordinates
│   ├── Test.csv
│   ├── Train/
│   └── Test/
├── traffic_sign_cnn.keras   # Saved Model weights
├── traffic_sign_pipeline.ipynb # Code Execution Notebook
└── README.md                # Project documentation
```

---

## 🚀 How to Run locally
1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/traffic-sign-recognition-cnn.git
   cd traffic-sign-recognition-cnn
   ```
2. **Install dependencies**:
   ```bash
   pip install tensorflow opencv-python pandas numpy matplotlib seaborn scikit-learn
   ```
3. **Run the Notebook**: Open and execute all cells in `traffic_sign_pipeline.ipynb` to download dependencies, extract the dataset, train, and plot evaluation curves.
```
