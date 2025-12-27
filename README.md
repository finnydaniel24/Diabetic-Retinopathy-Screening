👉 Diabetic-Retinopathy-Screening

# 🩺 Diabetic Retinopathy Screening using CNN (ResNet50)

This project focuses on **automated screening of Diabetic Retinopathy (DR)** from retinal fundus images using **Deep Learning**.  
A **Convolutional Neural Network (CNN)** with **Transfer Learning (ResNet50)** is used to classify retinal images into **Normal** and **Diabetic Retinopathy** categories.

---

## 📌 Project Overview

Diabetic Retinopathy is a diabetes complication that can lead to vision loss if not detected early.  
This project aims to assist medical screening by providing an **AI-based binary classification model** that analyzes retinal images and predicts the presence of DR.

---

## 🚀 Features

- Transfer Learning using **ResNet50 (ImageNet weights)**
- Image preprocessing and augmentation using `ImageDataGenerator`
- Binary classification (Normal vs Diabetic Retinopathy)
- Model evaluation using:
  - Accuracy
  - Confusion Matrix
  - F1 Score
  - Classification Report
- Clean and modular Jupyter Notebook workflow

---

## 🧠 Model Architecture

- **Base Model:** ResNet50 (pre-trained on ImageNet)
- **Top Layers:**
  - Global Average Pooling
  - Fully Connected Dense Layer
  - Dropout (regularization)
  - Sigmoid Output Layer
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## 📁 Dataset Structure



dataset/
└── train/
├── 0_normal/
└── 1_diabetic_retinopathy/


> ⚠️ Dataset is **not included** in this repository due to size constraints.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/finnydaniel24/Diabetic-Retinopathy-Screening.git
cd Diabetic-Retinopathy-Screening

###2️⃣ Create virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

###3️⃣ Install dependencies
pip install tensorflow scipy numpy scikit-learn matplotlib pillow jupyter

###4️⃣ Run the notebook
jupyter notebook


###Open Diabetic_Retinopathy_CNN.ipynb and Run All Cells.

📊 Evaluation Metrics

The model is evaluated using:

Confusion Matrix

Precision, Recall, F1-score

Validation Accuracy

Example evaluation code:

from sklearn.metrics import confusion_matrix, f1_score, classification_report

📈 Results

The model successfully learns discriminative features from retinal images.

Transfer Learning significantly improves performance with limited data.

Results demonstrate the feasibility of CNN-based DR screening systems.

Detailed metrics can be found inside the notebook output cells.

🔮 Future Improvements

Multi-class classification (DR severity levels)

Use larger and balanced datasets

Apply Grad-CAM for model interpretability

Deploy using Streamlit or Flask

Integrate into a clinical decision-support system

👨‍💻 Author

Finny Daniel (K Finny Daniel)

MSc Cybersecurity & Data Science

GitHub: https://github.com/finnydaniel24

📜 Disclaimer

This project is intended for educational and research purposes only and should not be used as a medical diagnostic tool.

