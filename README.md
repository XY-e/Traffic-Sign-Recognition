# Traffic Sign Recognition

This project uses **deep learning (CNN)** to classify German traffic signs from the **GTSRB dataset**.  
We train a **custom CNN model** and compare its performance with a **transfer learning approach (MobileNetV2)**.  

---

## Project Description
Traffic sign recognition is an important component of self-driving cars and advanced driver-assistance systems (ADAS).  
The goal of this project is to classify traffic signs into one of **43 categories** using deep learning models.

### Features:
- Image preprocessing (resizing, normalization)
- Data augmentation for robustness
- Training a **custom CNN** from scratch
- Training a **transfer learning model (MobileNetV2)**
- Model evaluation with accuracy, confusion matrix, and classification report

---

## Dataset
- **Dataset:** [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)  
- **Classes:** 43  
- **Training samples:** ~31,368  
- **Validation samples:** ~7,841  

---

##  Model Architectures
1. Custom CNN
- 3 × Conv2D + MaxPooling layers
- Dense layers (128 units + softmax output)
- Optimizer: Adam
- Loss: Categorical Crossentropy
2. MobileNetV2 (Transfer Learning)
- Pretrained MobileNetV2 (ImageNet weights, frozen base)
- GlobalAveragePooling2D + Dense(128)
- Softmax output layer
- Optimizer: Adam

---

##  Training & Evaluation
1. Custom CNN
- Validation Accuracy : 90.09%
2. MobileNetV2
- Validation Accuracy : 45.12%

---

## Installation & Requirements
Install the dependencies before running the notebook:

```bash
pip install tensorflow keras numpy matplotlib seaborn scikit-learn kagglehub
