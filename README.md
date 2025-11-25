# 🧠 Brain Tumor Classification using CNN  
A Complete End-to-End Deep Learning Project  
(Exploratory Data Analysis, Model Training, Testing & Evaluation)

---

## 📌 Introduction  
This project focuses on **classifying brain MRI scans** into four tumor categories using a **Convolutional Neural Network (CNN)**. The dataset includes MRI images labeled as:

- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

The goal is to build a deep learning model capable of detecting brain tumors from MRI images with high accuracy.

---

# 🔍 Exploratory Data Analysis (EDA)

## ✔ 1. Class Distribution
- Counted total images in each class of train and test datasets.
- Identified class imbalance (some classes had more images).

## ✔ 2. Sample Images
Displayed several images from each class to visually inspect differences among tumor types.

## ✔ 3. Image Shape & Channel Analysis
- Some images were grayscale (1 channel).  
- Some images were RGB (3 channels).  
- All images were converted to **RGB** for consistency.

## ✔ 4. Image Dimension Distribution
- Checked heights and widths of all images.
- Identified variation → fixed using resizing during preprocessing.

## ✔ 5. Average Image per Class
- Converted images to **RGB**
- Resized to **128×128**
- Computed mean pixel values per class

This helped visually understand structural differences among MRI categories.

---

# 🧠 CNN Model Development

A custom CNN model was built with:

### ✔ Layers Used:
- 2D Convolution layers (Conv2D)
- MaxPooling layers
- BatchNormalization
- Dropout layers (to avoid overfitting)
- Fully connected Dense layers
- Softmax output layer (4 neurons → 4 classes)

### ✔ Loss & Optimizer:
- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy

---

# 🚀 Model Training

### ✔ Data Augmentation Applied:
Used `ImageDataGenerator` to improve generalization:

- Rotation
- Zooming
- Width/height shifting
- Horizontal flips
- Rescaling (1/255)

### ✔ Training Process:
- Trained using augmented training dataset
- Validation done on test dataset

---

# 🎯 Model Evaluation (Final Results)

The model was tested on **1311 MRI images**.

### 📌 **Classification Report**

          precision    recall  f1-score   support

  glioma       0.94      0.73      0.82       300
accuracy                           0.84      1311

### ✔ Key Points:
- Overall accuracy: **84%**
- Strong performance on **No Tumor** and **Pituitary**
- Meningioma misclassified more due to similarity with glioma

---

# 📊 Confusion Matrix

[[219 68 0 13]
[ 9 183 83 31]
[ 3 1 399 2]
[ 1 3 0 296]]

### ✔ Interpretation:
- **Glioma →** often confused with meningioma  
- **Meningioma →** confused with pituitary  
- **No tumor →** highest recall (99%)  
- **Pituitary →** strong accuracy  

---

# 🌡 Heatmap Visualization
A heatmap was plotted using seaborn to visualize the confusion matrix and misclassification patterns.

---

# 🛠 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# 📦 Project Folder Structure
brain_tumor_project.zip
│
├── brain_tumor_prediction.ipynb
└── dataset/
    ├── train/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── pituitary/
    │   └── no_tumor/
    │
    └── test/
        ├── glioma/
        ├── meningioma/
        ├── pituitary/
        └── no_tumor/



