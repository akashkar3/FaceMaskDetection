# Face Mask Detection Streamlit App

This is a Deep Learning-based Face Mask Detection System that uses a Convolutional Neural Network (CNN) to classify whether a person in an uploaded image is wearing a face mask or not. The trained model is deployed through a Streamlit web application for real-time image classification.

## 🚀 Project Overview

**Objective:** Develop a CNN-based computer vision model for face mask detection.

**Dataset Size:** 7,553+ images

**Model Architecture:** Convolutional Neural Network (CNN)

**Input Shape:** 128 × 128 × 3 RGB images

**Frameworks Used:** TensorFlow, Keras, NumPy, Streamlit

**Deployment:** Interactive Streamlit Web Application

**Output Classes:**

* **Class 0:** No Mask
* **Class 1:** Mask

---

## ✨ Features

* Upload an image (JPG, JPEG, PNG)
* View the uploaded image
* Predict whether a face is wearing a mask
* Display prediction confidence score
* Real-time inference using a trained CNN model
* Simple and interactive Streamlit interface

---

## 📊 Results

✅ Successfully classified Mask and No Mask images

✅ Built an end-to-end Computer Vision pipeline

✅ Deployed model through a Streamlit dashboard

✅ Integrated image upload, preprocessing, prediction, and result visualization

---

## 🛠 Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/akashkar3/FaceMaskDetection.git
cd FaceMaskDetection
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**On Windows**

```bash
venv\Scripts\activate
```

**On Linux / Mac**

```bash
source venv/bin/activate
```

### 4. Install Required Packages

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

After execution, Streamlit will generate a local URL such as:

```text
http://localhost:8501
```

Open the URL in your browser to access the application.

---

## 📁 Project Structure

```text
FaceMaskDetection/
│
├── app.py
├── face_mask_detector_model.h5
├── requirements.txt
├── README.md
└── screenshots/
```

---

## 🧠 Model Details

* **Input Image Size:** 128 × 128 pixels
* **Color Format:** RGB (3 channels)
* **Preprocessing:** Image resizing, RGB conversion, normalization to [0,1]
* **Model Type:** Convolutional Neural Network (CNN)
* **Framework:** TensorFlow / Keras

### Output Classes

* **Class 0:** No Mask
* **Class 1:** Mask

---

## 💻 Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pillow
* Streamlit


