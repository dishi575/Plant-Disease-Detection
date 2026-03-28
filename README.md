🌿 Plant Disease Detection using CNN

📌 Project Overview

This project focuses on detecting plant diseases from leaf images using **Computer Vision and Deep Learning**. A Convolutional Neural Network (CNN) with **transfer learning (MobileNetV2)** is used to classify plant leaves into healthy and diseased categories.


🎯 Problem Statement

Plant diseases significantly affect crop yield and quality. Early detection through image-based analysis can help farmers take timely action. This project aims to automate disease detection using leaf images.


📂 Dataset

* Dataset used: **PlantVillage Dataset**
* Source: Kaggle (`emmarex/plantdisease`)
* The dataset contains labeled images of healthy and diseased plant leaves.


🧠 Methodology

* Data preprocessing (resizing, normalization)
* Dataset splitting (training & validation)
* Transfer learning using **MobileNetV2**
* Model training on reduced dataset
* Prediction on unseen images


🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib


⚙️ Setup Instructions

1. Clone Repository

```bash
git clone https://github.com/YOUR-USERNAME/plant-disease-detection.git
cd plant-disease-detection
```

2. Install Dependencies

```bash
pip install -r requirements.txt
```


🚀 How to Run

🔹 Train Model

```bash
python src/train.py
```

🔹 Predict Image

```bash
python src/predict.py --image test.jpg
```


📊 Output

* Model predicts the **disease class**
* Outputs **confidence score**


📈 Results

* Achieved good accuracy using transfer learning


🔮 Future Work

* Improve accuracy using full dataset
* Add real-time detection using webcam
* Deploy as a web or mobile application



Dishita Chaturvedi