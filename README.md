# Potato-Disease-Classification-Using-CNN
DATASET : https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset?select=Potato___Late_blight
Overview
This project is an end-to-end deep learning application aimed at detecting potato plant diseases, specifically early blight and late blight, in the agriculture domain. The system allows farmers to use a mobile application to take a picture of their potato plants, and the app will determine if the plant is healthy or infected with one of the two diseases. This solution leverages convolutional neural networks (CNN) for image classification, TensorFlow Lite for model optimization, and a backend built using FastAPI and TF Serving for efficient model serving. The project also involves deployment on Google Cloud Platform (GCP) with Google Cloud Functions.

The project is divided into multiple stages, which include data collection, model building, deployment, and app development. The series of tutorials will guide you through each phase, starting from the basics of data gathering to the deployment of the solution on a mobile app.

Project Components
1. Data Collection & Preprocessing
Collect images of potato plants that are healthy or infected with early blight or late blight.
Clean and preprocess the data using TensorFlow datasets (TFDS) and apply data augmentation techniques (rotation, flipping, contrast adjustments) to ensure a diverse training dataset.
2. Model Building
Use Convolutional Neural Networks (CNNs) to build a model for image classification.
Train the model to detect early blight, late blight, and healthy potato plants.
Export the trained model for deployment.
3. ML Ops with TF Serving
Implement TensorFlow (TF) Serving to manage and serve the trained model.
Create a FastAPI backend server that communicates with TF Serving to perform inference requests.
4. Frontend Development (Website)
Build a web application using ReactJS that allows users to upload images of potato plants.
The application will display whether the plant is healthy or infected, based on the results returned by the backend server.
5. Mobile App Development
Develop a mobile application using React Native.
The mobile app will integrate with the backend server, allowing farmers to upload photos of their plants and get disease predictions.
6. Model Optimization (TensorFlow Lite)
Convert the trained model to TensorFlow Lite (TF Lite) format for optimized performance on mobile devices.
Use model quantization to reduce the size of the model and improve inference speed.
7. Deployment on Google Cloud
Deploy the model and backend server on Google Cloud Platform (GCP).
Use Google Cloud Functions to create a serverless environment for scalable, efficient model inference.
Key Technologies
Deep Learning Framework: TensorFlow, CNN (Convolutional Neural Networks)
Backend Server: FastAPI, TF Serving (for serving models)
Frontend: ReactJS (Web), React Native (Mobile App)
Model Optimization: TensorFlow Lite, Quantization
Cloud Deployment: Google Cloud Platform (GCP), Google Cloud Functions
Prerequisites
Before starting the project, make sure you have the following skills and knowledge:

Python Programming - Basic Python knowledge is required.

Suggested resource: Python Tutorials (CodeBasics)
Deep Learning Basics - Familiarity with neural networks and CNNs.

Suggested resource: Deep Learning Playlist (CodeBasics)
ReactJS and React Native - Basic knowledge of frontend web and mobile app development.

Suggested resource: [ReactJS and React Native Tutorials (Various Sources)].
Cloud Computing (Optional) - Familiarity with Google Cloud Platform and serverless architecture is helpful for deployment.

Installation & Setup
Environment Setup
Install Python dependencies:

bash
Copiar código
pip install tensorflow fastapi uvicorn tensorflow-serving-api
Install Frontend Dependencies (ReactJS):

Install Node.js and NPM (if not installed).
Navigate to the frontend directory and install dependencies:
bash
Copiar código
npm install
Install React Native Dependencies:

Install Node.js and NPM.
Use React Native CLI to set up your development environment:
bash
Copiar código
npx react-native init PotatoDiseaseApp
Data Collection
Gather Images: Collect images of potato plants (healthy, early blight, late blight).
Preprocessing: Clean the data and apply augmentations to increase dataset diversity.
python
Copiar código
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
Model Building
CNN Model: Use TensorFlow to build and train a CNN model for classifying potato plant diseases.
python
Copiar código
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Add more layers here
    tf.keras.layers.Dense(3, activation='softmax')
])
Model Optimization
Convert to TF Lite: Once the model is trained, convert it to TensorFlow Lite format for mobile optimization.
python
Copiar código
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
Deployment on GCP
Upload Model to Google Cloud Storage (GCS).
Set up Google Cloud Functions to create a serverless function to serve the model for inference.
Deploy FastAPI Backend Server and connect it to the cloud functions.

Contributing
If you'd like to contribute to this project, feel free to fork the repository and create a pull request. Contributions can include improvements to the code, documentation, or features.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
AtliQ Agriculture for the inspiration behind this project.
TensorFlow and FastAPI for their robust frameworks and libraries.
React Native and Google Cloud for enabling seamless app development and deployment.
