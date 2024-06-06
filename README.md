# FoodVision Model

This repository contains the code and resources for the FoodVision model, a fine-tuned version of the ResNet50 neural network, aimed at classifying images of various Indian foods. The model has been trained and tested on the massive Indian Food Dataset, leveraging transfer learning for improved performance and accuracy.

## Project Overview

FoodVision is designed to help in the classification of 15 different categories of Indian cuisine. By utilizing the pre-trained ResNet50 architecture, this project demonstrates how transfer learning can be applied to a specific domain of food classification. The model has been fine-tuned and optimized for accuracy.

## Features

- **Pre-trained ResNet50 Base Model**: The project uses the ResNet50 model without the top layer as the base for transfer learning.
- **Custom Fine-tuning**: The top layers have been modified and trained on the Indian Food Dataset to enhance model accuracy for specific food categories.
- **Data Preprocessing**: Includes steps for downloading, unzipping, and inspecting the dataset.
- **Image Augmentation**: Utilizes image augmentation techniques to increase the diversity of training data and improve model generalization.
- **Callbacks for Model**: Implements callbacks for monitoring and optimizing model performance during training.
- **Model Training and Evaluation**: Comprehensive training and validation process, including the use of TensorBoard for monitoring.
- **Comparison of Training History**: Compares training history between different model configurations to analyze performance improvements.
- **Streamlit Web App**: A web application built using Streamlit for image classification.

## Dataset

The dataset used for this project is the [Massive Indian Food Dataset](https://www.kaggle.com/datasets/anshulmehtakaggl/themassiveindianfooddataset), which contains a large collection of images categorized into 15 different Indian food classes.

## Installation

To get started with the project, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/foodvision-model.git
cd foodvision-model
pip install -r requirements.txt