# Overview
This project involves the classification of pediatric chest X-ray images to detect the presence of pneumonia. It utilizes a convolutional neural network with hypertuning and a step-by-step machine learning pipeline, split between four Jupyter notebooks. The project aims to optimize the model for high recall to minimize false negatives, which are critical in medical applications. The dataset consists of grayscale images of lungs categorized as healthy or pneumonia-affected, with training, validation, and test sets provided.  The training dataset is not included due to its size, but it can be downloaded from kaggle here:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The project includes exploratory data analysis (EDA), model development, evaluation, and visualization of results. The final model is trained for six epochs based on optimal validation performance. The project is structured as follows:


# Notebooks Overview

1. **01_eda.ipynb** - Exploratory Data Analysis (EDA):
    - Visualizes chest X-ray images.
    - Compares healthy lungs to lungs affected by pneumonia.
    - Observes general trends, such as cloudiness in pneumonia-affected lungs.

2. **02_preprocessing.ipynb** - Data Preprocessing:
    - no preprocessing was required as the dataset was already split into test, train, and val and classified sick versus healthy.

3. **03_modeling.ipynb** - Model Development:
    - Implements hyperparameter tuning to optimize convolutional neural networks (CNNs).
    - Focuses on maximizing recall due to the medical implications of false negatives.
    - Trains a CNN model to classify grayscale images of lungs.
    - Saves the best model based on validation performance.

4. **04_evaluation.ipynb** - Model Evaluation:
    - Evaluates model accuracy, recall, and loss on validation and test datasets.
    - Identifies false positives and analyzes them visually.
    - Saves model evaluation metrics and visualizations.

# Setup Instructions



To run the notebooks, ensure you have the required libraries installed.


This project relies on the following libraries and tools:

1. **Libraries:** 
    - numpy
    - matplotlib
    - tensorflow
    - keras-tuner

2. **Tools:**
    - Python 3
    - Jupyter Notebook

# Data Source

The dataset contains chest X-ray images categorized as either healthy or pneumonia-affected. It is structured into training, validation, and test sets, with grayscale images of arbitrary size and dimension. The images are stored in the data directory. The test and validation sets are stored in the data folder. The training dataset is not included in the repository due to its size, but it can be downloaded from kaggle here:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The model is pre-trained and the KERAS save file has the trained model saved which can be loaded and used for predictions or further analysis. This file is saved in the models directory.

# Additional Notes

If you are training the model, make sure to download it and ensure the dataset is placed in the correct directory (data) before running the notebooks.

Results such as accuracy plots, recall graphs, and false positive visualizations are saved in the results directory.

For further questions or support, contact:  
    - Adin Sacho-Tanzer [Adin.SachoTanzer@cooper.edu]  
    - Gabriel Kret [gabriel.kret@cooper.edu]  

# File Structure

project/  
├── README.md # Setup instructions, dependencies, data source  
├── data/  
│ ├── test/ # Test dataset  
│ └── val/ # Validation dataset  
├── notebooks/  
│ ├── 01_eda.ipynb # Exploratory Data Analysis  
│ ├── 02_preprocessing.ipynb # Data cleaning & preparation  
│ ├── 03_modeling.ipynb # Model development & training  
│ └── 04_evaluation.ipynb # Results & interpretation  
├── results/   
│ ├── figures/ # Generated visualizations  
│ ├── models/ # Saved model checkpoints  
│ └── metrics/ # Performance evaluations  
└── docs/  
  └── video/ # Code walkthrough video  
