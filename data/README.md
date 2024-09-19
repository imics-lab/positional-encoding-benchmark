# Data

This folder utilizes multiple time-series datasets for classification tasks across domains such as human activity recognition (HAR), EEG, ECG, Traffic, device data, and financial data. These datasets are preprocessed and loaded using Python, with deep learning models for time-series classification. Below is a detailed description of the datasets used and the data preprocessing steps.

## Datasets Overview

### 1. **Sleep**
- **Type**: EEG (Electroencephalogram) Data
- **Train Size**: 478,785
- **Test Size**: 90,315
- **Length**: 178 time steps
- **No. of Classes**: 5
- **Description**: The Sleep dataset contains EEG signals used to classify five different stages of sleep. Each instance consists of 178 time steps, representing EEG data over a given period.

### 2. **ElectricDevices**
- **Type**: Device-based Data
- **Train Size**: 8,926
- **Test Size**: 7,711
- **Length**: 96 time steps
- **No. of Classes**: 7
- **Description**: The ElectricDevices dataset represents the electrical signatures of household devices. The goal is to classify the device type using 96 time steps of electrical consumption data.

### 3. **FaceDetection**
- **Type**: Image-based Data (EEG for Face Detection)
- **Train Size**: 5,890
- **Test Size**: 3,524
- **Length**: 62 time steps
- **No. of Classes**: 2
- **Description**: The FaceDetection dataset contains EEG signals recorded during face detection tasks. The objective is to classify EEG signals into two categories: recognizing a face or not. Each sample consists of 62 time steps.

### 4. **MelbournePedestrian**
- **Type**: Traffic Data
- **Train Size**: 1,194
- **Test Size**: 2,439
- **Length**: 24 time steps
- **No. of Classes**: 10
- **Description**: The MelbournePedestrian dataset records sensor data on pedestrian movement in Melbourne. It consists of 24 time steps per instance, representing pedestrian flow, with a 10-class classification task.

### 5. **SharePriceIncrease**
- **Type**: Financial Data
- **Train Size**: 965
- **Test Size**: 965
- **Length**: 60 time steps
- **No. of Classes**: 2
- **Description**: This dataset captures time-series data related to share prices and is used to predict whether a company's share price will increase. Each instance has 60 time steps, representing daily price movements.

### 6. **LSST**
- **Type**: Astronomical Data
- **Train Size**: 2,459
- **Test Size**: 2,466
- **Length**: 36 time steps
- **No. of Classes**: 14
- **Description**: The LSST (Large Synoptic Survey Telescope) dataset contains time-series data from astronomical observations. The task is to classify celestial objects based on their light curves, with 36 time steps per instance and 14 different classes of objects.

### 7. **RacketSports**
- **Type**: Human Activity Recognition (HAR)
- **Train Size**: 151
- **Test Size**: 152
- **Length**: 30 time steps
- **No. of Classes**: 4
- **Description**: This dataset records human movements during various racket sports, such as tennis and badminton. Each instance contains 30 time steps, capturing movement data with a four-class classification task.

### 8. **SelfRegulationSCP1**
- **Type**: EEG (Electroencephalogram) Data
- **Train Size**: 268
- **Test Size**: 293
- **Length**: 896 time steps
- **No. of Classes**: 2
- **Description**: The SelfRegulationSCP1 dataset consists of EEG signals related to self-regulation through slow cortical potentials (SCPs). Each instance contains 896 time steps, and the classification task is binary.

### 9. **UniMiB-SHAR**
- **Type**: Human Activity Recognition (HAR)
- **Train Size**: 4,601
- **Validation Size**: 1,454
- **Test Size**: 1,524
- **Length**: 151 time steps
- **No. of Classes**: 9
- **Description**: The UniMiB-SHAR dataset is used for classifying human activities based on sensor data. It contains training, validation, and test sets with 151 time steps per instance.

### 10. **Leotta_2021**
- **Type**: Human Activity Recognition (HAR)
- **Train Size**: 2,391
- **Validation Size**: 1,167
- **Test Size**: 1,987
- **Length**: 300 time steps
- **No. of Classes**: 18
- **Description**: The Leotta 2021 dataset includes sensor data for various activities and is used for classifying 18 different activities. Each instance consists of 300 time steps.

## Downloading Datasets

1. **Time-Series Classification Datasets**:  
   You can download the majority of the datasets used in this project from the UCR/UEA Time Series Classification repository:  
   [https://timeseriesclassification.com/dataset.php](https://timeseriesclassification.com/dataset.php)

2. **UniMiB-SHAR Dataset**:  
   The UniMiB-SHAR dataset can be downloaded directly from the files provided in the `UniMiB-SHAR` folder of this project.

3. **Leotta_2021 Dataset**:  
   The Leotta 2021 dataset can be downloaded from the files provided in the `Leotta_2021` folder of this project.


## Data Preprocessing Steps

1. **Loading the Data**: 
   - The datasets are mostly loaded from `.arff` files using the `scipy.io` library. In some cases, we load the datasets from `.ts` files.
   
2. **Feature Extraction**: 
   - After loading, features are extracted by separating the target labels from the feature columns.
   
3. **Data Normalization**: 
   - The features are normalized using `StandardScaler` to standardize the data for better model performance.

4. **Reshaping**: 
   - The time-series data is reshaped into a 3D format suitable for deep learning models. Each instance is reshaped to represent samples, time steps, and dimensions.

5. **Splitting the Data**:
   - The datasets are split into training, validation, and test sets. For some datasets, the test data is further split to create a validation set for model evaluation.
   
6. **Tensor Conversion**: 
   - The preprocessed data is converted into PyTorch tensors to be fed into deep learning models.

7. **Batching**: 
   - The data is batched using `DataLoader` to handle efficient loading during model training.

## Usage

1. **Prerequisites**:
   - Install the required Python libraries: `numpy`, `pandas`, `scikit-learn`, `scipy`, and `torch`.
   
   ```bash
   pip install numpy pandas scikit-learn scipy torch
