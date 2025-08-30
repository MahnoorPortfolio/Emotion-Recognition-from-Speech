# Emotion Recognition from Speech

This project is a solution for the Emotion Recognition from Speech task as part of the CodeAlpha Machine Learning Internship.

## Objective

The goal is to build a deep learning model that can classify human emotions from audio recordings. This project involves audio processing, feature extraction, and building a 1D Convolutional Neural Network (CNN).

## Dataset

We used the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset. It contains audio files from 24 professional actors (12 male, 12 female) speaking with 8 different emotions:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Methodology

1.  **Feature Extraction:** We processed 1440 audio files. For each file, we extracted **Mel-Frequency Cepstral Coefficients (MFCCs)**, which are a standard representation of the sonic texture of a sound. The mean of 40 MFCCs over the duration of each clip was used as the feature set.

2.  **Model Architecture:** A **1D Convolutional Neural Network (1D-CNN)** was built using TensorFlow and Keras. This type of network is well-suited for finding patterns in sequential data like our audio features.

3.  **Training:** The model was trained for 50 epochs.

## Results

Our model development went through two main versions:

### Version 1

A baseline 1D-CNN model was established.

-   **Test Accuracy:** `53.82%`
-   **Observation:** The model showed signs of overfitting, where training accuracy (~71%) was much higher than validation accuracy. This indicated that the model was memorizing the training data.

### Version 2 (Improved)

To combat overfitting, we improved the architecture by:
- Adding `BatchNormalization` layers to stabilize learning.
- Increasing the `Dropout` rate to 0.4 for better regularization.
- Increasing the training duration to 100 epochs.

This resulted in a significant performance boost:

-   **Test Accuracy:** `60.42%`

This demonstrates that regularization techniques were effective in helping the model generalize better to new, unseen data.

## How to Run

1.  **Install dependencies**:
    ```
    pip install -r requirements.txt
    ```
2.  **Download Data:** Download the `Audio_Speech_Actors_01-24.zip` from the [RAVDESS dataset website](https://zenodo.org/records/1188976).
3.  **Organize Data:** Unzip the file and place all 24 `Actor_...` folders inside a `data` folder in the main project directory.
4.  **Run the script**:
    ```
    python main.py
    ```
    The script will first process the audio files and create an `emotions.csv`. On subsequent runs, it will skip this step and proceed directly to training the model.
