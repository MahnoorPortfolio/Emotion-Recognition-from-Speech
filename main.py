import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# Define the path to the dataset and the output CSV
DATA_PATH = './data/'
CSV_PATH = 'emotions.csv'

def extract_features(file_path):
    """Extracts MFCCs from an audio file."""
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_features_csv():
    """Creates a CSV file with extracted features and labels."""
    emotions = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    extracted_data = []
    for actor_folder in os.listdir(DATA_PATH):
        actor_path = os.path.join(DATA_PATH, actor_folder)
        if os.path.isdir(actor_path):
            for file_name in os.listdir(actor_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(actor_path, file_name)
                    try:
                        emotion_code = file_name.split('-')[2]
                        emotion = emotions[emotion_code]
                    except (KeyError, IndexError):
                        continue
                    features = extract_features(file_path)
                    if features is not None:
                        extracted_data.append([features, emotion])
    df = pd.DataFrame(extracted_data, columns=['features', 'emotion'])
    df.to_csv(CSV_PATH, index=False)
    print(f"Successfully processed and saved {len(df)} audio files to {CSV_PATH}.")

def train_model():
    """Loads data, builds, trains, and evaluates the model."""
    # Load the data from CSV
    df = pd.read_csv(CSV_PATH)

    # Convert features from string back to numpy array
    # This is a bit of a hack; a better way is to save to a different format like .npy
    df['features'] = df['features'].apply(lambda x: np.fromstring(x.replace('[', '').replace(']', ''), sep=' '))

    # Separate features and labels
    X = np.array(df['features'].tolist())
    y = np.array(df['emotion'].tolist())

    # Encode the labels
    lb = LabelEncoder()
    y = to_categorical(lb.fit_transform(y))

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape for 1D CNN
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    # Build the improved 1D CNN model
    model = Sequential([
        Conv1D(256, 5, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=5),
        Dropout(0.4),

        Conv1D(128, 5, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=5),
        Dropout(0.4),

        Flatten(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(y_train.shape[1], activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("\n--- Training the model ---")
    model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), verbose=2)

    # Evaluate the model
    print("\n--- Evaluating the model ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy*100:.2f}%')

if __name__ == '__main__':
    # Check if the features CSV already exists
    if not os.path.exists(CSV_PATH):
        print("Feature file not found. Starting feature extraction...")
        create_features_csv()
    else:
        print("Feature file found. Skipping extraction.")
    
    # Train the model
    train_model()
