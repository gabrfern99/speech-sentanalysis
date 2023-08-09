import pandas as pd
import numpy as np
import librosa
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

data_dir = "TESS Toronto emotional speech set data"
# Load the data and labels
all_wav_files = []
all_labels = []

for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    if os.path.isdir(label_dir):
        for wav_file in os.listdir(label_dir):
            all_wav_files.append(os.path.join(label_dir, wav_file))
            all_labels.append(label)

# Convert list to a pandas dataframe

df = pd.DataFrame({'Filename': all_wav_files, 'Class': all_labels})
print(df[:10])

# MFCC Extraction
def extract_mfcc(wav_file_name, max_pad_len=400):
    try:
        audio, sample_rate = librosa.load(wav_file_name, sr=16000)
        
        # Ensure the audio is mono (single channel)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
        
        # Pad or truncate MFCC sequences to the same length
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0: # If mfccs is shorter than max_pad_len
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:  # If mfccs is longer than max_pad_len
            mfccs = mfccs[:, :max_pad_len]
        
    except Exception as e:
        print("Error encountered while parsing file: ", wav_file_name, e)
        return None

    return mfccs

df['mfcc'] = df['Filename'].apply(extract_mfcc)
# Convert MFCCs to numpy arrays
X = np.array(df.mfcc.tolist())
#X = np.expand_dims(X, axis=2)
y = np.array(df.Class.tolist())

# Encode labels
label_encoder = LabelEncoder()
y = to_categorical(label_encoder.fit_transform(y))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Define the CRNN model
def add_cnn_layers(model, input_shape):
    """Add CNN layers to the model."""
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.3))

def add_lstm_layers(model):
    """Add LSTM layers to the model."""
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.LSTM(64, return_sequences=False))
    model.add(layers.Dropout(0.3))

def add_dense_layers(model, num_classes):
    """Add Dense layers to the model."""
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation="softmax"))

# Define the CRNN model in a structured manner
model = models.Sequential()

add_cnn_layers(model, input_shape=(X_train.shape[1], X_train.shape[2]))
add_lstm_layers(model)
add_dense_layers(model, len(df.Class.unique()))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Show model summary
model.summary()
# Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=8, validation_data=(X_test, y_test))

# Plotting the training and validation accuracy
plt.figure(figsize=(12, 5))

# Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Show reports

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_true, y_pred_classes)
report = classification_report(y_true, y_pred_classes)
print(report)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
