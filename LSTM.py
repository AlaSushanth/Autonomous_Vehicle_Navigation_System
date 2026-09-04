# ============================================
# NVIDIA CNN + LSTM AUTONOMOUS VEHICLE
# NAVIGATION SYSTEM
# ============================================


# ============================================
# HIDE TENSORFLOW WARNINGS
# ============================================

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ============================================
# IMPORT LIBRARIES
# ============================================

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv2D,
    Dropout,
    LSTM,
    TimeDistributed
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import cv2
import matplotlib.pyplot as plt


# ============================================
# LOAD CSV FILE
# ============================================

df = pd.read_csv(
    'driving_log.csv',
    header=None
)


# ============================================
# ADD COLUMN NAMES
# ============================================

df.columns = [
    'center',
    'left',
    'right',
    'steering',
    'throttle',
    'brake',
    'speed'
]


# ============================================
# FIX IMAGE PATHS
# ============================================

for col in ['center', 'left', 'right']:

    df[col] = df[col].apply(
        lambda x: x.split('\\')[-1]
    )

    df[col] = 'IMG/' + df[col]


# ============================================
# FUNCTION TO LOAD AND PREPROCESS IMAGE
# ============================================

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Crop road region
    image = image[60:140, :, :]

    # Resize
    image = cv2.resize(
        image,
        (200, 66)
    )

    # Normalize
    image = image / 255.0

    return image


# ============================================
# CREATE SEQUENCE DATA
# ============================================

sequence_length = 5


X_sequences = []
y_sequences = []


# ============================================
# CREATE CENTER-CAMERA SEQUENCES
# ============================================

for i in range(
    sequence_length - 1,
    len(df)
):

    sequence = []

    valid_sequence = True


    # ----------------------------------------
    # GET 5 CONSECUTIVE CENTER FRAMES
    # ----------------------------------------

    for j in range(
        i - sequence_length + 1,
        i + 1
    ):

        image_path = df.iloc[j]['center']

        image = preprocess_image(
            image_path
        )

        # If image could not be loaded
        if image is None:

            valid_sequence = False

            break

        sequence.append(image)


    # ----------------------------------------
    # SAVE SEQUENCE
    # ----------------------------------------

    if valid_sequence:

        X_sequences.append(
            sequence
        )

        # Steering of LAST frame
        y_sequences.append(
            df.iloc[i]['steering']
        )


# ============================================
# CONVERT TO NUMPY
# ============================================

X_sequences = np.array(
    X_sequences,
    dtype=np.float32
)

y_sequences = np.array(
    y_sequences,
    dtype=np.float32
)


# ============================================
# PRINT SHAPES
# ============================================

print(
    "Center sequence X shape:",
    X_sequences.shape
)

print(
    "Center sequence y shape:",
    y_sequences.shape
)


# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_sequences,
    y_sequences,
    test_size=0.3,
    random_state=42
)


print(
    "X_train shape:",
    X_train.shape
)

print(
    "y_train shape:",
    y_train.shape
)

print(
    "X_test shape:",
    X_test.shape
)

print(
    "y_test shape:",
    y_test.shape
)


# ============================================
# CNN FEATURE EXTRACTOR
# ============================================

cnn_input = Input(
    shape=(66, 200, 3)
)


# ============================================
# NVIDIA CONVOLUTION LAYER 1
# ============================================

x = Conv2D(
    24,
    (5, 5),
    strides=(2, 2),
    activation='relu'
)(cnn_input)


# ============================================
# NVIDIA CONVOLUTION LAYER 2
# ============================================

x = Conv2D(
    36,
    (5, 5),
    strides=(2, 2),
    activation='relu'
)(x)


# ============================================
# NVIDIA CONVOLUTION LAYER 3
# ============================================

x = Conv2D(
    48,
    (5, 5),
    strides=(2, 2),
    activation='relu'
)(x)


# ============================================
# NVIDIA CONVOLUTION LAYER 4
# ============================================

x = Conv2D(
    64,
    (3, 3),
    activation='relu'
)(x)


# ============================================
# NVIDIA CONVOLUTION LAYER 5
# ============================================

x = Conv2D(
    64,
    (3, 3),
    activation='relu'
)(x)


# ============================================
# FLATTEN
# ============================================

x = Flatten()(x)


# ============================================
# CNN MODEL
# ============================================

cnn = Model(
    inputs=cnn_input,
    outputs=x
)


# ============================================
# CNN + LSTM INPUT
# ============================================

sequence_input = Input(
    shape=(
        sequence_length,
        66,
        200,
        3
    )
)


# ============================================
# APPLY CNN TO EACH FRAME
# ============================================

features = TimeDistributed(
    cnn
)(sequence_input)


# ============================================
# LSTM
# ============================================

x = LSTM(
    128,
    return_sequences=False
)(features)


# ============================================
# DENSE LAYER 1
# ============================================

x = Dense(
    100,
    activation='relu'
)(x)


# ============================================
# DROPOUT
# ============================================

x = Dropout(
    0.5
)(x)


# ============================================
# DENSE LAYER 2
# ============================================

x = Dense(
    50,
    activation='relu'
)(x)


# ============================================
# DENSE LAYER 3
# ============================================

x = Dense(
    10,
    activation='relu'
)(x)


# ============================================
# OUTPUT
# ============================================

output = Dense(
    1
)(x)


# ============================================
# FINAL CNN + LSTM MODEL
# ============================================

AVNS = Model(
    inputs=sequence_input,
    outputs=output
)


# ============================================
# COMPILE
# ============================================

AVNS.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss='mse',
    metrics=['mae']
)


# ============================================
# MODEL SUMMARY
# ============================================

AVNS.summary()


# ============================================
# TRAIN
# ============================================

AVNS.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32
)


# ============================================
# PREDICT
# ============================================

predictions = AVNS.predict(
    X_test
)


# ============================================
# RMSE
# ============================================

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions.flatten()
    )
)


# ============================================
# SHOW FIRST 5 SEQUENCES
# ============================================

for i in range(5):

    plt.figure(
        figsize=(8, 4)
    )

    # Show the LAST frame
    plt.imshow(
        X_test[i][-1]
    )

    plt.title(
        f"Predicted: {predictions[i][0]:.3f} | "
        f"Actual: {y_test[i]:.3f}"
    )

    plt.axis("off")

    plt.show()


# ============================================
# PRINT FIRST 5 PREDICTIONS
# ============================================

for i in range(5):

    print("--------------------------------")

    print(
        "Sequence :",
        i + 1
    )

    print(
        "Predicted Steering :",
        predictions[i][0]
    )

    print(
        "Actual Steering    :",
        y_test[i]
    )


# ============================================
# EVALUATE
# ============================================

loss, mae = AVNS.evaluate(
    X_test,
    y_test,
    verbose=0
)


# ============================================
# PERFORMANCE
# ============================================

print(
    "\n==============================="
)

print(
    "CNN + LSTM MODEL PERFORMANCE"
)

print(
    "==============================="
)

print(
    "Loss :",
    round(loss, 4)
)

print(
    "MAE  :",
    round(mae, 4)
)

print(
    "RMSE :",
    round(rmse, 4)
)


# ============================================
# SAVE MODEL
# ============================================

AVNS.save(
    'nvidia_cnn_lstm_model.h5'
)

print(
    "\nModel saved successfully!"
)