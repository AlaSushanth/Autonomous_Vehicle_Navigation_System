# ============================================
# NVIDIA AUTONOMOUS VEHICLE NAVIGATION SYSTEM
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

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    Dropout
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

    df[col] = (
        'IMG/'
        + df[col]
    )


# ============================================
# CREATE EMPTY LISTS
# ============================================

images = []

steerings = []


# ============================================
# LOAD ALL 3 CAMERA IMAGES
# ============================================

for i in range(len(df)):


    # ========================================
    # READ IMAGES
    # ========================================

    center_img = cv2.imread(
        df.iloc[i]['center']
    )

    left_img = cv2.imread(
        df.iloc[i]['left']
    )

    right_img = cv2.imread(
        df.iloc[i]['right']
    )

    steering = df.iloc[i]['steering']


    # ========================================
    # CONVERT BGR TO RGB
    # ========================================

    center_img = cv2.cvtColor(
        center_img,
        cv2.COLOR_BGR2RGB
    )

    left_img = cv2.cvtColor(
        left_img,
        cv2.COLOR_BGR2RGB
    )

    right_img = cv2.cvtColor(
        right_img,
        cv2.COLOR_BGR2RGB
    )


    # ========================================
    # CROP ROAD REGION
    # ========================================

    center_img = center_img[60:140,:,:]

    left_img = left_img[60:140,:,:]

    right_img = right_img[60:140,:,:]


    # ========================================
    # RESIZE IMAGES
    # ========================================

    center_img = cv2.resize(
        center_img,
        (200,66)
    )

    left_img = cv2.resize(
        left_img,
        (200,66)
    )

    right_img = cv2.resize(
        right_img,
        (200,66)
    )


    # ========================================
    # CENTER IMAGE
    # ========================================

    images.append(center_img)

    steerings.append(steering)


    # ========================================
    # LEFT IMAGE
    # ========================================

    images.append(left_img)

    steerings.append(steering + 0.2)


    # ========================================
    # RIGHT IMAGE
    # ========================================

    images.append(right_img)

    steerings.append(steering - 0.2)


# ============================================
# CONVERT TO NUMPY ARRAYS
# ============================================

X = np.array(images)

y = np.array(steerings)


# ============================================
# CHECK SHAPES
# ============================================

print(X.shape)

print(y.shape)


# ============================================
# NORMALIZATION
# ============================================

X = X / 255.0


# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


print(X_train.shape)

print(y_train.shape)


# ============================================
# NVIDIA MODEL
# ============================================

AVNS = Sequential()


# ============================================
# INPUT LAYER
# ============================================

AVNS.add(
    tf.keras.Input(
        shape=(66,200,3)
    )
)


# ============================================
# NVIDIA CONVOLUTION LAYER 1
# ============================================

AVNS.add(
    Conv2D(
        24,
        (5,5),
        strides=(2,2),
        activation='relu'
    )
)


# ============================================
# NVIDIA CONVOLUTION LAYER 2
# ============================================

AVNS.add(
    Conv2D(
        36,
        (5,5),
        strides=(2,2),
        activation='relu'
    )
)


# ============================================
# NVIDIA CONVOLUTION LAYER 3
# ============================================

AVNS.add(
    Conv2D(
        48,
        (5,5),
        strides=(2,2),
        activation='relu'
    )
)


# ============================================
# NVIDIA CONVOLUTION LAYER 4
# ============================================

AVNS.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)


# ============================================
# NVIDIA CONVOLUTION LAYER 5
# ============================================

AVNS.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)


# ============================================
# FLATTEN
# ============================================

AVNS.add(
    Flatten()
)


# ============================================
# DENSE LAYER 1
# ============================================

AVNS.add(
    Dense(
        100,
        activation='relu'
    )
)


# ============================================
# DROPOUT
# ============================================

AVNS.add(
    Dropout(0.5)
)


# ============================================
# DENSE LAYER 2
# ============================================

AVNS.add(
    Dense(
        50,
        activation='relu'
    )
)


# ============================================
# DENSE LAYER 3
# ============================================

AVNS.add(
    Dense(
        10,
        activation='relu'
    )
)


# ============================================
# OUTPUT LAYER
# ============================================

AVNS.add(
    Dense(1)
)


# ============================================
# COMPILE MODEL
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
# TRAIN MODEL
# ============================================

AVNS.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32
)


# ============================================
# PREDICT ALL TEST IMAGES (ONLY ONCE)
# ============================================

predictions = AVNS.predict(X_test)


# ============================================
# CALCULATE RMSE
# ============================================

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions.flatten()
    )
)


# ============================================
# SHOW FIRST 5 TEST IMAGES
# ============================================

for i in range(5):

    plt.figure(figsize=(8,4))

    plt.imshow(X_test[i])

    plt.title(
        f"Predicted: {predictions[i][0]:.3f} | Actual: {y_test[i]:.3f}"
    )

    plt.axis("off")

    plt.show()


# ============================================
# PRINT FIRST 5 PREDICTIONS
# ============================================

for i in range(5):

    print("--------------------------------")

    print("Image :", i+1)

    print("Predicted Steering :", predictions[i][0])

    print("Actual Steering    :", y_test[i])


# ============================================
# EVALUATE MODEL
# ============================================

loss, mae = AVNS.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n===============================")
print("MODEL PERFORMANCE")
print("===============================")

print("Loss :", round(loss,4))
print("MAE  :", round(mae,4))
print("RMSE :", round(rmse,4))

# ============================================
# SAVE MODEL
# ============================================

AVNS.save('nvidia_model.h5')