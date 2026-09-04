# ============================================
# NVIDIA CNN - UDACITY SIMULATOR TESTING
# ============================================


# ============================================
# IMPORT LIBRARIES
# ============================================

import socketio
import eventlet
import eventlet.wsgi

from flask import Flask

import numpy as np
import cv2

import base64

from io import BytesIO

from PIL import Image

from tensorflow.keras.models import load_model


# ============================================
# LOAD TRAINED MODEL
# ============================================

model = load_model(
    'nvidia_model.h5'
)

print("NVIDIA CNN model loaded successfully!")


# ============================================
# CREATE SOCKET.IO SERVER
# ============================================

sio = socketio.Server()

app = Flask(__name__)

app = socketio.Middleware(
    sio,
    app
)


# ============================================
# IMAGE PREPROCESSING
# ============================================

def preprocess_image(image):

    # ----------------------------------------
    # Image received from simulator is RGB
    # ----------------------------------------

    # Crop road region
    image = image[60:140, :, :]


    # ----------------------------------------
    # Resize
    # ----------------------------------------

    image = cv2.resize(
        image,
        (200, 66)
    )


    # ----------------------------------------
    # Normalize
    # ----------------------------------------

    image = image / 255.0


    return image


# ============================================
# SEND STEERING + THROTTLE TO SIMULATOR
# ============================================

def send_control(
    steering_angle,
    throttle
):

    sio.emit(
        'steer',
        data={
            'steering_angle':
                str(steering_angle),

            'throttle':
                str(throttle)
        }
    )


# ============================================
# RECEIVE DATA FROM SIMULATOR
# ============================================

@sio.on('telemetry')
def telemetry(
    sid,
    data
):

    if data:

        # ====================================
        # GET IMAGE FROM SIMULATOR
        # ====================================

        image_string = data['image']


        # ====================================
        # DECODE BASE64 IMAGE
        # ====================================

        image = Image.open(
            BytesIO(
                base64.b64decode(
                    image_string
                )
            )
        )


        # ====================================
        # CONVERT IMAGE TO NUMPY ARRAY
        # ====================================

        image = np.asarray(
            image
        )


        # ====================================
        # PREPROCESS IMAGE
        # ====================================

        image = preprocess_image(
            image
        )


        # ====================================
        # ADD BATCH DIMENSION
        # ====================================

        image = np.expand_dims(
            image,
            axis=0
        )


        # ====================================
        # PREDICT STEERING ANGLE
        # ====================================

        steering_angle = model.predict(
            image,
            verbose=0
        )[0][0]


        # ====================================
        # LIMIT STEERING
        # ====================================

        steering_angle = np.clip(
            steering_angle,
            -1.0,
            1.0
        )


        # ====================================
        # THROTTLE
        # ====================================

        throttle = 0.20


        # ====================================
        # SEND CONTROL TO SIMULATOR
        # ====================================

        send_control(
            steering_angle,
            throttle
        )


# ============================================
# WHEN SIMULATOR CONNECTS
# ============================================

@sio.on('connect')
def connect(
    sid,
    environ
):

    print(
        "Connected to Udacity Simulator!"
    )


    # Start with car stopped
    send_control(
        0,
        0
    )


# ============================================
# START SERVER
# ============================================

if __name__ == '__main__':

    print(
        "Starting server on port 4567..."
    )

    eventlet.wsgi.server(
        eventlet.listen(
            ('', 4567)
        ),
        app
    )