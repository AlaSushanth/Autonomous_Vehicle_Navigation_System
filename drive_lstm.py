# ============================================
# NVIDIA CNN + LSTM
# UDACITY SIMULATOR TESTING
# 30 SECOND SAFE TEST
# ============================================

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import base64
import time

from collections import deque

import numpy as np
import cv2
import tensorflow as tf

import socketio
import eventlet
import eventlet.wsgi


# ============================================
# SETTINGS
# ============================================

MODEL_PATH = 'nvidia_cnn_lstm_model.h5'

SEQUENCE_LENGTH = 5

# Start LOW for testing
BASE_THROTTLE = 0.12

# Maximum steering allowed initially
MAX_STEERING = 0.60

# Steering smoothing
SMOOTHING_ALPHA = 0.40

# Test for 30 seconds
TEST_DURATION = 60


# ============================================
# LOAD MODEL
# ============================================

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "CNN + LSTM model loaded successfully!"
)

print(
    "Model input shape:",
    model.input_shape
)


# ============================================
# FRAME BUFFER
# ============================================

frame_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)


# ============================================
# STEERING HISTORY
# ============================================

previous_steering = 0.0


# ============================================
# TEST START TIME
# ============================================

test_start_time = None


# ============================================
# SOCKET.IO
# ============================================

sio = socketio.Server()

app = socketio.WSGIApp(
    sio
)


# ============================================
# PREPROCESS IMAGE
# ============================================

def preprocess_image(image):

    # BGR → RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Same crop as training
    image = image[
        60:140,
        :,
        :
    ]

    # Same resize as training
    image = cv2.resize(
        image,
        (200, 66)
    )

    # Same normalization as training
    image = image / 255.0

    return image


# ============================================
# SEND CONTROL
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
# CONNECT
# ============================================

@sio.on('connect')
def connect(
    sid,
    environ
):

    global test_start_time
    global previous_steering

    print()
    print(
        "================================"
    )

    print(
        "Connected to Udacity Simulator!"
    )

    print(
        "Starting 30 second test..."
    )

    print(
        "================================"
    )

    print()

    # Reset everything
    frame_buffer.clear()

    previous_steering = 0.0

    test_start_time = time.time()

    # Start stopped
    send_control(
        0.0,
        0.0
    )


# ============================================
# DISCONNECT
# ============================================

@sio.on('disconnect')
def disconnect(
    sid
):

    print(
        "Disconnected from simulator."
    )


# ============================================
# TELEMETRY
# ============================================

@sio.on('telemetry')
def telemetry(
    sid,
    data
):

    global previous_steering

    try:

        # ====================================
        # CHECK 30 SECOND LIMIT
        # ====================================

        if test_start_time is not None:

            elapsed_time = (
                time.time()
                -
                test_start_time
            )

            if elapsed_time >= TEST_DURATION:

                print()
                print(
                    "================================"
                )

                print(
                    "30 SECOND TEST COMPLETED!"
                )

                print(
                    "Stopping car."
                )

                print(
                    "================================"
                )

                send_control(
                    0.0,
                    0.0
                )

                return


        # ====================================
        # GET IMAGE
        # ====================================

        image_string = data['image']


        # ====================================
        # DECODE BASE64
        # ====================================

        image_bytes = base64.b64decode(
            image_string
        )


        # ====================================
        # NUMPY
        # ====================================

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )


        # ====================================
        # DECODE IMAGE
        # ====================================

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if image is None:

            print(
                "Image decoding failed."
            )

            return


        # ====================================
        # PREPROCESS
        # ====================================

        image = preprocess_image(
            image
        )


        # ====================================
        # ADD FRAME
        # ====================================

        frame_buffer.append(
            image
        )


        # ====================================
        # WAIT FOR 5 FRAMES
        # ====================================

        if len(frame_buffer) < SEQUENCE_LENGTH:

            print(
                "Collecting frames:",
                len(frame_buffer),
                "/",
                SEQUENCE_LENGTH
            )

            send_control(
                0.0,
                0.0
            )

            return


        # ====================================
        # CREATE SEQUENCE
        # ====================================

        sequence = np.array(
            frame_buffer,
            dtype=np.float32
        )


        # Shape:
        #
        # (5, 66, 200, 3)


        # ====================================
        # ADD BATCH DIMENSION
        # ====================================

        sequence = np.expand_dims(
            sequence,
            axis=0
        )


        # Shape:
        #
        # (1, 5, 66, 200, 3)


        # ====================================
        # MODEL PREDICTION
        # ====================================

        prediction = model.predict(
            sequence,
            verbose=0
        )


        # ====================================
        # RAW STEERING
        # ====================================

        raw_steering = float(
            prediction[0][0]
        )


        # ====================================
        # LIMIT RAW STEERING
        # ====================================

        raw_steering = np.clip(
            raw_steering,
            -MAX_STEERING,
            MAX_STEERING
        )


        # ====================================
        # SMOOTH STEERING
        # ====================================

        steering = (
            SMOOTHING_ALPHA
            *
            raw_steering
        ) + (
            (1 - SMOOTHING_ALPHA)
            *
            previous_steering
        )


        # ====================================
        # SAVE PREVIOUS STEERING
        # ====================================

        previous_steering = steering


        # ====================================
        # ADAPTIVE THROTTLE
        # ====================================

        steering_magnitude = abs(
            steering
        )


        if steering_magnitude > 0.45:

            throttle = 0.07

        elif steering_magnitude > 0.30:

            throttle = 0.09

        else:

            throttle = BASE_THROTTLE


        # ====================================
        # SEND CONTROL
        # ====================================

        send_control(
            steering,
            throttle
        )


        # ====================================
        # TIME
        # ====================================

        elapsed_time = (
            time.time()
            -
            test_start_time
        )


        # ====================================
        # PRINT
        # ====================================

        print(
            f"Time: {elapsed_time:5.1f}s | "
            f"Raw: {raw_steering: .3f} | "
            f"Smooth: {steering: .3f} | "
            f"Throttle: {throttle:.2f}"
        )


    except Exception as e:

        print(
            "Telemetry error:",
            e
        )


# ============================================
# START SERVER
# ============================================

if __name__ == '__main__':

    print()
    print(
        "Starting CNN + LSTM server..."
    )

    print(
        "Port: 4567"
    )

    print(
        "Test duration: 30 seconds"
    )

    print()

    eventlet.wsgi.server(
        eventlet.listen(
            ('', 4567)
        ),
        app
    )