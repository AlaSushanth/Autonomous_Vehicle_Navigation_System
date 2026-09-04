# Autonomous Vehicle Navigation System

An end-to-end deep learning project that predicts steering angles from camera images to enable autonomous driving in the Udacity self-driving car simulator.

The project explores two approaches:

1. **NVIDIA-style Convolutional Neural Network (CNN)**
2. **CNN + Long Short-Term Memory (LSTM)**

The CNN learns visual features from individual road images, while the CNN + LSTM model processes a sequence of frames to learn temporal information from the vehicle's recent driving history.

---

## Project Overview

Autonomous vehicle navigation requires a model to understand the road and predict how the vehicle should steer.

In this project, the model receives camera images from the simulator and predicts a steering angle.

```text
Camera Image
     ↓
Image Preprocessing
     ↓
CNN Feature Extraction
     ↓
Steering Angle Prediction
     ↓
Vehicle Steering

For the CNN + LSTM approach:

Sequence of Camera Frames
     ↓
CNN Feature Extraction
     ↓
LSTM Temporal Learning
     ↓
Steering Angle Prediction
     ↓
Vehicle Steering

The project was developed using Python, TensorFlow, OpenCV, and the Udacity self-driving car simulator.

Objectives

Build an end-to-end autonomous driving system.

Understand how CNNs extract visual features from road images.

Explore the use of LSTM networks for sequential driving data.

Train models to predict steering angles.

Test trained models in a driving simulator.

Understand the challenges of autonomous driving, including steering instability and error accumulation.

Improve the system through better data collection and model development.

Models
1. NVIDIA-style CNN

The first model is based on the NVIDIA-style architecture used for end-to-end learning for self-driving cars.

The model learns visual patterns such as:

Road boundaries

Lane markings

Curves

Vehicle position

Road direction

Architecture
Input Image
(66 × 200 × 3)
       ↓
Conv2D (24 filters, 5×5, stride 2)
       ↓
Conv2D (36 filters, 5×5, stride 2)
       ↓
Conv2D (48 filters, 5×5, stride 2)
       ↓
Conv2D (64 filters, 3×3)
       ↓
Conv2D (64 filters, 3×3)
       ↓
Flatten
       ↓
Dense (100)
       ↓
Dropout (0.5)
       ↓
Dense (50)
       ↓
Dense (10)
       ↓
Dense (1)
       ↓
Steering Angle

The model uses Mean Squared Error (MSE) as the loss function and Adam as the optimizer.

2. CNN + LSTM

The second model extends the CNN by adding an LSTM layer.

Instead of processing only one image, the model processes a sequence of consecutive frames.

For example:

Frame 1 → Frame 2 → Frame 3 → Frame 4 → Frame 5

Each frame is passed through the CNN to extract visual features.

The LSTM then learns temporal information from these features.

5 Consecutive Frames
        ↓
CNN Feature Extraction
        ↓
Sequence of Feature Vectors
        ↓
LSTM (128 units)
        ↓
Dense (100)
        ↓
Dropout (0.5)
        ↓
Dense (50)
        ↓
Dense (10)
        ↓
Dense (1)
        ↓
Steering Angle
Why use LSTM?

A CNN mainly learns from the current image.

An LSTM can use information from previous frames to understand how the vehicle is moving over time.

For example:

Previous frames show the car moving left
                ↓
Current frame shows the car near the center
                ↓
LSTM uses temporal information
                ↓
Predicts a suitable steering angle

This makes CNN + LSTM an interesting approach for studying sequential decision-making in autonomous driving.

Dataset

The project uses driving data recorded from the Udacity self-driving car simulator.

The dataset contains:

Center camera images

Left camera images

Right camera images

Steering angles

Throttle values

Brake values

Vehicle speed

The driving data is stored in:

driving_log.csv

The images are stored in:

IMG/
Dataset structure
AVNS/
│
├── driving_log.csv
│
└── IMG/
    ├── center_....jpg
    ├── left_....jpg
    └── right_....jpg

The CSV file contains the image paths and corresponding driving measurements.

Data Preprocessing

The images are preprocessed before being given to the model.

Preprocessing steps

Read the image using OpenCV.

Convert the image from BGR to RGB.

Crop the image to focus on the road region.

Resize the image to:

66 × 200 × 3

Normalize pixel values to the range:

0 to 1
Why preprocessing is important

The original simulator image contains information that may not be necessary for steering prediction.

Cropping helps the model focus more on the road.

Resizing reduces computational cost.

Normalization helps the neural network train more effectively.

CNN + LSTM Sequence Preparation

For the CNN + LSTM model, consecutive center-camera frames are grouped into sequences.

The current implementation uses:

sequence_length = 5

This means the model receives five consecutive frames and predicts the steering angle of the last frame.

Frame 1
Frame 2
Frame 3
Frame 4
Frame 5
   ↓
Predict steering angle of Frame 5

The input shape is:

(number_of_sequences, 5, 66, 200, 3)

The output is:

(number_of_sequences, 1)
Project Files
AVNS/
│
├── main.py
│   └── Training code for the NVIDIA-style CNN
│
├── LSTM.py
│   └── Training code for the CNN + LSTM model
│
├── drive.py
│   └── Simulator testing code for the CNN model
│
├── drive_lstm.py
│   └── Simulator testing code for the CNN + LSTM model
│
├── driving_log.csv
│   └── Recorded driving data
│
├── IMG/
│   └── Recorded camera images
│
├── nvidia_model.h5
│   └── Trained CNN model
│
├── nvidia_cnn_lstm_model.h5
│   └── Trained CNN + LSTM model
│
├── requirements.txt
│   └── Required Python libraries
│
├── .gitignore
│   └── Files excluded from GitHub
│
└── README.md
    └── Project documentation
Technologies Used

Python

TensorFlow

Keras

NumPy

Pandas

OpenCV

Scikit-learn

Matplotlib

Udacity Self-Driving Car Simulator

VS Code

Git & GitHub

Installation
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/AVNS.git
2. Navigate to the project folder
cd AVNS
3. Create a virtual environment
python -m venv venv
4. Activate the virtual environment
Windows
venv\Scripts\activate
5. Install dependencies
pip install -r requirements.txt
Training the CNN Model

The CNN training code is available in:

main.py

Run:

python main.py

The training process:

Load driving_log.csv
        ↓
Load images
        ↓
Preprocess images
        ↓
Train-test split
        ↓
Train NVIDIA CNN
        ↓
Evaluate model
        ↓
Save trained model

The trained model is saved as:

nvidia_model.h5
Training the CNN + LSTM Model

The CNN + LSTM training code is available in:

LSTM.py

Run:

python LSTM.py

The training process:

Load driving_log.csv
        ↓
Load center-camera images
        ↓
Create consecutive frame sequences
        ↓
Preprocess images
        ↓
Train-test split
        ↓
CNN feature extraction
        ↓
LSTM temporal learning
        ↓
Steering prediction
        ↓
Evaluate model
        ↓
Save trained model

The trained model is saved as:

nvidia_cnn_lstm_model.h5
Testing in the Udacity Simulator

The trained models can be tested using the Udacity self-driving car simulator.

CNN model

Run:

python drive.py
CNN + LSTM model

Run:

python drive_lstm.py

The testing script:

Loads the trained model.

Connects to the simulator.

Receives camera images.

Preprocesses the images.

Predicts the steering angle.

Sends the steering command back to the simulator.

Simulator Camera
       ↓
Python Testing Script
       ↓
Image Preprocessing
       ↓
Trained Model
       ↓
Predicted Steering
       ↓
Simulator Vehicle
Simulator Testing

The project was tested using the Udacity self-driving car simulator.

The vehicle is controlled using predictions from the trained model.

The simulator provides a visual environment where the model can be evaluated under continuous driving conditions.

Testing observations

The model was able to drive for a period of time in the simulator.

However, longer autonomous driving revealed challenges such as:

Steering instability

Difficulty handling certain curves

Gradual drifting from the center of the road

Incorrect steering predictions in unfamiliar situations

Error accumulation during continuous driving

These observations motivated further dataset collection and model improvement.

Evaluation Metrics

Since the model predicts a continuous steering angle, this is a regression problem.

The project uses:

Mean Squared Error (MSE)

MSE measures the average squared difference between the actual and predicted steering angles.

MSE = Average of (Actual Steering - Predicted Steering)²
Mean Absolute Error (MAE)

MAE measures the average absolute difference between the actual and predicted steering angles.

MAE = Average of |Actual Steering - Predicted Steering|
Root Mean Squared Error (RMSE)

RMSE is the square root of MSE.

RMSE = √MSE

These metrics help evaluate how closely the model predicts the steering angle.

However, good offline regression metrics do not always guarantee stable autonomous driving.

Limitations
1. Limited dataset diversity

The model learns from the driving situations present in the recorded dataset.

If the dataset does not contain enough examples of different road positions, curves, and recovery situations, the model may struggle when it encounters those situations during autonomous driving.

2. Steering instability

The vehicle may sometimes predict a steering angle that is too large.

This can cause the car to turn more than necessary and move away from the center of the road.

3. Error accumulation

A small steering error can cause the vehicle to move away from the correct path.

The next camera image may then look different from the training data.

This can lead to another incorrect prediction.

Small Steering Error
        ↓
Vehicle Drifts
        ↓
Camera View Changes
        ↓
Incorrect Prediction
        ↓
Larger Drift
        ↓
Possible Off-road Driving
4. Limited recovery examples

If the training data mainly contains normal centered driving, the model may not learn how to recover when the vehicle moves too far left or right.

Recovery data is important for teaching the model how to return to the correct path.

5. Generalization limitations

The model is trained and tested in a simulator environment.

Its performance may not generalize directly to different tracks, lighting conditions, road layouts, or real-world driving situations.

6. Computational limitations

CNN + LSTM requires more computation than a simple CNN because it processes multiple frames and includes a recurrent layer.

This may affect training time and real-time inference performance.

7. Offline evaluation does not fully represent driving performance

A model may achieve reasonable validation results but still perform poorly during continuous autonomous driving.

This is because offline evaluation uses recorded test data, while simulator testing involves the model's own predictions influencing future camera observations.

Future Improvements
1. Collect more diverse driving data

Generate additional training data from the simulator.

The dataset should include:

Straight roads

Left curves

Right curves

Sharp turns

Different road positions

Different steering angles

Recovery situations

The goal is not only to increase the dataset size, but also to improve the diversity of driving situations.

2. Add recovery data

Deliberately place the vehicle slightly away from the center of the road and record how to steer it back.

This can help the model learn corrective steering behavior.

3. Improve data balancing

If most training samples have steering angles close to zero, the model may become biased toward predicting straight driving.

Balancing the steering-angle distribution can help the model learn turns more effectively.

4. Use data augmentation

Possible augmentation techniques include:

Brightness changes

Horizontal flipping

Small image translations

Shadow simulation

Different lighting conditions

These techniques can help improve generalization.

5. Experiment with sequence length

The current CNN + LSTM model uses five consecutive frames.

Future experiments can compare different sequence lengths to study how much temporal information is useful for steering prediction.

6. Improve the CNN + LSTM architecture

Possible improvements include:

Adding Batch Normalization

Experimenting with different LSTM units

Adjusting dropout

Tuning learning rate

Experimenting with different dense-layer sizes

Comparing different CNN architectures

7. Use better training strategies

Future improvements may include:

Learning-rate scheduling

Early stopping

Model checkpointing

Hyperparameter tuning

Data shuffling

Cross-validation where appropriate

8. Compare CNN and CNN + LSTM more systematically

The two models can be compared using:

Validation loss

MAE

RMSE

Simulator driving stability

Ability to handle curves

Ability to recover from small deviations

Duration of stable autonomous driving

This can help determine whether temporal information improves the driving performance.

9. Improve simulator testing

Future testing can include:

Longer continuous driving sessions

Multiple complete laps

Different starting positions

Different driving conditions

More systematic evaluation of steering stability

10. Explore advanced autonomous driving approaches

Future work could explore:

Transfer learning

More advanced CNN architectures

Attention mechanisms

Recurrent neural networks

Reinforcement learning

Behavioral cloning improvements

End-to-end autonomous driving with larger datasets

Learning Outcomes

Through this project, I gained practical experience in:

Deep learning for computer vision

CNN architecture design

LSTM-based sequence modeling

Image preprocessing

Regression problems

Model training and evaluation

Simulator-based testing

Data collection for autonomous driving

Understanding error accumulation

Debugging Python and TensorFlow environments

Building and testing an end-to-end ML pipeline

Conclusion

This project demonstrates an end-to-end autonomous vehicle navigation system using deep learning.

The NVIDIA-style CNN provides a baseline approach for steering prediction, while the CNN + LSTM model explores the use of temporal information from consecutive camera frames.

The project also highlights an important practical lesson:

Successful autonomous driving depends not only on model architecture, but also on the quality, diversity, and coverage of the training data.

Future improvements will focus on collecting better driving data, improving recovery behavior, experimenting with model architectures, and evaluating the system through longer and more systematic simulator testing.

Author

Sushanth Kumar Reddy

IIT Bhubaneswar

Data Science / Machine Learning

Acknowledgements

Udacity Self-Driving Car Simulator

NVIDIA End-to-End Learning for Self-Driving Cars

TensorFlow and Keras

OpenCV

Scikit-learn

License

This project is intended for educational and research purposes.


### One small thing before you save it

In the README, I used **`main.py` for CNN training** and **`LSTM.py` for CNN + LSTM training**, matching the files you showed.

Also, your current `drive_lstm.py` should be the **testing script that loads the saved LSTM model**, not the training script.

If you want, next I can give you the **exact `.gitignore` and `requirements.txt` files** for this project, 