AI-POWERED REAL-TIME EMERGENCY DETECTION AND RESPONSE SYSTEM
=============================================================
Use Video data for trainings


EXECUTION MANUAL

Prepared By:
Prem Gupta (linkedin id:-www.linkedin.com/in/prem-gupta-ab84bb343)


Project Title: AI-Powered Real Time Emergency Detection and Response System
Subject: Project Execution Steps

-------------------------------------------------------------
INTRODUCTION
-------------------------------------------------------------
This document provides a step-by-step guide to execute the final year project titled:

"AI-Powered Real-Time Emergency Detection and Response System (AI-ERDS)"

The project uses deep learning (CNN, LSTM, MobileNetV2), Google Colab (GPU acceleration recommended), Twilio API for SMS alerts, and an optional Flask web interface for user interaction.

-------------------------------------------------------------
STEP 1: Install Python
-------------------------------------------------------------
- Ensure Python 3.x+ is installed.
- Download from: https://www.python.org/downloads/
- Verify installation in terminal:
  > python --version

-------------------------------------------------------------
STEP 2: Set Up Your Project Environment
-------------------------------------------------------------
- Create a project folder on your machine or Google Drive:
  Suggested name:
    AI_ERDS_Project

- Organize subfolders:
    notebooks/   – Colab notebooks
    model/       – saved model weights
    flask_app/   – optional web interface

-------------------------------------------------------------
STEP 3: Prepare the Dataset
-------------------------------------------------------------
- Use any anomaly voilence or activity dataset with different classes lebelled.
  instead of creating dataset, we have used our own created dataset. 
  (dataset is attached with the mail)

- Our dataset file (typically named `project_dataset`)contains 
  3 classes i.e. Fire, car accident, violence where each class has videos
  of corresponding emergency.

- Ensure the dataset is placed inside your project directory.

- Preprocess each video:
    • Extract 15 frames per clip
    • Resize to 64x64 pixels
    • Normalize pixel values (0–1)
    • Store as .npy format

- Example Python preprocessing:
  (for illustration only)
    import cv2
    import numpy as np
    import os

    SEQ_LEN = 15
    IMG_SIZE = 64

    def process_video(video_path, output_dir):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
        cap.release()

        for i in range(0, len(frames) - SEQ_LEN + 1, 2):
            clip = frames[i:i + SEQ_LEN]
            if len(clip) == SEQ_LEN:
                np.save(os.path.join(output_dir, f'clip_{i}.npy'), np.array(clip))

- Augment data with flips, brightness, rotation using Albumentations.
- a sequences of processed data will be created.

-------------------------------------------------------------
STEP 4: Install Required Libraries
-------------------------------------------------------------
- In Colab or locally, install:
    > pip install tensorflow, keras, opencv-python, numpy, pandas, albumentations, twilio, flask

-------------------------------------------------------------
STEP 5: Set Up Google Colab (Recommended)
-------------------------------------------------------------
- Open Google Colab
- Enable TPU:
    Runtime > Change runtime type > Hardware accelerator > TPU

- Mount Google Drive:
    from google.colab import drive
    drive.mount('/content/drive')

-------------------------------------------------------------
STEP 6: Generate Metadata
-------------------------------------------------------------
- Create a metadata CSV/JSON mapping each .npy clip to its label.

  Example CSV format:
    Clip ID, File Path, Label
    0001, data/fire/fire_001.npy, fire
    0002, data/road_accident/acc_004.npy, road_accident
    0003, data/violence/violence_015.npy, violence

- Example code:
    import pandas as pd
    import os

    data = []
    for root, _, files in os.walk('sequences/'):
        for file in files:
            if file.endswith('.npy'):
                label = root.split('/')[-1]
                data.append({'file': os.path.join(root, file), 'label': label})

    df = pd.DataFrame(data)
    df.to_csv('metadata.csv', index=False)

-------------------------------------------------------------
STEP 7: Build and Train the Model
-------------------------------------------------------------
- Architecture:
    • TimeDistributed CNN (MobileNetV2)
    • LSTM for temporal learning
    • Dense layer with softmax for classification (fire, accident, violence)

- Example Keras code:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout

    cnn_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64,64,3))

    model = Sequential([
        TimeDistributed(cnn_base, input_shape=(15, 64, 64, 3)),
        TimeDistributed(Dense(128, activation='relu')),
        LSTM(128),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

- Train with metadata-split data:
    • 80% training / 20% validation
    • Batch size example: 8
    • Epochs example: 5

- Expected results:
    ~98% accuracy
    ~100% validation accuracy

-------------------------------------------------------------
STEP 8: Integrate Twilio SMS Alert
-------------------------------------------------------------
- Install Twilio:
    > pip install twilio

- Example Python code:
    from twilio.rest import Client

    account_sid = 'YOUR_ACCOUNT_SID'
    auth_token = 'YOUR_AUTH_TOKEN'
    client = Client(account_sid, auth_token)

    def send_sms_alert(message_text):
        message = client.messages.create(
            body=message_text,
            from_='YOUR_TWILIO_NUMBER',
            to='RECIPIENT_NUMBER'
        )
        print(f'SMS sent: {message.sid}')

- Embed in prediction:
    if predicted_class in ['fire', 'road_accident', 'violence']:
        send_sms_alert(f"Emergency detected: {predicted_class.upper()}")

-------------------------------------------------------------
STEP 9: (Optional) Deploy Flask Web App
-------------------------------------------------------------
- Create a Flask app:
    • User uploads video
    • Model predicts class
    • SMS alert sent automatically

- Example structure:
    flask_app/
        app.py
        templates/
        static/

- Example Flask snippet:
    from flask import Flask, request, render_template
    app = Flask(__name__)

    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            # Handle uploaded video
            # Run prediction
            # Send SMS if emergency detected
            pass
        return render_template('index.html')

    if __name__ == '__main__':
        app.run(debug=True)

-------------------------------------------------------------
STEP 10: Execute the Complete Pipeline
-------------------------------------------------------------
- On Google Colab:
    • Mount Drive
    • Load data
    • Train model (or load saved weights)
    • Predict on new video

- Local Flask:
    • Upload video
    • View prediction
    • Confirm SMS alert

-------------------------------------------------------------
NOTES
-------------------------------------------------------------
- Ensure correct file paths in code.
- Secure Twilio credentials.
- Test SMS functionality with valid phone numbers.
- At least GPU recommended for training.
- Web interface is optional but useful for demos.

-------------------------------------------------------------
PROJECT OVERVIEW
-------------------------------------------------------------
The AI-ERDS system:
- Automates emergency detection (fire, accident, violence)
- Uses CNN + LSTM in TimeDistributed architecture
- Achieves ~98% accuracy on validation set
- Sends real-time SMS alerts via Twilio
- Includes optional Flask-based web interface

----------------------------x---------------------------------




