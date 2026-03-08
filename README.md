# Real-Time Object Tracker (Webcam)

This project implements a **real-time object tracking system** using a
webcam.\
The user selects an object in the first frame by drawing a bounding box,
and the system tracks the object as it moves in the video stream.

Two different tracking approaches are implemented: 
1. **CSRT Tracker(Classical OpenCV tracker)**
2.  **DaSiamRPN Tracker (Deep-learning based tracker)**

------------------------------------------------------------------------

# Features

-   Select an object manually using a bounding box
-   Real‑time tracking from webcam feed
-   Live visualization of the tracked object
-   Reset tracking anytime
-   Two alternative tracking approaches for comparison

------------------------------------------------------------------------

# Project Structure

    .
    ├── Obj_Tracker_CSRT.py
    ├── Obj_Tracker_DaSiamRPN.py
    ├── requirements.txt
    └── DaSiamRPN/
        ├── dasiamrpn_model.onnx
        ├── dasiamrpn_kernel_r1.onnx
        └── dasiamrpn_kernel_cls1.onnx

------------------------------------------------------------------------

# Installation

## 1. Clone the repository

``` bash
git clone https://github.com/MariamElwakel/Object-Tracker
cd Object-Tracker
```

## 2. Create a virtual environment

``` bash
python -m venv venv
```

## 3. Activate it:

``` bash
venv\Scripts\activate
```

## 4. Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Running the Project

## CSRT Tracker

Run:

``` bash
python Obj_Tracker_CSRT.py
```

Steps:

1.  Webcam feed opens
2.  Use the mouse to draw a bounding box around the target object
3.  The tracker follows the object in real time

Controls:

-   **R** → Reset tracking
-   **ESC** → Exit

------------------------------------------------------------------------

## DaSiamRPN Tracker

Run:

``` bash
python Obj_Tracker_DaSiamRPN.py
```

Steps:

1. Webcam feed opens
2. Draw a bounding box around the object
3. The deep tracker follows the object and outputs a **tracking score**
4. If the score drops below a threshold, the system indicates that **tracking is lost**
5. The tracker then attempts to **re-detect the object in a search region near its last known position**


Controls:

-   **R** → Reset tracker
-   **ESC** → Exit

------------------------------------------------------------------------

# Implementation Details (Short Overview)

## 1. User Object Selection

Both implementations allow the user to select the target object using
the mouse.\
A bounding box is drawn on the first frame and used to initialize the
tracker.

## 2. CSRT Tracking

The **CSRT (Discriminative Correlation Filter with Channel and Spatial
Reliability)** tracker is used from OpenCV.

Workflow:

1.  User selects object
2.  Tracker is initialized with the bounding box
3.  Each new frame updates the tracker
4.  The new object location is returned and drawn on the frame

## 3. DaSiamRPN Tracking

DaSiamRPN is a **deep Siamese network tracker**.

Workflow:

1.  ONNX deep model is loaded
2.  User selects the object
3.  The tracker compares the current frame with the initial target
    appearance
4.  A **tracking score** determines confidence in the result

------------------------------------------------------------------------

# Comparison of the Two Approaches

| Feature        | CSRT Tracker      | DaSiamRPN Tracker     |
| -------------- | ----------------- | --------------------- |
| Type           | Classical tracker | Deep learning tracker |
| Dependencies   | OpenCV            | OpenCV + ONNX model   |
| Speed          | Moderate          | Faster                |
| Robustness     | Good              | Better                |
| Tracking Score | No score          | Confidence score      |
| Complexity     | Simple            | More complex          |
| Model Files    | Not required      | Required              |

------------------------------------------------------------------------

# Notes

-   Webcam access is required.
-   Good lighting improves tracking performance.
-   Tracking may fail if the object becomes fully occluded.

------------------------------------------------------------------------
