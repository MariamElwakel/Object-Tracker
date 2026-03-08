import cv2
import os


# Capture video from the default webcam, and check if it was opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()


# Paths to the DaSiamRPN model and its kernels
MODEL = os.path.join("DaSiamRPN", "dasiamrpn_model.onnx") # Main Resnet-50 backbone model
KERNEL_R1 = os.path.join("DaSiamRPN", "dasiamrpn_kernel_r1.onnx") # R1 kernel used to predict bounding box location changes
KERNEL_CLS1 = os.path.join("DaSiamRPN", "dasiamrpn_kernel_cls1.onnx") # CLS1 kernel used to predict the classification score of the tracked object


# Initialize global variables for drawing bounding box and tracking
drawing_bbox = False
ix, iy  = -1, -1
bbox    = None
tracker = None
frame   = None
score_threshold = 0.5


# Mouse callback function to handle drawing the bounding box and initializing the tracker
def mouse_events(event, x, y, flags, param):

    global ix, iy, drawing_bbox, bbox, tracker

    # if the left mouse button was pressed, start drawing the bounding box
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_bbox = True
        ix, iy  = x, y

    # if the mouse is moving and we are currently drawing, update the bounding box dimensions
    elif event == cv2.EVENT_MOUSEMOVE and drawing_bbox:
        bbox = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))

    # if the left mouse button was released, finalize the bounding box and initialize the tracker
    elif event == cv2.EVENT_LBUTTONUP:
        drawing_bbox = False
        bbox = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
        bx, by, bw, bh = bbox
        # If the bounding box is sufficiently large, initialize the tracker with the current frame and bounding box
        if bw > 5 and bh > 5:
            # tracker = init_tracker(frame, frame)
            params = cv2.TrackerDaSiamRPN_Params()
            params.model = MODEL
            params.kernel_r1 = KERNEL_R1
            params.kernel_cls1 = KERNEL_CLS1
            tracker = cv2.TrackerDaSiamRPN_create(params)
            tracker.init(frame, bbox)


# Create a window and set the mouse callback function for drawing the bounding box
cv2.namedWindow("Tracking")
cv2.setMouseCallback("Tracking", mouse_events)


# Main loop to read frames from the webcam and perform tracking
while True:

    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # Flip the frame horizontally for a mirror effect
    h, w = frame.shape[:2] # Get the height and width of the frame

    # If the tracker is initialized, update it with the current frame and get the new bounding box and tracking score
    if tracker is not None:
        _, bbox = tracker.update(frame)
        score   = tracker.getTrackingScore()

        # If the tracking score is above the threshold, draw the bounding box and display the score; otherwise, indicate that tracking is lost
        if score >= score_threshold:
            x, y, bw, bh = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cv2.putText(frame, f"score: {score:.2f}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1)
        else:
            cv2.putText(frame, f"Lost (score: {score:.2f})", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # If the user is currently drawing a bounding box, show the temporary rectangle
    elif bbox is not None:
        x, y, bw, bh = bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    # Draw a semi-transparent bar at the bottom of the frame for displaying hints
    bar_h = 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Display hints for resetting and exiting the application
    hints = [
        ("[R] Reset",  (0, 255, 255)),   # yellow
        ("  |  [ESC] Exit", (0, 60, 220)),  # red
    ]

    cx = 12
    cy = h - 9
    for text, color in hints:
        cv2.putText(frame, text, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cx += tw

    cv2.imshow("Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC key to exit
        break

    elif key in (ord('r'), ord('R')): # R key to reset the tracker and snapshot
        bbox    = None
        tracker = None

cap.release()
cv2.destroyAllWindows()