import cv2


# Capture video from the default webcam, and check if it was opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Initialize global variables for drawing bounding box and tracking
drawing_bbox = False 
ix, iy = -1, -1
bbox = None

tracker = None
tracking_flag = False

snapshot = None
frame = None


# Function to reset all tracking and drawing states
def reset():

    global drawing_bbox, ix, iy, bbox, tracker, tracking_flag, snapshot

    drawing_bbox = False
    ix, iy = -1, -1
    bbox = None

    tracker = None
    tracking_flag = False
    
    snapshot = None


# Mouse callback function to handle drawing the bounding box and initializing the tracker
def mouse_events(event, x, y, flags, param):

    global ix, iy, drawing_bbox, bbox, tracker, tracking_flag, snapshot

    # if the left mouse button was pressed, start drawing the bounding box
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_bbox = True
        ix, iy = x, y

    # if the mouse is moving and we are currently drawing, update the bounding box dimensions
    elif event == cv2.EVENT_MOUSEMOVE and drawing_bbox:
        bbox = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))

    # if the left mouse button was released, finalize the bounding box and initialize the tracker
    elif event == cv2.EVENT_LBUTTONUP:
        drawing_bbox = False
        bbox = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))

        bx, by, bw, bh = bbox
        # If the bounding box is sufficiently large, take a snapshot
        if bw > 5 and bh > 5:
            crop = frame[by:by+bh, bx:bx+bw]
            if crop.size > 0:
                snapshot = cv2.resize(crop, (80, 80))

        # Initialize the CSRT tracker with the current frame and bounding box
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        tracking_flag = True


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

    # If tracking is active, update the tracker and draw the bounding box
    if tracking_flag:
        success, bbox = tracker.update(frame)
        if success:
            x, y, bw, bh = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Tracking failed", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            
    # If the user is currently drawing a bounding box, show the temporary rectangle
    elif bbox is not None:
        x, y, bw, bh = bbox
        cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)

    # If a snapshot of the target object is available, display it in the top-left corner
    if snapshot is not None:
        th, tw = snapshot.shape[:2]
        pad = 6
        cv2.rectangle(frame, (pad, pad), (pad+tw+4, pad+th+18), (20, 20, 20), -1)
        frame[pad+2:pad+2+th, pad+2:pad+2+tw] = snapshot
        cv2.putText(frame, "Target", (pad+2, pad+th+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

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
    elif key == ord('r') or key == ord('R'): # R key to reset the tracker and snapshot
        reset()

cap.release()
cv2.destroyAllWindows()