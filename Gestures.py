import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_finger_gesture(hand_landmarks, is_right_hand=True):
    # Get finger coordinates
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP] 
    middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]

    # Get MCP (base) coordinates for reference
    index_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP] 
    ring_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP]

    # Check if fingers are closed by comparing y coordinates
    # A finger is considered closed if tip is below MCP
    index_closed = index_tip.y > index_mcp.y
    middle_closed = middle_tip.y > middle_mcp.y
    ring_closed = ring_tip.y > ring_mcp.y
    pinky_closed = pinky_tip.y > pinky_mcp.y

    # Detect gestures based on finger states
    if is_right_hand:
        if index_closed and middle_closed and ring_closed and pinky_closed:
            return "right_4_fingers_closed"
        elif index_closed and middle_closed and ring_closed and not pinky_closed:
            return "right_3_fingers_closed"
        elif index_closed and not middle_closed and not ring_closed and not pinky_closed:
            return "right_index_closed"
        elif index_closed and middle_closed and not ring_closed and not pinky_closed:
            return "right_index_middle_closed"
        elif not index_closed and middle_closed and not ring_closed and not pinky_closed:
            return "right_middle_closed"
        elif not index_closed and middle_closed and ring_closed and not pinky_closed:
            return "right_middle_ring_closed"
        elif not index_closed and not middle_closed and ring_closed and not pinky_closed:
            return "right_ring_closed"
        elif not index_closed and not middle_closed and ring_closed and pinky_closed:
            return "right_ring_pinky_closed"
    else:
        if index_closed and middle_closed and ring_closed and pinky_closed:
            return "left_4_fingers_closed"
        elif index_closed and middle_closed and ring_closed and not pinky_closed:
            return "left_3_fingers_closed"
        elif index_closed and not middle_closed and not ring_closed and not pinky_closed:
            return "left_index_closed"
        elif index_closed and middle_closed and not ring_closed and not pinky_closed:
            return "left_index_middle_closed"
        elif not index_closed and middle_closed and not ring_closed and not pinky_closed:
            return "left_middle_closed"
        elif not index_closed and middle_closed and ring_closed and not pinky_closed:
            return "left_middle_ring_closed"
        elif not index_closed and not middle_closed and ring_closed and not pinky_closed:
            return "left_ring_closed"
        elif not index_closed and not middle_closed and ring_closed and pinky_closed:
            return "left_ring_pinky_closed"
    
    return "no_gesture_detected"

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    max_num_hands=2,  # Detect up to 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip the image horizontally for a mirrored view
    image = cv2.flip(image, 1)
    
    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Determine if it's a right or left hand
            is_right_hand = results.multi_handedness[idx].classification[0].label == "Right"
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
            
            # Detect gesture
            gesture = detect_finger_gesture(hand_landmarks, is_right_hand)
            
            # Get the position for text
            hand_x = int(min(hand_landmarks.landmark[0].x * image.shape[1], image.shape[1] - 200))
            hand_y = int(hand_landmarks.landmark[0].y * image.shape[0])
            
            # Put text on image
            cv2.putText(
                image,
                f"Gesture: {gesture}",
                (hand_x, hand_y - 20),  # Position above the hand
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),  # Green color
                2
            )

    # Display the image
    cv2.imshow('Hand Gesture Detection', image)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()