from OpenGL.GL import *
from OpenGL.GLU import *
import pyglet
import numpy as np
import cv2
import mediapipe as mp
from Gestures import detect_finger_gesture  # Import the gesture detection function
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RubiksCube:
    def __init__(self):
        self.cube_size = 0.3  # Reduced size for better visibility
        self.gap = 0.01  # Gap between cubelets
        # Add camera parameters
        self.camera_distance = 5.0
        self.camera_rotation_x = 30
        self.camera_rotation_y = 45
        self.colors = {
            'blue': (0.0, 0.0, 1.0),    # Front
            'green': (0.0, 0.8, 0.0),   # Back
            'orange': (1.0, 0.5, 0.0),  # Left
            'red': (1.0, 0.0, 0.0),     # Right
            'white': (1.0, 1.0, 1.0),   # Top
            'yellow': (1.0, 1.0, 0.0)   # Bottom
        }

        # Initialize cube state (3x3x3 array representing colors)
        self.state = {
            'front': [['blue']*3 for _ in range(3)],
            'back': [['green']*3 for _ in range(3)],
            'left': [['orange']*3 for _ in range(3)],
            'right': [['red']*3 for _ in range(3)],
            'top': [['white']*3 for _ in range(3)],
            'bottom': [['yellow']*3 for _ in range(3)]
        }

    def rotate_face_clockwise(self, face):
        self.state[face] = [list(row) for row in zip(*self.state[face][::-1])]
    
    def rotate_face_counterclockwise(self, face):
        self.state[face] = [list(row) for row in zip(*self.state[face])][::-1]
    
    def rotate_horizontal_row(self, row_idx, direction='left'):
        # Store the front row
        temp = self.state['front'][row_idx].copy()
        
        if direction == 'left':
            self.state['front'][row_idx] = self.state['right'][row_idx]
            self.state['right'][row_idx] = self.state['back'][row_idx]
            self.state['back'][row_idx] = self.state['left'][row_idx]
            self.state['left'][row_idx] = temp
        else:  # right
            self.state['front'][row_idx] = self.state['left'][row_idx]
            self.state['left'][row_idx] = self.state['back'][row_idx]
            self.state['back'][row_idx] = self.state['right'][row_idx]
            self.state['right'][row_idx] = temp
    
    def rotate_vertical_column(self, col_idx, direction='up'):
        # Get column values
        front_col = [row[col_idx] for row in self.state['front']]
        top_col = [row[col_idx] for row in self.state['top']]
        back_col = [row[col_idx] for row in self.state['back']]
        bottom_col = [row[col_idx] for row in self.state['bottom']]
        
        if direction == 'up':
            # Update columns
            for i in range(3):
                self.state['front'][i][col_idx] = bottom_col[i]
                self.state['top'][i][col_idx] = front_col[i]
                self.state['back'][i][col_idx] = top_col[i]
                self.state['bottom'][i][col_idx] = back_col[i]
        else:  # down
            for i in range(3):
                self.state['front'][i][col_idx] = top_col[i]
                self.state['top'][i][col_idx] = back_col[i]
                self.state['back'][i][col_idx] = bottom_col[i]
                self.state['bottom'][i][col_idx] = front_col[i]

    def handle_gesture(self, gesture):
        if gesture == "right_4_fingers_closed":
            self.rotate_face_counterclockwise('front')
        elif gesture == "right_3_fingers_closed":
            self.rotate_face_clockwise('front')
        elif gesture == "right_index_closed":
            self.rotate_horizontal_row(0, 'left')
        elif gesture == "right_index_middle_closed":
            self.rotate_horizontal_row(0, 'right')
        elif gesture == "right_middle_closed":
            self.rotate_horizontal_row(1, 'left')
        elif gesture == "right_middle_ring_closed":
            self.rotate_horizontal_row(1, 'right')
        elif gesture == "right_ring_closed":
            self.rotate_horizontal_row(2, 'left')
        elif gesture == "right_ring_pinky_closed":
            self.rotate_horizontal_row(2, 'right')
        elif gesture == "left_4_fingers_closed":
            self.rotate_face_counterclockwise('right')
        elif gesture == "left_3_fingers_closed":
            self.rotate_face_clockwise('right')
        elif gesture == "left_index_closed":
            self.rotate_vertical_column(0, 'up')
        elif gesture == "left_index_middle_closed":
            self.rotate_vertical_column(0, 'down')
        elif gesture == "left_middle_closed":
            self.rotate_vertical_column(1, 'up')
        elif gesture == "left_middle_ring_closed":
            self.rotate_vertical_column(1, 'down')
        elif gesture == "left_ring_closed":
            self.rotate_vertical_column(2, 'up')
        elif gesture == "left_ring_pinky_closed":
            self.rotate_vertical_column(2, 'down')

    def draw_cubelet(self, x, y, z):
        # Calculate face indices based on position
        face_x = x + 1  # Convert from [-1,0,1] to [0,1,2]
        face_y = y + 1
        face_z = z + 1
        
        vertices = np.array([
            # Front face
            [-1, -1,  1],
            [ 1, -1,  1],
            [ 1,  1,  1],
            [-1,  1,  1],
            # Back face
            [-1, -1, -1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [-1,  1, -1],
        ]) * self.cube_size

        vertices += np.array([x, y, z]) * (self.cube_size * 2 + self.gap)

        faces = [
            ([0, 1, 2, 3], self.state['front'][2-face_y][face_x]),    # Front
            ([4, 5, 6, 7], self.state['back'][2-face_y][2-face_x]),   # Back
            ([0, 3, 7, 4], self.state['left'][2-face_y][2-face_z]),   # Left
            ([1, 2, 6, 5], self.state['right'][2-face_y][face_z]),    # Right
            ([3, 2, 6, 7], self.state['top'][face_z][face_x]),        # Top
            ([0, 1, 5, 4], self.state['bottom'][2-face_z][face_x])    # Bottom
        ]

        glBegin(GL_QUADS)
        for indices, color in faces:
            glColor3fv(self.colors[color])
            for i in indices:
                glVertex3fv(vertices[i])
        glEnd()

    def draw_cube(self):
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    self.draw_cubelet(x, y, z)

# Add error checking for camera
def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        sys.exit(1)
    return cap

def process_camera_feed():
    try:
        success, image = cap.read()
        if not success:
            logger.warning("Failed to read camera frame")
            return

        # Flip the image horizontally for a mirrored view
        image = cv2.flip(image, 1)
        
        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                try:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Detect and process gesture
                    is_right_hand = results.multi_handedness[idx].classification[0].label == "Right"
                    gesture = detect_finger_gesture(hand_landmarks, is_right_hand)
                    
                    if gesture:
                        cube.handle_gesture(gesture)
                    
                    # Display gesture text
                    hand_x = int(min(hand_landmarks.landmark[0].x * image.shape[1], image.shape[1] - 200))
                    hand_y = int(hand_landmarks.landmark[0].y * image.shape[0])
                    cv2.putText(
                        image,
                        f"Gesture: {gesture}",
                        (hand_x, hand_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                except Exception as e:
                    logger.error(f"Error processing hand landmarks: {e}")

        # Show the camera feed
        cv2.imshow('Hand Gesture Detection', image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            window.close()

    except Exception as e:
        logger.error(f"Error in process_camera_feed: {e}")

def update(dt):
    try:
        process_camera_feed()
    except Exception as e:
        logger.error(f"Error in update function: {e}")

# Initialize camera and MediaPipe
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = initialize_camera()
except Exception as e:
    logger.error(f"Failed to initialize camera or MediaPipe: {e}")
    sys.exit(1)

# Create window with OpenGL context
try:
    config = pyglet.gl.Config(double_buffer=True, depth_size=24)
    window = pyglet.window.Window(width=800, height=600, resizable=True, config=config)
    cube = RubiksCube()
except Exception as e:
    logger.error(f"Failed to create window or initialize cube: {e}")
    sys.exit(1)

@window.event
def on_resize(width, height):
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    
    # Initialize OpenGL context
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect_ratio = width / float(height)
    gluPerspective(45, aspect_ratio, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    return True

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.LEFT:
        cube.camera_rotation_y -= 15
    elif symbol == pyglet.window.key.RIGHT:
        cube.camera_rotation_y += 15
    elif symbol == pyglet.window.key.UP:
        cube.camera_rotation_x -= 15
    elif symbol == pyglet.window.key.DOWN:
        cube.camera_rotation_x += 15
    elif symbol == pyglet.window.key.EQUAL:  # Regular + key (usually Shift+=)
        cube.camera_distance = max(2.0, cube.camera_distance - 0.5)
    elif symbol == pyglet.window.key.MINUS:  # Regular - key
        cube.camera_distance = min(10.0, cube.camera_distance + 0.5)
    elif symbol == pyglet.window.key.NUM_ADD:  # Numpad +
        cube.camera_distance = max(2.0, cube.camera_distance - 0.5)
    elif symbol == pyglet.window.key.NUM_SUBTRACT:  # Numpad -
        cube.camera_distance = min(10.0, cube.camera_distance + 0.5)

@window.event
def on_draw():
    window.clear()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Use camera parameters for view
    x = cube.camera_distance * np.sin(np.radians(cube.camera_rotation_y)) * np.cos(np.radians(cube.camera_rotation_x))
    y = cube.camera_distance * np.sin(np.radians(cube.camera_rotation_x))
    z = cube.camera_distance * np.cos(np.radians(cube.camera_rotation_y)) * np.cos(np.radians(cube.camera_rotation_x))
    
    gluLookAt(x, y, z, 0, 0, 0, 0, 1, 0)
    cube.draw_cube()

@window.event
def on_show():
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, window.width / float(window.height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

@window.event
def on_close():
    try:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    finally:
        sys.exit(0)

if __name__ == '__main__':
    try:
        pyglet.clock.schedule_interval(update, 1/30.0)
        pyglet.app.run()
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
        sys.exit(1)