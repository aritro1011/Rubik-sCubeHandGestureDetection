# 🖐️ Hand Gesture-Controlled Rubik's Cube 🎲

This project implements a hand gesture-controlled Rubik's Cube using MediaPipe for gesture detection and OpenGL for 3D visualization. The project consists of two Python scripts:

- **Gestures.py**: Handles hand gesture recognition using MediaPipe.
- **Rubik'sCube.py**: Implements the Rubik's Cube logic and visualization using OpenGL.

## ✨ Features
- 🚀 Real-time hand gesture detection.
- 🎮 Gesture-based interactions for manipulating a Rubik's Cube.
- 🖥️ OpenGL-based 3D Rubik's Cube rendering.

## 📥 Installation
Ensure you have Python installed, then install the required dependencies:

```sh
pip install -r requirements.txt
```

## ▶️ Usage
Run the following command to start the application:

```sh
python RubiksCube.py
```

Ensure that your webcam is enabled for hand gesture detection.

## 📝 **Note:**
-When the program is run, only the video camera window pops open.  
-Test all the gestures are working on it and kill it by pressing the **Q** key on your keyboard.  
-Then, two new windows—the cube and the video camera—will open up where you can play the game.

This is not the final version—only the first prototype.

## 🎮 Controls
### ✋ Right Hand:
-  **4 fingers closed** - Change the face of the Rubik's Cube on the X plane (horizontally) to **left**
-  **3 fingers closed** - Change the face of the Rubik's Cube on the X plane (horizontally) to **right**
-  **Index finger closed** - First row of the cube shifts **left**
-  **Index and middle finger closed** - First row of the cube shifts **right**
-  **Middle finger closed** - Second row of the cube shifts **left**
   **Middle and ring finger closed** - Second row of the cube shifts **right**
-  **Ring finger closed** - Third row of the cube shifts **left**
-  **Ring and pinky finger closed** - Third row of the cube shifts **right**

### 🤚 Left Hand:
-  **4 fingers closed** - Change the face of the Rubik's Cube on the Y plane (vertically) **up**
-  **3 fingers closed** - Change the face of the Rubik's Cube on the Y plane (vertically) **down**
-  **Index finger closed** - First column goes **up**
-  **Index and middle finger closed** - First column goes **down**
-  **Middle finger closed** - Second column goes **up**
-  **Middle and ring finger closed** - Second column goes **down**
-  **Ring finger closed** - Third column goes **up**
-  **Ring and pinky finger closed** - Third column goes **down**

## 📦 Dependencies
The required dependencies are listed in `requirements.txt`, which includes:
- `mediapipe`
- `numpy`
- `opencv-python`
- `PyOpenGL`
- `PyOpenGL_accelerate`

## 🚀 Future Improvements
- 🔍 Enhance gesture recognition accuracy.
- ✋ Add more intuitive hand gestures for better interaction.
- 🎨 Improve the rendering performance of the Rubik's Cube.


## 🙌 Acknowledgments
- 🎯 [MediaPipe](https://mediapipe.dev/) for hand tracking.
- 🖥️ [PyOpenGL](http://pyopengl.sourceforge.net/) for rendering support.

