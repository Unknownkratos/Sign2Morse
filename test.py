import cv2
import mediapipe as mp
from mediapipe.framework.formats import image as mp_image
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import vision

# STEP 1: Create a MediaPipe Hands instance.
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# STEP 2: Create a GestureRecognizer object.
base_options = vision.GestureRecognizerBaseOptions(model_resource_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Open a video capture object to access the webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read frames from the webcam feed.
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the image to RGB format.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a Mediapipe Image from the frame.
    mediapipe_image = mp_image.Image(format=mp_image.ImageFormat.SRGB, data=image)

    # Recognize gestures in the input image.
    recognition_result = recognizer.recognize(mediapipe_image)

    # Process the result. In this case, visualize it.
    top_gesture = recognition_result.gestures[0][0] if recognition_result.gestures else None
    hand_landmarks = recognition_result.hand_landmarks
    # Perform any custom visualization or further processing here.

    # Display the frame with any visualizations or overlays.
    cv2.imshow('MediaPipe Hands', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the capture object and close any open windows.
cap.release()
cv2.destroyAllWindows()
