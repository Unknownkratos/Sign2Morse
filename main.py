import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the hand tracking module
hands = mp_hands.Hands()

# webcam
cap = cv2.VideoCapture(0)  

# Initialize variables to store the sequence of gestures
gesture_sequence = []
thumbs_up_count = 0
thumbs_down_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the landmark coordinates for thumb tip, and also for the tip and base of the index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            # Calculate the distances between thumb tip and index finger tip/base
            thumb_to_index_tip_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2 + (thumb_tip.z - index_tip.z) ** 2) ** 0.5
            thumb_to_index_base_distance = ((thumb_tip.x - index_base.x) ** 2 + (thumb_tip.y - index_base.y) ** 2 + (thumb_tip.z - index_base.z) ** 2) ** 0.5

            # Recognize thumbs up or thumbs down based on the distances
            if thumb_to_index_tip_distance < thumb_to_index_base_distance:
                gesture_label = "Thumbs Down"
                thumbs_down_count += 1
            else:
                gesture_label = "Thumbs Up"
                thumbs_up_count += 1
            
            # Add the recognized gesture to the sequence
            gesture_sequence.append(gesture_label)

            # Print the recognized gesture label on the frame
            cv2.putText(frame, gesture_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame with landmarks and recognized gesture label
    cv2.imshow('GestureTracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the sequence of gestures and their counts to a text file
with open('gesture_sequence.txt', 'w') as file:
    file.write("Thumbs Up: " + str(thumbs_up_count) + "\n")
    file.write("Thumbs Down: " + str(thumbs_down_count) + "\n")
    file.write("\nGesture Sequence:\n")
    file.write('\n'.join(gesture_sequence))

# Translate the gesture sequence to Morse code
morse_code = {'Thumbs Up': '.', 'Thumbs Down': '-'}
morse_sequence = [morse_code[gesture] for gesture in gesture_sequence]

# Function to translate Morse code to English
def morse_to_english(morse_sequence):
    english_sequence = []
    morse_to_english_dict = {'.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y', '--..': 'Z', '.----': '1', '..---': '2', '...--': '3', '....-': '4', '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9', '-----': '0'}
