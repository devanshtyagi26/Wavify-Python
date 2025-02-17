import mediapipe as mp
import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cam = cv2.VideoCapture(0)

# Store distances globally
hand_distances = {"left": 0, "right": 0, "thumb_distance": 0}

def calculate_distance(landmark1, landmark2, w, h):
    """Calculate Euclidean distance between two landmarks."""
    x1, y1 = int(landmark1.x * w), int(landmark1.y * h)
    x2, y2 = int(landmark2.x * w), int(landmark2.y * h)
    return round(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 2)

def generate_frames():
    global hand_distances

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while True:
            success, frame = cam.read()
            if not success:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Reset distances
            hand_distances = {"left": 0, "right": 0, "thumb_distance": 0}
            thumb_positions = {}

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    h, w, _ = image.shape

                    # Extract landmarks
                    landmark_4 = hand_landmarks.landmark[4]  # Thumb Tip
                    landmark_8 = hand_landmarks.landmark[8]  # Index Tip

                    # Calculate distance between thumb and index for each hand
                    distance = calculate_distance(landmark_4, landmark_8, w, h)

                    # Store distance based on hand (Left/Right)
                    label = handedness.classification[0].label  # 'Left' or 'Right'
                    hand_distances[label.lower()] = distance  # Store in global variable

                    # Store thumb position for later use
                    thumb_positions[label.lower()] = (landmark_4, w, h)

                    # Draw landmarks
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Draw a line between 4 and 8
                    x4, y4 = int(landmark_4.x * w), int(landmark_4.y * h)
                    x8, y8 = int(landmark_8.x * w), int(landmark_8.y * h)
                    cv2.line(image, (x4, y4), (x8, y8), (0, 255, 0), 3)

                    # Display distance on frame
                    cv2.putText(image, f"{label}: {distance}px", (x8, y8 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Calculate the distance between Left and Right Thumb
                if "left" in thumb_positions and "right" in thumb_positions:
                    left_thumb, w, h = thumb_positions["left"]
                    right_thumb, _, _ = thumb_positions["right"]

                    thumb_distance = calculate_distance(left_thumb, right_thumb, w, h)
                    hand_distances["thumb_distance"] = thumb_distance

                    # Draw a line between left and right thumb
                    x_left, y_left = int(left_thumb.x * w), int(left_thumb.y * h)
                    x_right, y_right = int(right_thumb.x * w), int(right_thumb.y * h)
                    cv2.line(image, (x_left, y_left), (x_right, y_right), (0, 0, 255), 3)

                    # Display thumb distance on frame
                    cv2.putText(image, f"Thumb Distance: {thumb_distance}px", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    """Send hand distances as JSON to the frontend."""
    return jsonify(hand_distances)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
