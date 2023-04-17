import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import mediapipe as mp


# Define the hand landmark detection module from Mediapipe
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define the function to extract hand landmarks from an image using Mediapipe
def extract_hand_landmarks(image):
    # Convert the image to RGB format and process it with the hand landmark detection module
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = (image_rgb * 255).astype(np.uint8)
    results = mp_hands.process(image_rgb)

    # If hand landmarks are detected, extract them and return the coordinates
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmark_coords = np.zeros((21, 2))
        for idx, landmark in enumerate(hand_landmarks.landmark):
            landmark_coords[idx] = [landmark.x, landmark.y]
        return landmark_coords

    # If no hand landmarks are detected, return None
    else:
        return None

# Define the function to preprocess an image by extracting the hand landmarks and resizing it
def preprocess_image(image):
    # Load the image and extract the hand landmarks
    #image = cv2.imread(image_path)
    landmarks = extract_hand_landmarks(image)

    # If hand landmarks are detected, crop the image to the bounding box around the hand
    if landmarks is not None:
        xmin = int(np.min(landmarks[:, 0]) * image.shape[1])
        ymin = int(np.min(landmarks[:, 1]) * image.shape[0])
        xmax = int(np.max(landmarks[:, 0]) * image.shape[1])
        ymax = int(np.max(landmarks[:, 1]) * image.shape[0])
        image = image[ymin:ymax, xmin:xmax]

    # Resize the image to the desired input size for the model
    image = cv2.resize(image, (96, 96))

    # Return the preprocessed image
    return image

# Define the function to predict the hand sign from an image using the trained model
def predict_hand_sign(image):
    # Preprocess the image and expand the dimensions to match the input shape of the model
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)

    # Use the trained model to predict the hand sign from the image
    prediction = model.predict(image)

    # Return the predicted hand sign
    return prediction.argmax()

def run_hand_sign_detection(model_path):
    # Load the saved model
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    # Define the hand landmark detection module from Mediapipe
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize the webcam video stream
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        # Extract the hand landmarks and preprocess the image
        landmarks = extract_hand_landmarks(frame)
        if landmarks is not None:
            image = preprocess_image(frame)

            # Make a prediction with the trained model
            prediction = model.predict(np.expand_dims(image, axis=0))[0]
            label = np.argmax(prediction)
            confidence = prediction[label]
            labels = {"0":"hello","1":"iloveyou","2":"no","3":"Thank You","4":"yes"}
            label = labels[str(label)]
            # Draw a bounding box around the detected hand and display the predicted label and confidence score
            xmin = int(np.min(landmarks[:, 0]) * frame.shape[1])
            ymin = int(np.min(landmarks[:, 1]) * frame.shape[0])
            xmax = int(np.max(landmarks[:, 0]) * frame.shape[1])
            ymax = int(np.max(landmarks[:, 1]) * frame.shape[0])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Hand Sign Detection", frame)

        # Exit if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the resources
    cap.release()
    cv2.destroyAllWindows()
    
run_hand_sign_detection('sign_language_model.h5')  
