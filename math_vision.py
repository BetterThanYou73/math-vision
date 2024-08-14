import cv2
import ast
import pytesseract
import numpy as np
import pandas as pd
from loss import focal_loss
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

MODEL_PATH = 'model/model.h5'
SAVED_MODEL_CSV = 'model/current_model.csv'
SIZE = (140, 140)  # Assuming SIZE is (140, 140) or adjust accordingly

model = load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss(gamma=2, alpha=.25)})

# Load char_to_num mapping from CSV
def load_char_to_num(csv_path):
    df = pd.read_csv(csv_path)
    
    # Extract the first (and presumably only) entry in the 'char_to_num' column
    char_to_num_str = df.at[0, 'char_to_num']
    
    # Convert the string representation of the dictionary into an actual dictionary
    char_to_num = ast.literal_eval(char_to_num_str)
    
    return char_to_num

char_to_num = load_char_to_num(SAVED_MODEL_CSV)

# Set debug flag to True to enable image visualization
DEBUG = False


def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Using adaptive thresholding for better separation of characters
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return binary_image

def segment_characters(binary_image):
    # Dilate the image slightly to merge close contours
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_images = []
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out small contours that might not be characters
        if w > 10 and h > 10:  # You can adjust these values based on your dataset
            bounding_boxes.append((x, y, w, h))

    # Sort bounding boxes by centroid (center of the box)
    bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[0] + b[2] / 2, b[1] + b[3] / 2))

    # Segment and process each character
    for (x, y, w, h) in bounding_boxes:
        char_image = binary_image[y:y+h, x:x+w]
        
        # Add padding around the character
        padding = 20
        char_image = cv2.copyMakeBorder(char_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Invert the image to make it black-on-white
        char_image = cv2.bitwise_not(char_image)
        
        char_images.append(char_image)

    return char_images

def recognize_characters(char_images):
    recognized_chars = []

    for char_image in char_images:
        # Resize and normalize the character image
        char_image = cv2.resize(char_image, SIZE)  # Resize to match model input
        char_image = char_image.astype('float32') / 255.0  # Normalize to [0, 1]
        char_image = np.expand_dims(char_image, axis=0)  # Expand dimensions for the model

        if DEBUG:
            # Show the image before feeding it into the model
            cv2.imshow("Character Image", char_image[0])
            cv2.waitKey(0)  # Wait for key press to proceed

        char_image = np.expand_dims(char_image, axis=-1)  # Add a channel dimension
        prediction = model.predict(char_image)
        predicted_label_index = np.argmax(prediction, axis=1)[0]

        # Map the predicted label index back to the corresponding character
        predicted_label = list(char_to_num.keys())[list(char_to_num.values()).index(predicted_label_index)]
        recognized_char = chr(int(predicted_label))  # Convert to ASCII character
        
        print(f"Predicted Class: {predicted_label_index}, Recognized Character: {recognized_char}")
        recognized_chars.append(recognized_char)

    if DEBUG:
        cv2.destroyAllWindows()  # Close all OpenCV windows after debugging

    return ''.join(recognized_chars)

def recognize_with_tesseract(image_path):
    text = pytesseract.image_to_string(image_path)
    return text.strip()

def evaluate_expression(expression, image_path):
    try:
        result = eval(expression)
        return result
    except Exception as e:
        try:
            expression = recognize_with_tesseract(image_path)
            print(f"Fallback to Tesseract OCR, Recognized Expression: {expression}")
            result = eval(expression)
            return result
        except Exception as e:
            return str(e)

def main():
    image_path = 'test/3.png'
    binary_image = preprocess_image(image_path)
    char_images = segment_characters(binary_image)

    # First try with TensorFlow model
    expression = recognize_characters(char_images)
    print(f"Recognized Expression: {expression}")

    # If confidence is low, switch to Tesseract
    if len(expression) == 0 or not any(char.isdigit() for char in expression):
        expression = recognize_with_tesseract(image_path)
        print(f"Fallback to Tesseract OCR, Recognized Expression: {expression}")

    result = evaluate_expression(expression, image_path)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()