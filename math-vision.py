import cv2
import pytesseract
import numpy as np
from loss import focal_loss
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

MODEL_PATH = 'model/model.h5'
SIZE = (140, 140)  # Assuming SIZE is (140, 140) or adjust accordingly

model = load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss(gamma=2, alpha=.25)})

char_to_num = {
    '100': 0, '101': 1, '102': 2, '103': 3, '104': 4, '105': 5, '106': 6, '107': 7, '108': 8, '109': 9, '110': 10, '111': 11,
    '112': 12, '113': 13, '114': 14, '115': 15, '116': 16, '117': 17, '118': 18, '119': 19, '120': 20, '121': 21, '122': 22,
    '123': 23, '124': 24, '125': 25, '33': 26, '35': 27, '36': 28, '37': 29, '38': 30, '40': 31, '41': 32, '42': 33, '43': 34,
    '44': 35, '45': 36, '46': 37, '47': 38, '48': 39, '49': 40, '50': 41, '51': 42, '52': 43, '53': 44, '54': 45, '55': 46,
    '56': 47, '57': 48, '58': 49, '59': 50, '60': 51, '61': 52, '62': 53, '63': 54, '64': 55, '65': 56, '66': 57, '67': 58,
    '68': 59, '69': 60, '70': 61, '71': 62, '72': 63, '73': 64, '74': 65, '75': 66, '76': 67, '77': 68, '78': 69, '79': 70,
    '80': 71, '81': 72, '82': 73, '83': 74, '84': 75, '85': 76, '86': 77, '87': 78, '88': 79, '89': 80, '90': 81, '91': 82,
    '93': 83, '94': 84, '95': 85, '97': 86, '98': 87, '99': 88
}

# Set debug flag to True to enable image visualization
DEBUG = True

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    return binary_image

def segment_characters(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_images = []
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

    # Sort the bounding boxes from left to right, top to bottom
    bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1], b[0]))

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

def evaluate_expression(expression):
    try:
        result = eval(expression)
        return result
    except Exception as e:
        return str(e)

def main():
    image_path = 'test/1.png'
    binary_image = preprocess_image(image_path)
    char_images = segment_characters(binary_image)

    # First try with TensorFlow model
    expression = recognize_characters(char_images)
    print(f"Recognized Expression: {expression}")

    # If confidence is low, switch to Tesseract
    if len(expression) == 0 or not expression.isalnum():
        expression = recognize_with_tesseract(image_path)
        print(f"Fallback to Tesseract OCR, Recognized Expression: {expression}")

    result = evaluate_expression(expression)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
