import cv2
import numpy as np
import pytesseract
from tensorflow.keras.models import load_model
from loss import focal_loss

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

MODEL_PATH = 'model/model.h5'

model = load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss(gamma=2, alpha=.25)})

# Character map provided by you
CHARACTER_MAP = {
    33: '!',
    35: '#',
    36: '$',
    37: '%',
    38: '&',
    40: '(',
    41: ')',
    42: '*',
    43: '+',
    44: ',',
    45: '-',
    46: '.',
    47: '/',
    48: '0',
    49: '1',
    50: '2',
    51: '3',
    52: '4',
    53: '5',
    54: '6',
    55: '7',
    56: '8',
    57: '9',
    58: ':',
    59: ';',
    60: '<',
    61: '=',
    62: '>',
    63: '?',
    64: '@',
    65: 'A',
    66: 'B',
    67: 'C',
    68: 'D',
    69: 'E',
    70: 'F',
    71: 'G',
    72: 'H',
    73: 'I',
    74: 'J',
    75: 'K',
    76: 'L',
    77: 'M',
    78: 'N',
    79: 'O',
    80: 'P',
    81: 'Q',
    82: 'R',
    83: 'S',
    84: 'T',
    85: 'U',
    86: 'V',
    87: 'W',
    88: 'X',
    89: 'Y',
    90: 'Z',
    91: '[',
    93: ']',
    94: '^',
    95: '_',
    97: 'a',
    98: 'b',
    99: 'c',
    100: 'd',
    101: 'e',
    102: 'f',
    103: 'g',
    104: 'h',
    105: 'i',
    106: 'j',
    107: 'k',
    108: 'l',
    109: 'm',
    110: 'n',
    111: 'o',
    112: 'p',
    113: 'q',
    114: 'r',
    115: 's',
    116: 't',
    117: 'u',
    118: 'v',
    119: 'w',
    120: 'x',
    121: 'y',
    122: 'z',
    123: '{',
    124: '|',
    125: '}',
}

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
        char_images.append(char_image)
        cv2.imshow("Character", char_image)  # Display the character
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    return char_images


def recognize_characters(char_images):
    recognized_chars = []
    confidence_threshold = 0.5  # Set a confidence threshold

    for char_image in char_images:
        char_image = cv2.resize(char_image, (140, 140))  # Resize to match model input
        char_image = char_image.reshape(1, 140, 140, 1)  # Reshape for the model
        prediction = model.predict(char_image)
        confidence = np.max(prediction)
        recognized_class = np.argmax(prediction)

        print(f"Predicted Class: {recognized_class}, Confidence: {confidence}")

        if confidence >= confidence_threshold:
            recognized_char = CHARACTER_MAP.get(recognized_class, '')
            print(f"Recognized Character: {recognized_char}")
            recognized_chars.append(recognized_char)
        else:
            print("Low confidence, skipping character.")
            recognized_chars.append('')  # Or handle it differently

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
