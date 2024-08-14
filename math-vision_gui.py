import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas, Scrollbar
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import os
import pandas as pd
import ast
from tensorflow.keras.models import load_model
from loss import focal_loss
import pytesseract
import threading
import sys
import io

from fetch_model import list_model, model_save, fetch_model
from math_vision import preprocess_image, segment_characters, recognize_characters, recognize_with_tesseract, evaluate_expression

pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

MODEL_PATH = 'model/model.h5'
SAVED_MODEL_CSV = 'model/current_model.csv'
SIZE = (140, 140)

# Redirect stdout to the terminal_text widget
class TextRedirector(io.StringIO):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.configure(state=ctk.NORMAL)
        self.text_widget.insert(ctk.END, message)
        self.text_widget.configure(state=ctk.DISABLED)
        self.text_widget.see(ctk.END)

    def flush(self):
        pass  # Handle the flush command (do nothing here)

# Load the model
model = load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss(gamma=2, alpha=.25)})

# Load char_to_num mapping from CSV
def load_char_to_num(csv_path):
    df = pd.read_csv(csv_path)
    char_to_num_str = df.at[0, 'char_to_num']
    char_to_num = ast.literal_eval(char_to_num_str)
    return char_to_num

char_to_num = load_char_to_num(SAVED_MODEL_CSV)

def process_image(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = preprocess_image(image)
    char_images = segment_characters(binary_image)
    expression = recognize_characters(char_images)
    print(f"Recognized Expression: {expression}")
    
    if len(expression) == 0 or not any(char.isdigit() for char in expression):
        expression = recognize_with_tesseract(image)
        print(f"Fallback to Tesseract OCR, Recognized Expression: {expression}")
    
    result = evaluate_expression(expression)
    print(f"Result: {result}")
    messagebox.showinfo("Recognized Expression and Result", f"Expression: {expression}\nResult: {result}")

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        process_image(image)

def capture_image():
    def capture():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                cap.release()
                cv2.destroyAllWindows()
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                process_image(gray_frame)
                break

    threading.Thread(target=capture).start()

def draw_on_canvas():
    brush_color = "black"
    brush_size = 5

    def change_brush_size(new_size):
        nonlocal brush_size
        brush_size = int(new_size)
        print(f"Brush size set to: {brush_size}")

    def toggle_eraser():
        nonlocal brush_color
        brush_color = canvas["background"]  # Set brush to the background color to simulate eraser
        print(f"Brush color set to eraser")

    def use_brush():
        nonlocal brush_color
        brush_color = "black"  # Reset to black color for normal drawing
        print(f"Brush color set to black")

    def save_canvas():
        canvas.postscript(file="drawn_image.ps", colormode='color')
        img = Image.open("drawn_image.ps")
        img_gray = img.convert('L')
        img_gray.save("drawn_image.png")
        process_image(np.array(img_gray))

    def process_canvas_only():
        img = Image.open("drawn_image.ps")
        img_gray = img.convert('L')
        img_gray.save("drawn_image.png")
        process_image(np.array(img_gray))

    def on_closing():
        draw_window.destroy()

    draw_window = ctk.CTkToplevel(root)
    draw_window.title("Draw Image")
    draw_window.protocol("WM_DELETE_WINDOW", on_closing)

    canvas = Canvas(draw_window, width=400, height=400, bg="white")
    canvas.pack()

    def paint(event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        canvas.create_oval(x1, y1, x2, y2, fill=brush_color, width=brush_size)

    canvas.bind("<B1-Motion>", paint)

    save_button = ctk.CTkButton(draw_window, text="Save and Process", command=save_canvas)
    save_button.pack(pady=5)

    process_button = ctk.CTkButton(draw_window, text="Process Only", command=process_canvas_only)
    process_button.pack(pady=5)

    eraser_button = ctk.CTkButton(draw_window, text="Toggle Eraser", command=toggle_eraser)
    eraser_button.pack(pady=5)

    brush_button = ctk.CTkButton(draw_window, text="Use Brush", command=use_brush)
    brush_button.pack(pady=5)

    size_slider = ctk.CTkSlider(draw_window, from_=1, to_=20, command=change_brush_size)
    size_slider.pack(pady=5)

def update_current_model_display():
    try:
        df = pd.read_csv(SAVED_MODEL_CSV)
        model_name = df.at[0, 'model']
        model_label.configure(text=f"Current Model: {model_name}")
    except Exception as e:
        model_label.configure(text="Current Model: None")

# Main GUI
ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "dark-blue", "green"

root = ctk.CTk()
root.title("Math Vision GUI")
root.geometry("600x600")

# Buttons for different functions
load_button = ctk.CTkButton(root, text="Load Image", command=load_image)
load_button.pack(pady=10)

capture_button = ctk.CTkButton(root, text="Capture Image", command=capture_image)
capture_button.pack(pady=10)

draw_button = ctk.CTkButton(root, text="Draw Image", command=draw_on_canvas)
draw_button.pack(pady=10)

fetch_button = ctk.CTkButton(root, text="Fetch Model", command=lambda: [fetch_model(), update_current_model_display()])
fetch_button.pack(pady=10)

# Terminal output area
terminal_frame = ctk.CTkFrame(root)
terminal_frame.pack(pady=10, fill=tk.BOTH, expand=True)

terminal_text = ctk.CTkTextbox(terminal_frame, height=10, state=tk.DISABLED, fg_color="gray", text_color="white")
terminal_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ctk.CTkScrollbar(terminal_frame, command=terminal_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

terminal_text.configure(yscrollcommand=scrollbar.set)

# Redirect stdout to the Text widget
sys.stdout = TextRedirector(terminal_text)

# Current model display
model_label = ctk.CTkLabel(root, text="Current Model: None")
model_label.pack(pady=10)

update_current_model_display()

root.mainloop()
