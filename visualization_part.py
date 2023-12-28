import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from main import text_recognition, shape_recognition, image_preprocessing
import warnings

global input_folder
input_folder = './inputs'

def open_images():
    input_folder = filedialog.askdirectory(
        title="Choose folder with images"
        )
    if input_folder:
        example_image = next((f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))), None)
        if example_image:
            img = Image.open(os.path.join(input_folder, example_image))
            img = img.resize((300, 300))
            photo = ImageTk.PhotoImage(img)
            image_label.config(image=photo)
            image_label.image = photo
        path_var.set(input_folder)

def execution_process():
    tmp_folder = './tmp' 
    xml_output_folder = './outputs'

    for input_file in os.listdir(input_folder):
        if input_file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):

            name, _ = os.path.splitext(input_file)

            input_image_path = os.path.join(input_folder, input_file)
            image_processor = image_preprocessing.ImageProcessing(input_image_path, name)
            image_processor.process_image(tmp_folder)

            text_recognition.recognize(name, tmp_folder, xml_output_folder)
            shape_recognition.shape_digitalization(name, tmp_folder, xml_output_folder)

root = tk.Tk()
root.title("GUI for flipchart digitalization")

path_var = tk.StringVar()

open_button = tk.Button(root, text="Choose folder", command=open_images)
process_button = tk.Button(root, text="Process images", command=execution_process)
image_label = tk.Label(root)

open_button.grid(row=0, column=0, pady=10)
process_button.grid(row=1, column=0, pady=10)
image_label.grid(row=2, column=0)

warnings.filterwarnings("ignore")

root.mainloop()
