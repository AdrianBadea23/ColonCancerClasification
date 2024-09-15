import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

model = tf.keras.models.load_model(r'ColonV1.keras')
result = ""
def model_function():

    global result
    result = ""
    def preprocess_images(image_path, target_size=(224, 224)):
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    def predict(image_path):
        preprocessed_image = preprocess_images(image_path)
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        print(prediction)
        return predicted_class

    classes = ['colon_aca', 'colon_n']
    predicted_class = predict(filepath)
    result = classes[predicted_class]
    print(result)
    print(f"Raw prediction: {predicted_class} | Predicted result {result}")

app = tk.Tk()
app.geometry('800x800')

# Declare a global variable for file path
filepath = ""

def open_text_file():
    global filepath
    filepath = ""
    # Open file dialog and update the global filepath variable
    filepath = filedialog.askopenfilename(
        initialdir=r"C:\Users\Adrian\Downloads\archive\The IQ-OTHNCCD lung cancer dataset", 
        title="Select file", 
        filetypes=[('All files', '*.*')]
    )
    print("Selected file path:", filepath)  # Optional: print the selected file path

# Create a button to open the file dialog
open_button = tk.Button(app, text='Select file', command=open_text_file)
open_button.grid(sticky='w', padx=20, pady=20)

run_button = tk.Button(app, text='Run prediction', command=model_function)
run_button.grid(sticky='w', padx=20, pady=60)
T = tk.Text(app, height=5, width=32)
T.insert(tk.END, result)
# Start the Tkinter event loop
app.mainloop()