import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image as keras_image  # Rename the image module to keras_image
import numpy as np

# Define paths
path_to_dataset = r'C:\Users\Adrian\Downloads\archive(1)\lung_colon_image_set\colon_image_sets'  # Replace with your dataset directory

# Load the saved model
model = tf.keras.models.load_model('ColonV1.keras')  # Replace with the path to your saved model

# Define the image size and batch size
image_size = (224, 224)  # Ensure this matches your model's expected input size
batch_size = 64

# Define the preprocessing function
def preprocess_image(image_path):
    img = keras_image.load_img(image_path, target_size=image_size)  # Use keras_image to load the image
    img_array = keras_image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.image.convert_image_dtype(img_array, dtype=tf.float32)  # Normalize
    return img_array

# Test with a single image
image_path = r"C:\Users\Adrian\Downloads\archive(1)\lung_colon_image_set\colon_image_sets\colon_aca\colonca67.jpeg"
preprocessed_image = preprocess_image(image_path)  # Pass the image path as a string
prediction = model.predict(preprocessed_image)
predicted_class = np.argmax(prediction, axis=1)[0]
classes = ['Colon adenocarcinoma', 'Benign']
# Output the prediction and predicted class
print(f"Prediction: {classes[predicted_class]}")
print(f"Predicted class: {predicted_class}")
