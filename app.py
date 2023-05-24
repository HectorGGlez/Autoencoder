import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras 
from tensorflow.keras.applications.resnet50 import preprocess_input

autoencoder = keras.models.load_model("D:/midterm/autoencoder/models/denoising_autoencoder_weights.h5")
encoder = keras.models.load_model("D:/midterm/autoencoder/models/encoder.h5")
decoder = keras.models.load_model("D:/midterm/autoencoder/models/decoder.h5")


# Define the Gradio interface
def denoise_image(input_image):
    # Open the image
    input_image= np.resize(input_image,(32,32,3))
    input_array = np.array(input_image)
    input_array = preprocess_input(input_array)
    input_array = np.expand_dims(input_array, axis=0)
    hash = encoder.predict(input_array)
    output = decoder.predict(hash)
    hash_image = Image.fromarray((hash[0].reshape(32,32) * 255).astype(np.uint8))
    output_image = Image.fromarray((output[0] * 255).astype(np.uint8))
    return [input_image, hash_image, output_image]

iface = gr.Interface(
    fn=denoise_image,
    inputs= [
         gr.Image (label = "Original Image")
    ],
    outputs=[
         gr.Image (label = "Decoded Output"),
         gr.Image (label= "Hash Output"),
         ],
    title="Denoising Autoencoder",
    description="Upload an image and see its denoised version using a denoising autoencoder.",
    examples=[
        ["D:/midterm/autoencoder/example.jpg"]
      ],
)

iface.launch(share = True, server_port=3001)
