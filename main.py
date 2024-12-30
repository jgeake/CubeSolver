#IM SO HAPPY, I FINALLY DID IT, I GOT THE COLOUR RESULTS TO DISPLAY
        #THE AMOUNT OF THINGS THAT WERE WRONG WITH THAT STUPID CODE
        #I DON'T EVEN KNOW WHERE TO START
    #i wasn't preprocessing the image at first, so it was the wrong size, but then when i did preprocess it,
#i kinda had it in the wrong format/size as well. Then i found that i hadn't denormalized the hsv values, so i did that
    #there were also some syntax errors and loop logic erros
    #and a whole bunch of other things that just didn't work right

from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

WEIGHTS_PATH = "model4squares6.weights.h5"

# Simulated user database
user_data = {
    "Joel": "code1",
    "Dome": "code2",
    "Penny": "code3",
    "Zoe": "code4",
    "Ved": "code5"
}


class UserRequest(BaseModel):
    username: str
    code: str
    
sign_in= False

# Root route to serve index.html
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files 
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the root route to serve index.html
@app.get("/")
async def serve_index():
    index_path = Path("static/index.html")
    if index_path.is_file():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h1>404: Index file not found</h1>", status_code=404)

if __name__ == '__main__':
    app.run(debug=True)

@app.post("/process-user/")
async def process_user(request: UserRequest):
    username = request.username
    code = request.code

    # Validate username and code presence
    if not username:
        return {"message": "Missing username.", "sign_in": False}
    if not code:
        return {"message": "Missing code.", "sign_in": False}

    # Check if user data exists (replace with your user data retrieval logic)
    if user_data.get(username) != code:
        # Check if username exists but code is wrong
        if user_data.get(username):
            return {"message": "Invalid code. Try again.", "sign_in": False}
        else:
            return {"message": "Username not found. Please check your username and try again.", "sign_in": False}
    if user_data.get(username) == code:
        sign_in = True
        return {"message": f"Welcome back, {username}!", "sign_in": True}
    else:
        raise HTTPException(status_code=404, detail="No user was found.")

import os

@app.post("/upload")
async def upload_file(file: UploadFile):
    # Ensure the temp directory exists
    temp_dir = "temp/"
    os.makedirs(temp_dir, exist_ok=True)

    # Save the uploaded file temporarily
    file_path = os.path.join(temp_dir, file.filename)
    try:
        file_contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_contents)
    except Exception as e:
        return {"error": f"Error saving the file: {e}"}

    # Preprocess the uploaded image
    try:
        preprocessed_image = preprocess_image(file_path)
    except ValueError as e:
        return {"error": f"Error preprocessing image: {e}"}

    # Make predictions
    try:
        
        predictions = model.predict(preprocessed_image)  # Returns a batch of predictions
        denormalized_predictions = denormalize(predictions[0])  # Assume first batch element is relevant
    except Exception as e:
        return {"error": f"Error making predictions: {e}"}

    # Calculate foundColours
    try:
        scaled_predictions = [(predictions[0][i] * 224) for i in range(8)]  # Scaled predictions for 224x224
        found_results, found_colours = findingColours(preprocessed_image, scaled_predictions)
    except Exception as e:
        return {"error": f"Error processing colours: {e}"}

    # Return the response
    return {
        "predictions": denormalized_predictions.tolist(),
        "foundColours": found_colours,
        "found_results": found_results
    }
    
    
    
# communication with java
import subprocess
java_directory = r"C:\Users\jgeak\eclipse-workspace\2x2\bin"

@app.get("/greet")
async def greet(name: str):
    try:
        # Run Java process with subprocess and specify the classpath
        result = subprocess.run(
            ["java", "-cp", java_directory, "Solver", name],
            capture_output=True, text=True, check=True
        )
        
        # Get the output from Java process
        output = result.stdout.strip()
        
        return {"message": output}
    except subprocess.CalledProcessError as e:
        return {"error": f"Java process failed: {e}"}    




def create_model():
    input_shape = (224, 224, 3)  # Image dimensions
    num_boxes = 1  # Number of bounding boxes per image

    # Load MobileNetV2 as the base model
    base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights="imagenet")

    # Freeze base model layers
    base_model.trainable = False

    # Add custom layers
    x = base_model.output
    x = Conv2D(32, (6, 6), activation="sigmoid", padding="same")(x)
    x = Conv2D(64, (4, 4), activation="relu", padding="same")(x)
    x = Conv2D(128, (2, 2), activation="relu", padding="same")(x)
    x = Flatten()(x)
    x = Dense(256, activation="sigmoid")(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="sigmoid")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="sigmoid")(x)
    x = Dropout(0.1)(x)
    output = Dense(num_boxes * 8, activation="linear")(x)  # 8 values per box (4x, 4y)

    # Create the complete model
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Load model and weights
def load_model_with_weights():
    model = create_model()
    model.load_weights(WEIGHTS_PATH)
    print("Model loaded successfully with weights!")
    return model

def preprocess_image(image_path):
    try:
        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))  # Resize to model input size
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        return np.expand_dims(image_array, axis=0)  # Add batch dimension
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")

def denormalize(predictions):
    predictions = predictions.astype(np.float32)

    # Apply scaling factors
    for i, pred in enumerate(predictions):
        if i % 2 == 0:  # x-coordinates
            predictions[i] *= 1536.0
        else:  # y-coordinates
            predictions[i] *= 2048.0

    print(predictions)
    return predictions.astype(np.float16)  # Return as float16 for consistency

# Load model globally
model = load_model_with_weights() 



colors = {
    "white": np.array([30, 13, 205]),
    "yellow": np.array([33, 190, 205]),
    "red": np.array([170, 175, 205]),
    "orange": np.array([11.5, 190, 205]),
    "green": np.array([47.5, 190, 205]),  
    "blue": np.array([94, 110, 205]),
}

def euclidean_distance(pixel_hsv, color_hsv):
    return np.sqrt(np.sum((np.array(pixel_hsv) - np.array(color_hsv)) ** 2))

# Find the closest color for a given pixel
def closest_color(pixel_hsv):
    closest = None
    min_distance = float("inf")

    for color_name, hsv_value in colors.items():
        distance = euclidean_distance(pixel_hsv, hsv_value)
        if distance < min_distance:
            min_distance = distance
            closest = color_name
    return closest

# Map colors to numeric values
def color_to_value(dominant_color):
    color_mapping = {
        "yellow": 0,
        "blue": 1,
        "white": 2,
        "green": 3,
        "orange": 4,
        "red": 5,
    }
    return color_mapping.get(dominant_color, -1)

def findingColours(image, total_predictions):
    try:
        predictions = generate_predictions(total_predictions)
        results, colour_results = process_images(image, predictions)
        return results, colour_results  # Return meaningful values
    except Exception as e:
        print(f"Error in findingColours: {e}")
        return [-1]  # Default value for error

def generate_predictions(pred):
    # Convert `pred` to a 2D array if it's not already
    predictions = np.array(pred).reshape(-1, 2)
    # Assuming each prediction is a pair of coordinates
    return predictions

def process_images(image, predictions):
    results = []
    try:
        result, colour_results = hsv_predicting(image, predictions)
        results.extend(result)
    except Exception as e:
        print(f"Error processing image: {e}")
        results.append(-1)
    return results, colour_results

def hsv_predicting(preprocessed_image, predictions):
    try:
        # Remove batch dimension if present
        if len(preprocessed_image.shape) == 4:
            preprocessed_image = preprocessed_image[0]

        # Convert image to uint8 format
        preprocessed_image = (preprocessed_image * 255).astype(np.uint8)

        # Convert RGB image to HSV
        test_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2HSV)

        # Create polygon mask
        polygon = np.array(predictions, np.int32)
        mask = np.zeros((224, 224), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        # Calculate bounding box and midpoints
        x_min, y_min, x_max, y_max = (
            min(x for x, y in predictions),
            min(y for x, y in predictions),
            max(x for x, y in predictions),
            max(y for x, y in predictions),
        )
        x_mid = (x_min + x_max) // 2
        y_mid = (y_min + y_max) // 2

        # Define quadrant masks
        quadrant_masks = {
            "top_left": (mask & (np.arange(224)[:, None] < y_mid) & (np.arange(224)[None, :] < x_mid)),
            "top_right": (mask & (np.arange(224)[:, None] < y_mid) & (np.arange(224)[None, :] >= x_mid)),
            "bottom_left": (mask & (np.arange(224)[:, None] >= y_mid) & (np.arange(224)[None, :] < x_mid)),
            "bottom_right": (mask & (np.arange(224)[:, None] >= y_mid) & (np.arange(224)[None, :] >= x_mid)),
        }

        # Extract HSV values and predict colors
        results = []
        colour_results = []
        for quadrant, mask_section in quadrant_masks.items():
            indices = np.column_stack(np.where(mask_section))
            if len(indices) > 0:
                avg_hsv = np.mean([test_image[y, x] for y, x in indices], axis=0)
                dominant_colour = closest_color(avg_hsv)
                results.append(color_to_value(dominant_colour))
                colour_results.append(dominant_colour)
            else:
                results.append(-1)  # No pixels in the quadrant
                colour_results.append(-1)

        return results, colour_results
    except Exception as e:
        print(f"Error in hsv_predicting: {e}")
        return [-1, -1, -1, -1]  # Default value for error