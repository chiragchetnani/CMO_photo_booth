from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import streamlit as st
import torch
import numpy as np
import cv2
import os
import pickle
import base64
from scipy.spatial.distance import cdist
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)

# Define paths
BASE_IMAGE_FOLDER = os.path.join(os.path.expanduser("~"), "uploaded_images")
PICKLE_FILE = os.path.join(os.path.expanduser("~"), "face_embeddings.pickle")

# Helper function to load or initialize embeddings database
def load_or_initialize_db():
    if os.path.exists(PICKLE_FILE):
        with open(PICKLE_FILE, 'rb') as f:
            try:
                database = pickle.load(f)
                database.setdefault('embeddings', [])
                database.setdefault('image_paths', [])
                database.setdefault('events', [])

                # Validate embedding dimensions
                if not all(emb.shape[0] == 512 for emb in database['embeddings']):
                    print("Inconsistent embedding dimensions detected, reinitializing database.")
                    database = {'embeddings': [], 'image_paths': [], 'events': []}
                    save_db(database)

                # Check for length consistency
                if len(database['embeddings']) != len(database['image_paths']) or len(database['embeddings']) != len(database['events']):
                    database = {'embeddings': [], 'image_paths': [], 'events': []}
                    save_db(database)

                return database
            except (EOFError, pickle.UnpicklingError):
                return {'embeddings': [], 'image_paths': [], 'events': []}
    else:
        return {'embeddings': [], 'image_paths': [], 'events': []}

# Save embeddings to pickle
def save_db(database):
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(database, f)

# Helper function to detect faces and return embeddings
def get_face_embeddings(image):
    image = image.convert('RGB')
    image_tensor = mtcnn(image)

    if image_tensor is None:
        return [], []

    embeddings = []
    for face in image_tensor:
        # Resize to 160x160, required by FaceNet
        face_resized = torch.nn.functional.interpolate(face.unsqueeze(0), size=(160, 160)).to(device)

        # Generate embedding
        with torch.no_grad():
            embedding = model(face_resized).cpu().numpy().flatten()

        embeddings.append(embedding)

    return embeddings
def search_in_database(image, event_name=None):
    database = load_or_initialize_db()
    
    if len(database['embeddings']) != len(database['image_paths']) or len(database['embeddings']) != len(database['events']):
        return {"error": "Database inconsistency detected!"}
    
    embeddings = get_face_embeddings(image)
    if not embeddings:
        return {"error": "No face detected in the image!"}

    if not database['embeddings']:
        return {"error": "Database is empty!"}

    # Check embedding dimensions
    if any(len(emb) != 512 for emb in embeddings):
        return {"error": "Inconsistent embedding dimensions in uploaded image!"}
    
    threshold = 0.35
    matched_images = set()

    for embedding in embeddings:
        distances = cdist(np.array(database['embeddings']), [embedding], metric='cosine')
        for idx, distance in enumerate(distances):
            if distance[0] <= threshold:
                if event_name is None or database['events'][idx] == event_name:
                    matched_images.add(database['image_paths'][idx])

    return {"matched_images": list(matched_images)}

# Endpoint to add a new face with event metadata, supporting base64 images
@app.post("/add_face/")
async def add_face(event_name: str = Form(...), file: UploadFile = Form(None), image_base64: str = Form(None)):
    # Check if base64 image data is provided
    if image_base64:
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
        except (base64.binascii.Error, IOError):
            return JSONResponse(content={"error": "Invalid base64 image data!"}, status_code=400)
    elif file:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
    else:
        return JSONResponse(content={"error": "No image file or base64 data provided!"}, status_code=400)

    # Ensure the image is in RGB format for consistency
    if image.mode == 'RGBA':
        image = Image.alpha_composite(Image.new("RGB", image.size, (255, 255, 255)), image)
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # Create the directory for the event if it doesn't exist
    event_folder = os.path.join(BASE_IMAGE_FOLDER, event_name)
    os.makedirs(event_folder, exist_ok=True)

    # Define the path to save the image
    image_filename = file.filename if file else f"{event_name}_{len(load_or_initialize_db()['image_paths'])}.jpg"
    image_path = os.path.join(event_folder, image_filename)
    image.save(image_path)

    # Load the database
    database = load_or_initialize_db()

    # Get face embeddings
    embeddings = get_face_embeddings(image)
    if embeddings:
        for embedding in embeddings:
            database['embeddings'].append(embedding)
            database['image_paths'].append(image_path)
            database['events'].append(event_name)
        save_db(database)
        return JSONResponse(content={"message": f"Added {len(embeddings)} face(s) to the database for event: {event_name}"})
    else:
        return JSONResponse(content={"error": "No face detected in the image!"}, status_code=400)

# Endpoint to get a list of all unique events in the database
@app.get("/list_events/")
async def list_events():
    database = load_or_initialize_db()
    unique_events = list(set(database['events']))
    return JSONResponse(content={"events": unique_events})

# Endpoint to get all images associated with a specific event name
@app.get("/get_images_by_event/")
async def get_images_by_event(event_name: str):
    database = load_or_initialize_db()
    
    # Filter images by event name
    matched_images = [
        path for path, event in zip(database['image_paths'], database['events'])
        if event == event_name
    ]
    
    if not matched_images:
        return JSONResponse(content={"error": "No images found for the specified event!"}, status_code=404)
    
    return JSONResponse(content={"matched_images": matched_images})

@app.delete("/delete_event/")
async def delete_event(event_name: str):
    # Load the database
    database = load_or_initialize_db()

    # Find indices of entries matching the event name
    indices_to_delete = [i for i, event in enumerate(database['events']) if event == event_name]

    if not indices_to_delete:
        raise HTTPException(status_code=404, detail="Event not found in the database.")

    # Delete images and database entries
    for idx in sorted(indices_to_delete, reverse=True):
        image_path = database['image_paths'][idx]
        
        # Delete image file if it exists
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Remove the database entries for this index
        del database['embeddings'][idx]
        del database['image_paths'][idx]
        del database['events'][idx]

    # Save the updated database
    save_db(database)

    return JSONResponse(content={"message": f"Event '{event_name}' and associated images have been deleted."})

# Endpoint for face recognition search
@app.post("/search_faces/")
async def search_faces(event_name: str = Form(...), file: UploadFile = Form(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    result = search_in_database(image, event_name)
    
    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)