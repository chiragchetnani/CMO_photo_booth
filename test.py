import os
import requests

# Base URL for the FastAPI server
BASE_URL = "http://127.0.0.1:8000"

# Folder with images to upload for testing
TEST_FOLDER = "test"

# Add each image in the 'test' folder to the /add_face/ endpoint with a sample event name
def add_images(event_name):
    print("Adding images to the database...")
    for filename in os.listdir(TEST_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(TEST_FOLDER, filename)
            with open(file_path, "rb") as file:
                response = requests.post(
                    f"{BASE_URL}/add_face/",
                    files={"file": file},
                    data={"event_name": event_name}
                )
                print(f"Response for {filename}:", response.json())

# List all unique events in the database
def list_events():
    print("\nListing all unique events in the database...")
    response = requests.get(f"{BASE_URL}/list_events/")
    print("Events:", response.json())

# Get all images associated with a specific event name
def get_images_by_event(event_name):
    print(f"\nGetting images for event: {event_name}...")
    response = requests.get(f"{BASE_URL}/get_images_by_event/", params={"event_name": event_name})
    print("Images for event:", response.json())

# Perform a face search on a specified image with an event name filter
def search_faces(event_name, search_image):
    print(f"\nSearching for faces in image: {search_image} for event: {event_name}...")
    with open(search_image, "rb") as file:
        response = requests.post(
            f"{BASE_URL}/search_faces/",
            files={"file": file},
            data={"event_name": event_name}
        )
    print("Search result:", response.json())

# Main function to run all tests
def main():
    event_name = "Test Event"

    # Step 1: Add images to the database
    add_images(event_name)

    # Step 2: List all unique events
    list_events()

    # Step 3: Get images by specific event
    get_images_by_event(event_name)

    # Step 4: Perform face search
    search_image_path = "photo-1.png"  # Replace with actual image path
    if os.path.exists(search_image_path):
        search_faces(event_name, search_image_path)
    else:
        print("Search image not found in the test folder!")

# Run the script
if __name__ == "__main__":
    main()
