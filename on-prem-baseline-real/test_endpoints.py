import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API base URL from environment or use default
BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:5001')

def test_health():
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")


# Test POST endpoint with real Eurosat test image 
def test_upload(): 
    print("Testing upload endpoint...")

    # Path to a real EuroSat test image 
    test_image_dir = "/test_images/" # Replace with path to folder with test image
    test_image_path = os.path.join(test_image_dir, "AnnualCrop_1275.jpg")
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}")
        return None

    #  Send POST request with the image
    with open(test_image_path, 'rb') as f:
        files = {'file': ('AnnualCrop_1275.jpg', f, 'image/jpeg')}
        response = requests.post(f"{BASE_URL}/upload", files=files)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.json()


def test_results(prediction_id=None):
    print("Testing results endpoint...")
    if prediction_id:
        response = requests.get(f"{BASE_URL}/results?id={prediction_id}")
    else:
        response = requests.get(f"{BASE_URL}/results")
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_stats():
    print("Testing stats endpoint...")
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

if __name__ == "__main__":
    print("Running On-Premise API Tests\n")
    print("=" * 50)
    
    test_health()
    time.sleep(1)
    
    upload_response = test_upload()
    time.sleep(1)
    
    if 'prediction_id' in upload_response:
        test_results(upload_response['prediction_id'])
    time.sleep(1)
    
    test_results()
    time.sleep(1)
    
    test_stats()