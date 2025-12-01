# Script to upload batch of Eurosat test images to API post endpoint
# Then save inference results locally to .csv  

# Imports 
import os
import pathlib
import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -----------------------
# Configuration 
# ------------------------

# Get API base URL from environment or use default
BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:5001')
DATA_DIR = pathlib.Path("/EuroSAT") # Change to your DIR path where Eurosat images are stored 
test_df = pd.read_csv(DATA_DIR/ "test.csv", index_col=0).reset_index(drop=True)
NUM_IMAGES = 100   # Number of test images to send
OUTPUT_CSV = "api_results_100_images.csv"

# Select subset of test images (e.g., first 100) and their corresponding labels
subset_test_df = test_df.iloc[:NUM_IMAGES]
true_labels = list(subset_test_df["ClassName"])

# Build list of file paths
image_paths = [os.path.join(DATA_DIR, fname) for fname in subset_test_df["Filename"]]
print(f"Loaded {len(image_paths)} image paths.")

# -----------------------------------
# Send batch request to API endpoint
# -----------------------------------

def send_batch(paths, true_labels):

    '''
    Function to send batch of images with their corresponding labels by their pathnames to 
    API /upload_batch endpoint
    Inputs:
     - paths: list of filepaths for Eurosat test images 
     - true_labels: list of the actual labels for the test images 
    Outputs:
     - serialized json response 
    '''
    files = []
    data = []

    for path, label in zip(paths, true_labels):
        with open(path, "rb") as f:
            files.append(("files", (os.path.basename(path), f.read(), "image/jpeg")))
        data.append(("true_label", label))

    response = requests.post(f"{BASE_URL}/upload_batch", files=files, data=data)
    try:
        return response.json()
    except Exception:
        print("Response was not JSON:", response.text)
        raise

print("\nSending batch request...\n")
batch_resp = send_batch(image_paths, true_labels)

if "results" in batch_resp:
    results = batch_resp["results"]

    # Compute accuracy
    preds = [r["predicted_label"] for r in results]
    true_labels_api = [r["true_label"] for r in results]
    accuracy_values = [p==t if t is not None else False for p, t in zip(preds, true_labels_api)]
    accuracy = sum(accuracy_values) / len(accuracy_values)
    print(f"Batch Accuracy: {accuracy*100:.2f}%")

    # Save CSV
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV}")

elif "error" in batch_resp:
    print("API returned an error:", batch_resp["error"])
else:
    print("Unexpected response:", batch_resp)
