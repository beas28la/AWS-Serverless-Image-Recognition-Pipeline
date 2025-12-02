import os
import json
import io
import time
import uuid # for request_id           
import boto3
import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image
from torchvision import transforms as T
import psycopg2

# ---- S3 client & env vars ----
s3 = boto3.client("s3")

MODEL_BUCKET = os.environ["MODEL_BUCKET"]   # e.g. ml-infer-models-alex
MODEL_KEY = os.environ["MODEL_KEY"]         # e.g. models/resnet_eurosat.pth

# Modify according to your dataset
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]
# ---- preprocessing / loader / predictor ----
preprocess = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

DB_HOST = os.environ["DB_HOST"]
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]

_db_conn = None

def get_db_conn():
    global _db_conn
    if _db_conn is not None:
        return _db_conn

    _db_conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=5
    )
    _db_conn.autocommit = True  # For simplicity, we use autocommit
    return _db_conn

def load_resnet_model(model_path, num_classes=10, device="cpu"):
    model = tv_models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    state_dict = torch.load(model_path, map_location=device)
    
    # Handle PyTorch Lightning format (keys have "model." prefix)
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("model."):
        print("Detected PyTorch Lightning format, removing 'model.' prefix...")
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items() 
                     if k.startswith("model.")}
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Model loaded successfully with {len(state_dict)} parameters")
    return model

def predict_image_bytes(model, image_bytes, class_names=None):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, dim=1)

    idx = idx.item()
    conf = float(conf.item())

    if class_names is not None:
        return class_names[idx], conf
    else:
        return idx, conf

# ---- global model cache ----
_model = None

def get_model():
    global _model
    if _model is not None:
        return _model

    local_path = "/tmp/resnet_eurosat.pth"
    if not os.path.exists(local_path):
        s3.download_file(MODEL_BUCKET, MODEL_KEY, local_path)

    _model = load_resnet_model(local_path, num_classes=len(CLASS_NAMES), device="cpu")
    return _model

# ---- Lambda handler: triggered by S3 event ----
def lambda_handler(event, context):
    print("EVENT:", json.dumps(event))

    record = event["Records"][0]
    image_bucket = record["s3"]["bucket"]["name"]
    image_key = record["s3"]["object"]["key"]
    
    # Extract request_id from image filename (without extension)
    image_filename = os.path.basename(image_key)
    request_id = os.path.splitext(image_filename)[0]
    print(f"Request ID: {request_id}")

    # 1) Read image from S3
    resp = s3.get_object(Bucket=image_bucket, Key=image_key)
    image_bytes = resp["Body"].read()

    # 2) Get model
    model = get_model()

    # 3) Predict with timing
    infer_start = time.time()
    label, conf = predict_image_bytes(model, image_bytes, CLASS_NAMES)
    infer_latency_ms = (time.time() - infer_start) * 1000

    print(f"Prediction for {image_key}: {label} ({conf:.4f}), Latency: {infer_latency_ms:.2f}ms, Request ID: {request_id}")

    # --- Write to RDS predictions table ---
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions (request_id, image_name, predicted_label, confidence, infer_latency_ms)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (request_id, image_key, label, conf, infer_latency_ms)
            )
        print("Inserted prediction into RDS for", image_key)
    except Exception as e:
        # For debugging, we print the error; whether to let the entire Lambda fail is up to you
        print("Failed to insert into RDS:", repr(e))

    return {
        "statusCode": 200,
        "body": json.dumps({
            "request_id": request_id,
            "image_key": image_key,
            "label": label,
            "confidence": conf,
            "infer_latency_ms": round(infer_latency_ms, 2)
        })
    }