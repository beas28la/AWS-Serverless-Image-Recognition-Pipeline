# Original Author: beas28la 
# Date modified: 19 Nov 2025 
# Updated to include inference code in POST endpoint

from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
import psutil
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the predict_image function 
from inference.predictor import predict_image

# Import the load_resnet_model function
from inference.model_loader import load_resnet_model

# --------------------
# Initialize Flask app
# --------------------

app = Flask(__name__)
CORS(app)

# ---------------------------------------------
#  Load fine-tuned Resnet50 Model saved locally 
# ---------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH")

# Load model
model = load_resnet_model(MODEL_PATH, num_classes=10, device='cpu')
if model is None:
    raise RuntimeError("Failed to load ResNet50 model.")

# Configuration
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'ml_inference'),
    # Using default username postgres
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

# Labels for EuroSAT dataset
EUROSAT_LABELS = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(**DB_CONFIG)

def log_resource_usage():
    """Log CPU and memory usage"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_used_mb': memory.used / (1024 * 1024)
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


@app.route('/upload', methods=['POST'])
def upload_image():

    """ Image upload endpoint"""
    start_time = time.time()
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Retrieve true label 
    true_label = request.form.get('true_label', None)

    # Save placeholder file
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)


    # -----------------------------
    # Generate real predictions
    # -----------------------------
    
    # Measure inference time 
    inference_start = time.time()
    try:
        predicted_label, confidence = predict_image(model, filepath, EUROSAT_LABELS)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    inference_ms = round((time.time() - inference_start) * 1000, 2)

    # ---- Total latency (upload + save + inference + DB) ----
    latency_ms = round((time.time() - start_time) * 1000, 2)
    
    # Insert into database
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predictions (image_name, predicted_label, true_label, confidence, inference_ms, latency_ms)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id""",
            (filename, predicted_label, true_label, confidence, inference_ms, latency_ms)
        )
        prediction_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    
    # Log resource usage
    resources = log_resource_usage()
    
    return jsonify({
        'prediction_id': prediction_id,
        'filename': filename,
        'predicted_label': predicted_label,
        'true_label': true_label, 
        'confidence': confidence,
        'inference_ms': inference_ms, 
        'latency_ms': latency_ms,
        'resources': resources
    }), 200


# Update: 11/28 

@app.route('/upload_batch', methods=['POST'])
def upload_batch():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    true_labels = request.form.getlist('true_label')  
    results = []

    for i, file in enumerate(files):
        start_time = time.time()

        # Save uploaded file
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Match true label if provided
        true_label = true_labels[i] if i < len(true_labels) else None

        # Inference timing
        try:
            inference_start = time.time()
            predicted_label, confidence = predict_image(model, filepath, EUROSAT_LABELS)
            inference_ms = round((time.time() - inference_start) * 1000, 2)
        except Exception as e:
            print(f"Error predicting {file.filename}: {e}")
            return jsonify({'error': f'Prediction failed for {file.filename}: {str(e)}'}), 500

        # Total latency
        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Insert into database
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO predictions 
                (image_name, predicted_label, true_label, confidence, inference_ms, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, created_at
                """,
                (filename, predicted_label, true_label, confidence, inference_ms, latency_ms)
            )
            prediction_id, created_at = cur.fetchone()
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"DB error for {file.filename}: {e}")
            return jsonify({'error': f'Database error for {file.filename}: {str(e)}'}), 500
        
        # Log resource usage
        resources = log_resource_usage()

        results.append({
            "filename": filename,
            "prediction_id": prediction_id,
            "predicted_label": predicted_label,
            "true_label": true_label,
            "confidence": confidence,
            "inference_ms": inference_ms,
            "latency_ms": latency_ms,
            "created_at": created_at.isoformat(),
            "resources": resources
        })

    return jsonify({"results": results}), 200


@app.route('/results', methods=['GET'])
def get_results():
    """Retrieve prediction results"""
    start_time = time.time()
    
    # Optional query parameters
    limit = request.args.get('limit', default=10, type=int)
    prediction_id = request.args.get('id', type=int)
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        if prediction_id:
            cur.execute(
                "SELECT * FROM predictions WHERE id = %s",
                (prediction_id,)
            )
            result = cur.fetchone()
            results = [dict(result)] if result else []
        else:
            cur.execute(
                "SELECT * FROM predictions ORDER BY created_at DESC LIMIT %s",
                (limit,)
            )
            results = [dict(row) for row in cur.fetchall()]
        
        cur.close()
        conn.close()
        
        # Convert datetime to string for JSON serialization
        for result in results:
            if 'created_at' in result:
                result['created_at'] = result['created_at'].isoformat()
        
        latency = (time.time() - start_time) * 1000
        
        return jsonify({
            'count': len(results),
            'results': results,
            'latency_ms': round(latency, 2)
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get total predictions count
        cur.execute("SELECT COUNT(*) FROM predictions")
        total_count = cur.fetchone()[0]
        
        # Get label distribution
        cur.execute("""
            SELECT predicted_label, COUNT(*) as count 
            FROM predictions 
            GROUP BY predicted_label 
            ORDER BY count DESC
        """)
        label_distribution = [{'label': row[0], 'count': row[1]} for row in cur.fetchall()]
        
        cur.close()
        conn.close()
        
        resources = log_resource_usage()
        
        return jsonify({
            'total_predictions': total_count,
            'label_distribution': label_distribution,
            'system_resources': resources
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500

if __name__ == '__main__':
    # Server configuration from environment variables
    HOST = os.getenv('FLASK_HOST', '127.0.0.1')
    PORT = int(os.getenv('FLASK_PORT', '5001'))
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print("Starting On-Premise ML Inference Server...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Database: {DB_CONFIG['dbname']} at {DB_CONFIG['host']}")
    print(f"Server running on: http://{HOST}:{PORT}")
    
    app.run(host=HOST, port=PORT, debug=DEBUG)
