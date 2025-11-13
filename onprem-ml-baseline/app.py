from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import random
from datetime import datetime
import psutil
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'ml_inference'),
    'user': os.getenv('DB_USER', 'beas28'),
    'password': os.getenv('DB_PASSWORD', ''),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

# Mock labels for EuroSAT dataset
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
    """Mock image upload endpoint"""
    start_time = time.time()
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Save placeholder file
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Generate mock prediction
    predicted_label = random.choice(EUROSAT_LABELS)
    confidence = round(random.uniform(0.70, 0.99), 4)
    
    # Insert into database
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (image_name, predicted_label, confidence) VALUES (%s, %s, %s) RETURNING id",
            (filename, predicted_label, confidence)
        )
        prediction_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    
    # Log resource usage
    resources = log_resource_usage()
    latency = (time.time() - start_time) * 1000  # Convert to ms
    
    return jsonify({
        'prediction_id': prediction_id,
        'filename': filename,
        'predicted_label': predicted_label,
        'confidence': confidence,
        'latency_ms': round(latency, 2),
        'resources': resources
    }), 200

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
