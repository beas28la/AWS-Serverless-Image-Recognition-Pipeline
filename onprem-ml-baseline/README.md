# On-Premise ML Inference Baseline

A Flask-based ML inference API with PostgreSQL database integration for managing predictions and system monitoring.

## Prerequisites
- Python 3.8+
- PostgreSQL 14+

## Setup

### 1. Install PostgreSQL
```bash
# macOS (using Homebrew)
brew install postgresql@14
brew services start postgresql@14
```

### 2. Create Database
```bash
# Connect to default postgres database (ml_inference doesn't exist yet)
psql postgres

# Create database and table
CREATE DATABASE ml_inference;
\c ml_inference

CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    image_name VARCHAR(255) NOT NULL,
    predicted_label VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `env.example` to `.env` and update values as needed:
```bash
cp env.example .env
```

Edit `.env` with your configuration:
```bash
# Flask Server Configuration
FLASK_HOST=127.0.0.1      # Use 127.0.0.1 for local only, 0.0.0.0 for network access
FLASK_PORT=5001           # API server port
FLASK_DEBUG=True          # Enable debug mode for development

# Database Configuration
DB_NAME=ml_inference      # PostgreSQL database name
DB_USER=your_username     # Your PostgreSQL username
DB_PASSWORD=              # Leave empty for local development
DB_HOST=localhost         # Database host
DB_PORT=5432             # PostgreSQL port

# Application Configuration
UPLOAD_FOLDER=./uploads   # Directory for uploaded images

# API Testing Configuration
API_BASE_URL=http://localhost:5001  # Base URL for test scripts
```

### 5. Run the Application
```bash
python app.py
```

The server will start at `http://127.0.0.1:5001` (or your configured host/port)

## API Endpoints

### Health Check
- **GET** `/health` - Check if the API is running
- Response: `{"status": "healthy", "timestamp": "..."}`

### Upload Image
- **POST** `/upload` - Upload an image and get prediction
- Body: `multipart/form-data` with `file` field
- Response: Returns prediction ID, label, confidence, latency, and resource usage

### Get Results
- **GET** `/results` - Retrieve prediction results
- Query params: 
  - `limit` (optional): Number of results to return (default: 10)
  - `id` (optional): Specific prediction ID
- Response: List of predictions with metadata

### Get Statistics
- **GET** `/stats` - System statistics and prediction metrics
- Response: Total predictions, label distribution, system resources

## Testing

### Run All Tests
```bash
python test_endpoints.py
```

### Test Individual Endpoints
```bash
# Health check
curl http://localhost:5001/health

# Upload test image
curl -X POST -F "file=@test_image.jpg" http://localhost:5001/upload

# Get results
curl http://localhost:5001/results

# Get statistics
curl http://localhost:5001/stats
```

### Verify Database Results
Check if predictions are stored in the database:

```bash
# Connect to the database
psql ml_inference

# View all predictions (inside psql)
SELECT * FROM predictions;

# Count total predictions
SELECT COUNT(*) FROM predictions;

# View the 5 most recent predictions
SELECT id, image_name, predicted_label, confidence, created_at 
FROM predictions 
ORDER BY created_at DESC 
LIMIT 5;

# Exit psql
\q
```

Or check directly from command line without entering psql:
```bash
# View all predictions
psql ml_inference -c "SELECT * FROM predictions;"

# Count predictions
psql ml_inference -c "SELECT COUNT(*) FROM predictions;"

# View recent predictions with formatted output
psql ml_inference -c "SELECT id, image_name, predicted_label, confidence, created_at FROM predictions ORDER BY created_at DESC LIMIT 5;"
```

## Project Structure
```
onprem-ml-baseline/
├── app.py              # Main Flask application
├── test_endpoints.py   # API test suite
├── env.example         # Environment variables template
├── .env                # Your local environment config (create from env.example)
├── requirements.txt    # Python dependencies
├── uploads/            # Uploaded images directory
└── README.md          # This file
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_HOST` | `127.0.0.1` | Server host (use `0.0.0.0` for network access) |
| `FLASK_PORT` | `5001` | Server port |
| `FLASK_DEBUG` | `True` | Enable Flask debug mode |
| `DB_NAME` | `ml_inference` | PostgreSQL database name |
| `DB_USER` | `<your_username>` | PostgreSQL username |
| `DB_PASSWORD` | `` | PostgreSQL password (empty for local) |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `UPLOAD_FOLDER` | `./uploads` | Directory for uploaded files |
| `API_BASE_URL` | `http://localhost:5001` | Base URL for API testing |

## Troubleshooting

### PostgreSQL connection error
- Ensure PostgreSQL is running: `brew services list`
- Start if needed: `brew services start postgresql@14`
- Verify credentials in `.env` file

### Permission denied on socket
- Check if PostgreSQL service is running
- Verify you have permission to connect to PostgreSQL