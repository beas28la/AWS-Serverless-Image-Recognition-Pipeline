# AWS Serverless Image Recognition Pipeline

A serverless ML inference pipeline built with AWS services, compared against an on-premise VM baseline. This project evaluates latency, throughput, scalability, and behavior under bursty workloads to highlight the benefits of cloud-native, event-driven ML systems.

<img width="4587" height="2758" alt="Architecture Diagram" src="https://github.com/user-attachments/assets/f61239de-2ec5-4433-a974-7c53835b2f3c" />

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Components](#components)
  - [AWS Serverless Pipeline](#aws-serverless-pipeline)
  - [On-Premise Baseline](#on-premise-baseline)
  - [ML Model Training](#ml-model-training)
  - [Frontend UI](#frontend-ui)
- [API Endpoints](#api-endpoints)
- [Dataset](#dataset)

---

## Overview

This project implements an **image classification system** using a fine-tuned **ResNet50** model on the **EuroSAT** dataset (satellite imagery with 10 land-use classes). The system is deployed in two configurations:

1. **AWS Serverless**: Event-driven architecture using Lambda, S3, API Gateway, RDS, ECR, and CloudWatch
2. **On-Premise Baseline**: Flask-based REST API running on a local VM with PostgreSQL

---

## Architecture

### AWS Serverless Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   ┌─────────────┐
│   Client    │────▶│ API Gateway │────▶│   Lambda    │──────────────────▶│     S3      │
│   (React)   │     │  (upload)   │     │ (upload.py) │                   │  (images)   │
└─────────────┘     └─────────────┘     └─────────────┘                   └──────┬──────┘
                                                                                 │
                                                                                 ▼ (S3 Trigger)
                                                                          ┌─────────────┐
                                                                          │   Lambda    │
                                                                          │ (infer.py)  │
                                                                          └──────┬──────┘
                                                                                 │
                                                                                 ▼ (Write)
┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   ┌─────────────┐
│   Client    │◀────│ API Gateway │◀────│   Lambda    │◀──────────────────│     RDS     │
│   (Poll)    │     │  (results)  │     │ (query.py)  │      (Read)       │ (PostgreSQL)│
└─────────────┘     └─────────────┘     └─────────────┘                   └─────────────┘
```

**Data Flow:**
1. **Upload**: Client → API Gateway → Lambda (upload.py) → S3
2. **Inference**: S3 Event → Lambda (infer.py) → RDS (write predictions)
3. **Query**: Client → API Gateway → Lambda (query.py) → RDS (read) → Client

### On-Premise Baseline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   Flask     │────▶│  PostgreSQL │
│             │     │   Server    │     │  (local)    │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## Project Structure

```
AWS-Serverless-Image-Recognition-Pipeline/
│
├── AWS_infer_lambda/                # AWS Lambda functions
│   ├── lambda_function.py           # Main inference Lambda (S3 trigger)
│   ├── upload_image.py              # Image upload Lambda (API Gateway)
│   ├── query_result.py              # Query results Lambda (API Gateway)
│   └── dockerfile                   # Docker image for Lambda deployment
│
├── ml-inference-ui/                 # React frontend application
│   ├── src/
│   │   ├── App.tsx                  # Main application component
│   │   ├── components/ui/           # Reusable UI components (shadcn/ui)
│   │   └── lib/utils.ts             # Utility functions
│   ├── package.json
│   └── vite.config.ts
│
├── onprem-ml-baseline/              # On-premise baseline (mock predictions)
│   ├── app.py                       # Flask API server
│   ├── requirements.txt
│   └── test_endpoints.py
│
├── on-prem-baseline-real/           # On-premise baseline (real inference)
│   ├── app.py                       # Flask API with real ResNet50 inference
│   ├── inference/
│   │   ├── model_loader.py          # ResNet50 model loading utilities
│   │   └── predictor.py             # Image prediction functions
│   ├── init_ml_inference_db.py      # Database initialization script
│   └── requirements.txt
│
├── onprem-ml-batch-test/            # Batch testing & performance analysis
│   ├── app.py                       # Flask API with batch upload support
│   ├── inference/
│   │   ├── model_loader.py
│   │   └── predictor.py
│   ├── test_batch_upload.py         # Batch upload testing script
│   ├── inference_analysis_notebook/
│   │   ├── onprem_test_analysis.ipynb
│   │   └── test_batch_analysis.ipynb
│   └── api_results_100_images.csv
│
├── resnet-ml/                       # ML model training
│   ├── eurosat-resnet-classifier.ipynb  # Training notebook
│   ├── documentation/
│   │   └── EuroSAT ResNet50 Training Report.pdf
│   ├── input/resnet50/              # Pre-trained weights
│   └── output/                      # Fine-tuned model weights
│       └── resnet50_eurosat_best.pth
│
└── README.md
```

---

## Tech Stack

### AWS Services
| Service | Purpose |
|---------|---------|
| **Lambda** | Serverless compute for inference & API handlers |
| **S3** | Image storage & event triggers |
| **API Gateway** | RESTful API endpoints |
| **RDS (PostgreSQL)** | Prediction results storage |
| **ECR** | Docker image registry for Lambda |
| **CloudWatch** | Monitoring, logging, and metrics for all Lambda functions |

### On-Premise Stack
| Technology | Purpose |
|------------|---------|
| **Flask** | REST API framework |
| **PostgreSQL** | Local database |
| **PyTorch** | Deep learning framework |
| **ResNet50** | CNN architecture for classification |

### Frontend
| Technology | Purpose |
|------------|---------|
| **React 19** | UI framework |
| **TypeScript** | Type-safe JavaScript |
| **Vite** | Build tool |
| **Tailwind CSS 4** | Styling |
| **shadcn/ui** | UI component library |

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- PostgreSQL 14+
- AWS CLI (for serverless deployment)
- Docker (for Lambda container images)

### On-Premise Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AWS-Serverless-Image-Recognition-Pipeline.git
   cd AWS-Serverless-Image-Recognition-Pipeline
   ```

2. **Set up the on-premise baseline**
   ```bash
   cd onprem-ml-batch-test
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your database credentials and model path
   ```

4. **Initialize the database**
   ```bash
   python init_ml_inference_db.py
   ```

5. **Start the Flask server**
   ```bash
   python app.py
   ```

### Frontend Setup

1. **Navigate to the UI directory**
   ```bash
   cd ml-inference-ui
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

4. **Open in browser**
   ```
   http://localhost:5173
   ```

---

## Components

### AWS Serverless Pipeline

The AWS pipeline consists of three Lambda functions:

1. **`upload_image.py`** - Receives images via API Gateway, stores in S3
2. **`lambda_function.py`** - Triggered by S3 events, performs inference using ResNet50, writes results to RDS
3. **`query_result.py`** - Queries prediction results from RDS

Key features:
- Container-based Lambda deployment via ECR
- Model caching in Lambda `/tmp` for warm starts
- Connection pooling for RDS
- Metrics collection (inference latency, CPU, memory)
- **CloudWatch monitoring** for all 3 Lambda functions:
  - Invocation count & error rates
  - Duration & memory usage
  - Custom metrics & log aggregation

### On-Premise Baseline

Flask-based REST API with two versions:

1. **`onprem-ml-baseline/`** - Mock predictions for quick testing
2. **`on-prem-baseline-real/`** - Real ResNet50 inference
3. **`onprem-ml-batch-test/`** - Batch upload support for performance testing

### ML Model Training

The ResNet50 model is fine-tuned on the EuroSAT dataset:

- **Dataset**: 27,000 satellite images (64x64 RGB)
- **Classes**: 10 land-use categories
- **Architecture**: ResNet50 with modified final FC layer
- **Training**: Transfer learning from ImageNet weights
- **Accuracy**: ~97% on test set

See `resnet-ml/eurosat-resnet-classifier.ipynb` for training details.

### Frontend UI

React-based single-page application featuring:

- Drag & drop image upload
- Real-time inference status polling
- Results table with latency metrics
- Image preview modal
- Responsive design with dark mode support

---

## API Endpoints

### AWS Serverless

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/prod/upload-image` | POST | Upload image for inference |
| `/dev/results` | GET | Query prediction results |
| `/dev/results?request_id=<id>` | GET | Get specific prediction |

### On-Premise

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/upload` | POST | Upload single image |
| `/upload_batch` | POST | Upload multiple images |
| `/results` | GET | Get recent predictions |
| `/results?id=<id>` | GET | Get specific prediction |
| `/stats` | GET | System statistics |

---

## Dataset

**EuroSAT** - Land Use and Land Cover Classification

| Class | Description |
|-------|-------------|
| AnnualCrop | Annual crop fields |
| Forest | Forest areas |
| HerbaceousVegetation | Herbaceous vegetation |
| Highway | Highway infrastructure |
| Industrial | Industrial areas |
| Pasture | Pasture lands |
| PermanentCrop | Permanent crop fields |
| Residential | Residential areas |
| River | Rivers |
| SeaLake | Sea and lake areas |

---

## Performance Comparison

| Metric | AWS Serverless | On-Premise |
|--------|---------------|------------|
| Cold Start Latency | ~5-10s | N/A |
| Warm Start Latency | ~100-300ms | ~50-150ms |
| Scalability | Auto-scaling | Manual |
| Cost Model | Pay-per-request | Fixed |
| Burst Handling | Excellent | Limited |

---

## License

MIT License

---

## Authors

- **beas28** - Initial work and implementation
