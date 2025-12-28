# NeuroScan AI - Backend API

Backend API service for Alzheimer's MRI classification using ConvNeXt and Grad-CAM++.

## Deployment on Render

This service can be deployed as a standalone web service on Render.

### Configuration

- **Environment**: Python
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn api:app --timeout 120 --workers 2 --threads 4 --worker-class gthread --bind 0.0.0.0:$PORT`
- **Region**: Oregon (recommended)

### Environment Variables

- `MODEL_PATH`: Path to the model file (default: `models/best_model.pth`)
- `SECRET_KEY`: Secret key for Flask (auto-generated recommended)
- `FLASK_ENV`: Environment (production)

### Health Check

- Endpoint: `/api/health`

## Local Development

```bash
pip install -r requirements.txt
python api.py
```