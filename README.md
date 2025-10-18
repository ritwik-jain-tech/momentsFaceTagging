# Moments Face Tagging Service

A production-grade facial recognition service built with FastAPI, InsightFace, and Google Cloud services. Automatically deployed to Cloud Run with serverless CI/CD pipeline.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Google Cloud SDK
- Docker (for local testing)
- Git repository (GitHub/GitLab)

### 1. Local Development

#### Option A: Automated Setup (Recommended)
```bash
# Clone and setup
git clone <your-repo-url>
cd momentsFaceTagging

# Run automated setup script
./setup-local.sh
```

#### Option B: Manual Setup
```bash
# Clone and setup
git clone <your-repo-url>
cd momentsFaceTagging

# Copy service account key (same as MomentsBackend) - DO NOT COMMIT THIS FILE
cp /path/to/momentsBackend/src/main/resources/serviceAccountKey.json ./serviceAccountKey.json

# Install dependencies
pip install -r requirements.txt

# Set environment
export ENVIRONMENT=development
export GOOGLE_CLOUD_PROJECT=moments-38b77
export GOOGLE_APPLICATION_CREDENTIALS=serviceAccountKey.json

# Test authentication
python test-auth.py

# Run locally
python main.py
```

### 2. Production Deployment
```bash
# Set your Google Cloud project
gcloud config set project YOUR_PROJECT_ID

# Run automated setup
./setup-cicd.sh

# Deploy by pushing to main branch
git add .
git commit -m "Deploy to production"
git push origin main
```

## 📋 Features

- **Facial Recognition**: Advanced face detection and embedding generation
- **Face Matching**: High-accuracy face comparison and tagging
- **Cloud Storage**: Secure image storage and retrieval
- **Firestore Database**: Scalable data persistence
- **Serverless Deployment**: Auto-scaling Cloud Run service
- **CI/CD Pipeline**: Automated deployment on Git push
- **Production Ready**: Health checks, monitoring, and security

## 🏗️ Architecture

```
Git Push → Cloud Build → Container Registry → Cloud Run → Firestore/Storage
```

### Components
- **FastAPI**: Web framework and API endpoints
- **InsightFace**: Face recognition and embedding generation
- **Google Cloud Firestore**: Database for face embeddings
- **Google Cloud Storage**: Image storage
- **Cloud Run**: Serverless hosting platform
- **Cloud Build**: CI/CD pipeline

## 🔧 Configuration

### Environment Variables

| Variable | Local | Production | Description |
|----------|-------|------------|-------------|
| `ENVIRONMENT` | `development` | `production` | Environment identifier |
| `GOOGLE_CLOUD_PROJECT` | `moments-38b77` | `moments-38b77` | GCP Project ID |
| `GOOGLE_APPLICATION_CREDENTIALS` | `serviceAccountKey.json` | Not set | Service account key |
| `USE_SERVICE_ACCOUNT` | `true` | `false` | Authentication method |
| `STORAGE_BUCKET` | `moments-storage` | `moments-storage` | Cloud Storage bucket |
| `FACE_RECOGNITION_ENABLED` | `true` | `true` | Enable face recognition |
| `FACE_MATCH_THRESHOLD` | `0.6` | `0.6` | Face matching threshold |

### Authentication

**Local Development:**
- Uses `serviceAccountKey.json` (same as MomentsBackend)
- Environment: `development`
- Debug mode enabled

**Production (Cloud Run):**
- Uses default credentials (Cloud Run service account)
- Environment: `production`
- No service account key needed

## 🚀 Deployment

### Automated CI/CD Pipeline

The service includes a complete serverless CI/CD pipeline:

1. **Git Push** → Triggers Cloud Build
2. **Cloud Build** → Builds Docker image
3. **Container Registry** → Stores image
4. **Cloud Run** → Deploys service
5. **Health Check** → Verifies deployment

### Manual Deployment

```bash
# Build and deploy manually
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/moments-face-tagging .

gcloud run deploy moments-face-tagging \
  --image gcr.io/YOUR_PROJECT_ID/moments-face-tagging \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 900 \
  --max-instances 10 \
  --min-instances 0 \
  --concurrency 80 \
  --port 8080
```

## 📊 API Endpoints

### Health Check
```http
GET /health
```

### Face Recognition
```http
POST /api/v1/face-embeddings/generate
POST /api/v1/face-embeddings/match
POST /api/v1/face-embeddings/process-moment
```

### Documentation
```http
GET /docs          # Swagger UI
GET /redoc         # ReDoc documentation
```

## 🧪 Testing

### Test Authentication
```bash
python test-auth.py
```

### Test CI/CD Pipeline
```bash
./test-cicd.sh
```

### Test Service
```bash
# Health check
curl https://your-service-url/health

# API documentation
open https://your-service-url/docs
```

## 📈 Monitoring

### View Logs
```bash
# Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision" --limit=20

# Build logs
gcloud builds list --limit=5
```

### Monitor Performance
- **Cloud Console** → Cloud Run → moments-face-tagging
- **Metrics**: Request count, latency, error rate
- **Logs**: Application logs, build logs
- **Traces**: Request tracing

## 🔒 Security

### Authentication
- **Local**: Service account key file (NOT COMMITTED TO GIT)
- **Production**: Cloud Run service account
- **Permissions**: Minimal required roles

### Best Practices
- **NEVER commit service account keys** - They are in `.gitignore`
- Use environment variables for configuration
- Enable audit logging
- Monitor access patterns
- Rotate service account keys regularly

### ⚠️ Important Security Notes
- `serviceAccountKey.json` is in `.gitignore` and should NEVER be committed
- Copy the service account key from your MomentsBackend project locally
- Production uses Cloud Run service account (no key files needed)
- GitHub push protection will block commits containing secrets

## 🛠️ Development

### Project Structure
```
momentsFaceTagging/
├── app/
│   ├── api/v1/endpoints/     # API endpoints
│   ├── core/                 # Core services
│   └── models/               # Data models
├── serviceAccountKey.json    # Service account key (local only, NOT COMMITTED)
├── env.local                 # Local environment config
├── env.production            # Production environment config
├── cloudbuild-trigger.yaml   # CI/CD configuration
├── setup-cicd.sh            # Automated setup script
├── test-auth.py             # Authentication testing
├── .gitignore               # Git ignore file (excludes secrets)
└── README.md                # This file
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ENVIRONMENT=development
export GOOGLE_CLOUD_PROJECT=moments-38b77
export GOOGLE_APPLICATION_CREDENTIALS=serviceAccountKey.json

# Run the service
python main.py
```

## 🚨 Troubleshooting

### Common Issues

1. **Authentication Failed**
   ```bash
   # Check service account key
   ls -la serviceAccountKey.json
   
   # Test authentication
   python test-auth.py
   ```

2. **Build Fails**
   ```bash
   # Check build logs
   gcloud builds list --limit=5
   gcloud builds log BUILD_ID
   ```

3. **Service Won't Start**
   ```bash
   # Check service logs
   gcloud logging read "resource.type=cloud_run_revision" --limit=20
   ```

### Debug Commands
```bash
# Check service status
gcloud run services describe moments-face-tagging --region=us-central1

# View recent builds
gcloud builds list --limit=10

# Check trigger status
gcloud builds triggers list
```

## 📚 Documentation

- **API Documentation**: Available at `/docs` when service is running
- **Environment Files**: `env.local` and `env.production` for configuration
- **Test Scripts**: `test-auth.py` and `test-cicd.sh` for validation
- **Setup Scripts**: `setup-cicd.sh` for automated deployment

## 🎯 Next Steps

1. **Set up monitoring** with Cloud Monitoring
2. **Configure alerts** for failures
3. **Implement logging** best practices
4. **Set up backup** and disaster recovery
5. **Optimize performance** based on usage patterns

## 📞 Support

For issues and questions:
- Check Google Cloud Run documentation
- Review application logs in Cloud Console
- Test locally with Docker before deploying
- Use the provided test scripts for validation

---

**Your Moments Face Tagging Service is now ready for production deployment!** 🚀