# KhushiDL — Ham V/s Spam Backend

FastAPI backend that exposes a REST API for classifying messages as **Ham** (legitimate) or **Spam** using a pre-trained Keras / TensorFlow deep learning model.

---

## 📁 Project Structure

```
Backend/
├── main.py            # FastAPI application with Keras model inference
├── requirements.txt   # Python dependencies
├── render.yaml        # Render deployment Blueprint specification
├── .gitignore
└── Model/
    └── model.keras    # Pre-trained Keras model with internal TextVectorization
```

---

## 🚀 Running Locally

```bash
# Navigate to the Backend directory
cd Backend

# Create a virtual environment (Python 3.11 or 3.12 recommended)
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the development server
uvicorn main:app --reload --port 8000
```

Interactive API documentation will be available at:
👉 **http://localhost:8000/docs**

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root health check and service info |
| `GET` | `/health` | Detailed health & model status |
| `POST` | `/predict` | Predict Ham vs Spam for a given message |

### Example Request (`POST /predict`)

```json
{
  "message": "URGENT: You have won a free $1000 Walmart gift card! Call 0800 now to claim!"
}
```

### Example Response

```json
{
  "label": "Spam",
  "confidence": 0.9882,
  "message": "URGENT: You have won a free $1000 Walmart gift card! Call 0800 now to claim!",
  "probability": 0.9882
}
```

---

## ☁️ Deploying to Render

This repository includes a `render.yaml` Blueprint file configured for one-click deployment:

1. Push this repository to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com), select **New → Blueprint**.
3. Connect your repository and Render will automatically detect `render.yaml`.
4. Deploy the service.
