# Ham vs Spam – Backend

FastAPI backend that exposes a REST API for classifying messages as **Ham** (legitimate) or **Spam** using a pre-trained TensorFlow/Keras model.

---

## 📁 Project Structure

```
Backend/
├── main.py            # FastAPI application entry-point
├── requirements.txt   # Python dependencies
├── render.yaml        # Render deployment configuration
├── .gitignore
└── Model/
    ├── model.keras    # Pre-trained Keras model
    └── tokenizer.pkl  # (optional) Keras tokenizer
```

---

## 🚀 Running Locally

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the development server
uvicorn main:app --reload --port 8000
```

API docs available at → **http://localhost:8000/docs**

---

## 🌐 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed health (model / tokenizer status) |
| `POST` | `/predict` | Classify a message |

### `POST /predict`

**Request body:**
```json
{ "message": "Congratulations! You've won a free iPhone." }
```

**Response:**
```json
{
  "label": "Spam",
  "confidence": 0.9874,
  "message": "Congratulations! You've won a free iPhone."
}
```

---

## ☁️ Deploy on Render

The `render.yaml` file in this directory configures everything automatically.

1. Push this repo to GitHub.
2. On [render.com](https://render.com), click **New → Blueprint** and point it to this repo.
3. Render will detect `render.yaml` and deploy the service.

Alternatively, create a **Web Service** manually:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_ORIGINS` | `*` | Comma-separated list of allowed CORS origins (e.g. `https://myapp.vercel.app`) |
| `PORT` | `8000` | Port (automatically set by Render) |
