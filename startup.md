# 🚀 Detectify Startup Guide (Windows)

Follow these steps to get the project running on your local machine.

## 📋 Prerequisites

- Python 3.10 or higher
- pip package manager
- Web browser (Chrome, Firefox, Edge)

---

## 🛠️ Step-by-Step Setup

### Step 1: Install Dependencies
Install all required Python libraries using the requirements file:
```powershell
pip install -r requirements.txt
pip install ultralytics
```

**Expected output:** All packages should install successfully without errors.

---

### Step 2: Initialize the Database
Create the local SQLite database and necessary tables:
```powershell
$env:PYTHONPATH = "src"
python scripts/init_db.py
```

**Expected output:** 
```
Database initialized successfully!
Tables created: detections
```

---

### Step 3: Start the API Server
Launch the REST API server (this will enable the dashboard):
```powershell
$env:PYTHONPATH = "src"
python -m detectify serve --host 0.0.0.0 --port 8001 --reload
```

**Alternative method:**
```powershell
uvicorn detectify.api.server:app --host 0.0.0.0 --port 8001 --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

---

### Step 4: Open the Dashboard
Once the server is running, open your web browser and navigate to:

**🌐 Dashboard URL:**
```
http://localhost:8001/dashboard
```

**What you'll see:**
- Real-time detection statistics
- Detection history table with images
- Links to API documentation and live stream

**Other useful URLs:**
- **API Docs (Swagger):** http://localhost:8001/docs
- **API Docs (ReDoc):** http://localhost:8001/redoc
- **Health Check:** http://localhost:8001/health
- **Live Stream:** http://localhost:8001/detect/live
- **JSON API:** http://localhost:8001/detections

---

### Step 5: Generate Some Detections (Optional)
To see data in the dashboard, you need to run some detections first:

**Option A: Run Live Webcam Detection**
```powershell
$env:PYTHONPATH = "src"
python -m detectify inference
```
Press `q` to quit when done.

**Option B: Detect from an Image**
```powershell
$env:PYTHONPATH = "src"
python -m detectify inference --source "path/to/your/image.jpg" --output "output.jpg"
```

**Option C: Upload via API**
```powershell
curl -X POST -F "file=@your_image.jpg" http://localhost:8001/detect?format=json
```

After running detections, refresh the dashboard to see the results!

---

## 🎯 Quick Start Commands Summary

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set Python path
$env:PYTHONPATH = "src"

# 3. Initialize database
python scripts/init_db.py

# 4. Start server (in one terminal)
python -m detectify serve --reload

# 5. Open dashboard in browser
# Navigate to: http://localhost:8001/dashboard

# 6. (Optional) Run detections in another terminal
python -m detectify inference
```

---

## 🔧 Additional Features

### ESP32 Dev Module (Pyramid) Sync
Test the connection and sync alerts to your ESP32 hardware:
```powershell
# Test connection
$env:PYTHONPATH = "src"
python -m detectify iot-test

# Run with ESP32-CAM stream (optional)
$env:PYTHONPATH = "src"
python -m detectify inference --source "http://192.168.1.100:81/stream"
```

### View Live Detection Stream
Open in browser:
```
# For local webcam:
http://localhost:8001/detect/live

# For ESP32 or IP Camera:
http://localhost:8001/detect/live?camera=http://192.168.1.100/cam1
```

### Query Detections via API
```powershell
# Get all detections
curl http://localhost:8001/detections

# Filter by class
curl "http://localhost:8001/detections?class_name=person&limit=10"

# Filter by camera
curl "http://localhost:8001/detections?camera_id=0&limit=20"
```

---

## ⚠️ Troubleshooting

**Problem:** Dashboard shows "No detections found"
- **Solution:** Run some detections first using Step 5 above

**Problem:** Server won't start - `[WinError 10013]` Port access forbidden
- **Solution 1 (Recommended):** Use a different port:
  ```powershell
  python -m detectify serve --port 8001
  ```
  Then access dashboard at: `http://localhost:8001/dashboard`

- **Solution 2:** Check what's using port 8000 and kill it:
  ```powershell
  # Find process using port 8000
  netstat -ano | findstr :8000
  
  # Kill the process (replace PID with the number from above)
  taskkill /PID <PID> /F
  ```

- **Solution 3:** Run PowerShell as Administrator and try again

- **Solution 4:** Change port in .env file:
  ```
  API_PORT=8001
  ```

**Problem:** Import errors
- **Solution:** Make sure `$env:PYTHONPATH = "src"` is set in your terminal

**Problem:** Database errors - "no such column" errors
- **Solution:** Reinitialize the database (this will recreate tables with correct schema):
  ```powershell
  $env:PYTHONPATH = "src"
  python scripts/init_db.py
  ```
  **Note:** This will drop and recreate all tables. Any existing detection data will be lost.

---

> [!IMPORTANT]
> **Email Alerts**: To receive emails, you must update the `SMTP_PASSWORD` in your `.env` with a **Gmail App Password**.
