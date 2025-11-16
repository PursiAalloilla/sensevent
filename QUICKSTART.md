# 🚁 Drone Tracking Server - Quick Start Guide

## ✅ Server is Running!

The Flask server is now running and ready to use.

### 📍 Access Points

**Main Dashboard:** http://localhost:5000

**Individual Tracking Views:**
- Drone Hovering: http://localhost:5000/view/dronehovering/drone-idle
- Drone Moving: http://localhost:5000/view/dronemoving/drone-moving
- Fan (Constant RPM): http://localhost:5000/view/fan/fan-const-rpm
- Fan (Varying RPM): http://localhost:5000/view/fan/fan-varying-rpm
- Fan (Varying RPM Turning): http://localhost:5000/view/fan/fan-varying-rpm-turning

**API Endpoint:** http://localhost:5000/api/sources

### 🎮 Controls

- Open any link above in your web browser
- The dashboard shows all available video sources
- Click on any source to view real-time tracking
- Press `Ctrl+C` in the terminal to stop the server

### 📊 What You'll See

Each tracking view shows:
- **Left side:** Raw event camera feed
- **Right side:** Tracking visualization with:
  - Yellow dots: Event clusters
  - Red box: Bounding box
  - Green line: Tracking path
  - Blue circle: Kalman filter position
  - Cyan circle: Raw detection
  - Red dots: Future prediction

### 🔄 Restarting the Server

To start the server again:
```bash
./start_server.sh
```

Or manually:
```bash
source venv/bin/activate
python app.py
```

### 📝 More Information

See `SERVER_README.md` for complete documentation.
