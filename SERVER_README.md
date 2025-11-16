# Drone Tracking Flask Server

A Flask-based web server that provides real-time visualization of drone tracking using event camera data. The system uses Kalman filtering and DBSCAN clustering to track drone movements and predict future trajectories.

## Features

- 🎥 **Real-time Video Streaming**: MJPEG streaming of tracking results
- 🎯 **Multiple Video Sources**: Automatically discovers all .dat files in the workspace
- 🤖 **Advanced Tracking**: Kalman filter-based tracking with future prediction
- 🌐 **Web Interface**: Clean, modern web interface for viewing tracking results
- 📊 **Visual Overlays**: 
  - Event feed (left side)
  - Tracking visualization (right side)
  - Bounding boxes
  - Tracking path
  - Future trajectory prediction

## Project Structure

```
.
├── app.py                      # Flask server application
├── tracking_processor.py       # Drone tracking logic
├── drone_tracking.py           # Original standalone tracking script
├── requirements.txt            # Python dependencies
├── templates/
│   ├── index.html             # Dashboard homepage
│   └── viewer.html            # Video viewer page
├── DroneHovering/             # Hovering drone test files
│   └── *.dat
├── DroneMoving/               # Moving drone test files
│   └── *.dat
└── fan/                       # Fan test files
    └── *.dat
```

## Installation

### Quick Setup (Recommended)

Run the setup script to create a virtual environment and install all dependencies:

```bash
./setup.sh
```

This will:
- Create a Python virtual environment
- Install the evio library
- Install all Python dependencies (Flask, OpenCV, NumPy, etc.)

### Manual Installation

If you prefer manual installation:

1. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install evio library**:
   ```bash
   cd evio
   pip install -e .
   cd ..
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Server

**Easy way** (automatically activates virtual environment):
```bash
./start_server.sh
```

**Manual way**:
```bash
source venv/bin/activate
python app.py
```

The server will start on `http://localhost:5000` and automatically discover all `.dat` files in the workspace.

### Accessing the Interface

1. **Dashboard**: Open `http://localhost:5000` in your browser
   - View all available tracking sources organized by category
   - Click on any source to view its tracking

2. **Direct Video Stream**: Access `http://localhost:5000/video/<route_name>`
   - Example: `http://localhost:5000/video/dronemoving/drone-moving`
   - Returns MJPEG stream suitable for embedding

3. **API Endpoint**: `http://localhost:5000/api/sources`
   - Returns JSON list of all available sources

## Available Routes

The server automatically creates routes for all `.dat` files:

- **DroneHovering/**
  - `/view/dronehovering/drone-idle`

- **DroneMoving/**
  - `/view/dronemoving/drone-moving`

- **fan/**
  - `/view/fan/fan-const-rpm`
  - `/view/fan/fan-varying-rpm`
  - `/view/fan/fan-varying-rpm-turning`

## Tracking Visualization

The combined view shows:

### Left Side - Event Feed
- White pixels: Positive polarity events
- Gray pixels: Negative polarity events
- Raw event camera output

### Right Side - Tracking View
- **Yellow dots**: Event cluster points
- **Red box**: Object bounding box
- **Green line**: Historical tracking path
- **Blue circle**: Kalman filter position (current)
- **Cyan circle**: Smoothed raw detection
- **Red dots**: Future trajectory prediction (10 steps ahead)

## Tracking Algorithm

The system uses a multi-stage tracking pipeline:

1. **Event Decoding**: Decode event camera data from .dat files
2. **Background Filtering**: Adaptive background activity filtering
3. **DBSCAN Clustering**: Cluster events to detect objects
4. **Kalman Filtering**: Smooth tracking with state prediction
5. **Exponential Smoothing**: Additional smoothing on raw detections
6. **Outlier Rejection**: Filter large jumps in position
7. **Future Prediction**: Predict future trajectory based on velocity

## Configuration

You can modify tracking parameters in `tracking_processor.py`:

```python
window_ms = 2              # Event window duration in milliseconds
alpha = 0.6                # Exponential smoothing factor (0-1)
max_jump = 150             # Maximum allowed jump in pixels
eps = 20                   # DBSCAN epsilon (clustering distance)
min_samples = 30           # DBSCAN minimum samples per cluster
```

## API Reference

### GET /
Returns the dashboard HTML page with all available sources.

### GET /view/<route_name>
Returns the viewer HTML page for a specific source.

### GET /video/<route_name>
Returns an MJPEG video stream for a specific source.

### GET /api/sources
Returns JSON array of all available sources:
```json
[
  {
    "route": "dronemoving/drone-moving",
    "file": "drone_moving.dat",
    "view_url": "/view/dronemoving/drone-moving",
    "stream_url": "/video/dronemoving/drone-moving"
  }
]
```

## Notes

- The server runs in threaded mode to handle multiple concurrent streams
- Each viewer creates a new tracking instance
- Event processing happens in real-time as frames are requested
- The evio library handles the event camera data format

## Troubleshooting

**Stream not loading:**
- Check that the .dat file exists and is readable
- Verify evio library is installed correctly
- Check server console for error messages

**Slow performance:**
- Reduce `window_ms` parameter
- Increase DBSCAN `eps` or decrease `min_samples`
- Check system resources (CPU/memory)

**No sources found:**
- Verify .dat files are in DroneHovering/, DroneMoving/, or fan/ directories
- Check file permissions
- Restart the server to re-discover files

## Dependencies

- Flask: Web framework
- OpenCV: Image processing and video encoding
- NumPy: Numerical operations
- FilterPy: Kalman filter implementation
- scikit-learn: DBSCAN clustering
- evio: Event camera data handling

## License

See the project LICENSE file for details.
