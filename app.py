from flask import Flask, Response, render_template, jsonify
import os
import glob
from tracking_processor import DroneTracker

app = Flask(__name__)

# Discover all .dat files in the workspace
DATA_FILES = {}

def discover_dat_files():
    """Discover all .dat files in the workspace"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Define the directories to search
    directories = ['DroneHovering', 'DroneMoving', 'fan']
    
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        if os.path.exists(dir_path):
            dat_files = glob.glob(os.path.join(dir_path, '*.dat'))
            for dat_file in dat_files:
                # Create a route-friendly name
                filename = os.path.basename(dat_file)
                route_name = filename.replace('.dat', '').replace('_', '-')
                full_route = f"{directory.lower()}/{route_name}"
                DATA_FILES[full_route] = dat_file

# Discover files on startup
discover_dat_files()

def generate_frames(dat_file_path):
    """
    Generator function for video streaming.
    Yields MJPEG frames from the tracking processor.
    """
    tracker = DroneTracker(dat_file_path)
    
    for frame in tracker.process_frames():
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    """Homepage listing all available tracking routes"""
    return render_template('index.html', data_files=DATA_FILES)


@app.route('/video/<path:route_name>')
def video_feed(route_name):
    """Video streaming route. Returns MJPEG stream."""
    if route_name not in DATA_FILES:
        return "Video source not found", 404
    
    dat_file = DATA_FILES[route_name]
    return Response(generate_frames(dat_file),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/view/<path:route_name>')
def view_tracking(route_name):
    """Display page for a specific tracking video"""
    if route_name not in DATA_FILES:
        return "Video source not found", 404
    
    return render_template('viewer.html', route_name=route_name, title=route_name.replace('-', ' ').title())


@app.route('/api/sources')
def list_sources():
    """API endpoint to list all available video sources"""
    sources = []
    for route_name, file_path in DATA_FILES.items():
        sources.append({
            'route': route_name,
            'file': os.path.basename(file_path),
            'view_url': f'/view/{route_name}',
            'stream_url': f'/video/{route_name}'
        })
    return jsonify(sources)


if __name__ == '__main__':
    print("Available tracking routes:")
    for route_name, file_path in DATA_FILES.items():
        print(f"  - /view/{route_name}")
    
    # Get port from environment variable (for Render) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    print(f"\nStarting Flask server on http://0.0.0.0:{port}")
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
