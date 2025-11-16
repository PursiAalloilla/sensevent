
import cv2
import numpy as np
import time
from filterpy.kalman import KalmanFilter
from sklearn.cluster import DBSCAN
from evio.source.dat_file import DatFileSource

# -----------------------------
# Event Decoding
# -----------------------------
def decode_window(event_words, time_order, start, stop):
    event_indexes = time_order[start:stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    polarities = ((words >> 28) & 0xF) > 0
    return x_coords, y_coords, polarities

# -----------------------------
# Frame Generation
# -----------------------------
def events_to_frame(window, width=1280, height=720):
    x_coords, y_coords, polarities = window
    frame = np.zeros((height, width), dtype=np.uint8)
    frame[y_coords[polarities], x_coords[polarities]] = 255
    frame[y_coords[~polarities], x_coords[~polarities]] = 128
    return frame

# -----------------------------
# Kalman Filter Initialization
# -----------------------------
def init_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0., 0., 0., 0.])
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 1000
    kf.R *= 2    # Lower measurement noise for smoother updates
    kf.Q *= 0.05 # Slightly higher process noise for responsiveness
    return kf

# -----------------------------
# Future Trajectory Prediction
# -----------------------------
def predict_future(kf, steps=10):
    future_points = []
    temp_kf = KalmanFilter(dim_x=4, dim_z=2)
    temp_kf.x = kf.x.copy()
    temp_kf.F = kf.F.copy()
    for _ in range(steps):
        temp_kf.predict()
        future_points.append((int(temp_kf.x[0]), int(temp_kf.x[1])))
    return future_points

# -----------------------------
# Background Activity Map
# -----------------------------
background_activity = np.zeros((720, 1280), dtype=np.float32)
decay_rate = 0.95
activity_threshold = 50

def filter_background(x_coords, y_coords):
    global background_activity
    background_activity *= decay_rate
    for x, y in zip(x_coords, y_coords):
        if 0 <= x < 1280 and 0 <= y < 720:
            background_activity[y, x] += 1
    mask = background_activity[y_coords, x_coords] < activity_threshold
    return x_coords[mask], y_coords[mask]

# -----------------------------
# Cluster All + Memory Check
# -----------------------------
def cluster_events_all(x_coords, y_coords, eps=20, min_samples=30, max_points=8000):
    points = np.column_stack((x_coords, y_coords))
    if len(points) == 0:
        return []
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
    clustering = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree').fit(points)
    labels = clustering.labels_
    clusters = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_points = points[labels == label]
        cx, cy = np.mean(cluster_points, axis=0).astype(int)
        clusters.append((cx, cy, cluster_points))
    return clusters

def compute_cluster_info(cluster_points):
    min_x, min_y = np.min(cluster_points, axis=0)
    max_x, max_y = np.max(cluster_points, axis=0)
    bbox = (int(min_x), int(min_y), int(max_x), int(max_y))
    size = len(cluster_points)
    center = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
    return center, size, bbox

def select_cluster_with_memory(previous_info, clusters, size_tolerance=0.5):
    if not clusters:
        return None, None, None
    cluster_infos = []
    for cx, cy, points in clusters:
        center, size, bbox = compute_cluster_info(points)
        cluster_infos.append((center, size, bbox, points))
    if previous_info is None:
        best = max(cluster_infos, key=lambda c: c[1])
        return best[0], best[3], (best[0], best[1], best[2])
    prev_center, prev_size, prev_bbox = previous_info
    best = min(cluster_infos, key=lambda c: np.hypot(c[0][0]-prev_center[0], c[0][1]-prev_center[1]))
    if abs(best[1] - prev_size) / prev_size > size_tolerance:
        return prev_center, None, previous_info
    return best[0], best[3], (best[0], best[1], best[2])

# -----------------------------
# Main Tracking and Display Loop
# -----------------------------
def track_and_display(dat_file, window_ms=2, target_fps=30):
    width, height = 1280, 720
    src = DatFileSource(dat_file, width=width, height=height, window_length_us=window_ms * 1000)
    kf = init_kalman()
    path_points = []
    previous_cluster_info = None
    prev_raw_x, prev_raw_y = None, None
    alpha = 0.6  # smoothing factor
    max_jump = 150  # max allowed jump in pixels

    cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Feed", width, height)
    cv2.namedWindow("Tracking View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking View", width, height)

    frame_interval = 1.0 / target_fps
    last_time = time.time()

    for batch_range in src.ranges():
        window = decode_window(src.event_words, src.order, batch_range.start, batch_range.stop)
        video_frame = events_to_frame(window, width, height)
        video_frame_bgr = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2BGR)

        tracking_frame = np.zeros((height, width, 3), dtype=np.uint8)

        x_coords, y_coords = filter_background(window[0], window[1])
        clusters = cluster_events_all(x_coords, y_coords)
        cluster_center, cluster_points, previous_cluster_info = select_cluster_with_memory(previous_cluster_info, clusters)

        # Kalman predict first
        kf.predict()
        smooth_x, smooth_y = int(kf.x[0]), int(kf.x[1])

        if cluster_center:
            raw_cx, raw_cy = cluster_center

            # Exponential smoothing
            if prev_raw_x is not None:
                raw_cx = int(alpha * raw_cx + (1 - alpha) * prev_raw_x)
                raw_cy = int(alpha * raw_cy + (1 - alpha) * prev_raw_y)

            # Outlier rejection
            if prev_raw_x is not None and np.hypot(raw_cx - prev_raw_x, raw_cy - prev_raw_y) > max_jump:
                raw_cx, raw_cy = smooth_x, smooth_y
            else:
                kf.update((raw_cx, raw_cy))

            prev_raw_x, prev_raw_y = raw_cx, raw_cy
            smooth_x, smooth_y = int(kf.x[0]), int(kf.x[1])
            path_points.append((smooth_x, smooth_y))

            # Draw cluster points safely
            if cluster_points is not None:
                for pt in cluster_points:
                    cv2.circle(tracking_frame, tuple(pt), 2, (255, 255, 0), -1)

            # Draw bounding box
            if previous_cluster_info:
                _, _, bbox = previous_cluster_info
                cv2.rectangle(tracking_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

        # Draw tracking path
        for i in range(1, len(path_points)):
            cv2.line(tracking_frame, path_points[i-1], path_points[i], (0, 255, 0), 2)

        # Draw current positions
        cv2.circle(tracking_frame, (smooth_x, smooth_y), 8, (255, 0, 0), -1)  # Kalman
        if prev_raw_x and prev_raw_y:
            cv2.circle(tracking_frame, (prev_raw_x, prev_raw_y), 5, (0, 255, 255), -1)  # Smoothed raw

        # Predict future trajectory
        future_points = predict_future(kf, steps=10)
        for pt in future_points:
            cv2.circle(tracking_frame, pt, 4, (0, 0, 255), -1)

        cv2.imshow("Video Feed", video_frame_bgr)
        cv2.imshow("Tracking View", tracking_frame)

        elapsed = time.time() - last_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        last_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Run
track_and_display("drone_moving.dat")
