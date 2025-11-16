import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from sklearn.cluster import DBSCAN
from evio.source.dat_file import DatFileSource


class DroneTracker:
    """
    Processes event camera data and tracks drone movements using Kalman filtering
    """
    
    def __init__(self, dat_file, window_ms=2, width=1280, height=720):
        self.dat_file = dat_file
        self.window_ms = window_ms
        self.width = width
        self.height = height
        self.src = DatFileSource(dat_file, width=width, height=height, window_length_us=window_ms * 1000)
        self.kf = self._init_kalman()
        self.path_points = []
        self.previous_cluster_info = None
        self.prev_raw_x = None
        self.prev_raw_y = None
        self.alpha = 0.6  # smoothing factor
        self.max_jump = 150  # max allowed jump in pixels
        
        # Background activity map
        self.background_activity = np.zeros((height, width), dtype=np.float32)
        self.decay_rate = 0.95
        self.activity_threshold = 50
    
    def _init_kalman(self):
        """Initialize Kalman filter for tracking"""
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
    
    def _decode_window(self, event_words, time_order, start, stop):
        """Decode event data from the window"""
        event_indexes = time_order[start:stop]
        words = event_words[event_indexes].astype(np.uint32, copy=False)
        x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
        y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
        polarities = ((words >> 28) & 0xF) > 0
        return x_coords, y_coords, polarities
    
    def _events_to_frame(self, window):
        """Convert events to a grayscale frame"""
        x_coords, y_coords, polarities = window
        frame = np.zeros((self.height, self.width), dtype=np.uint8)
        frame[y_coords[polarities], x_coords[polarities]] = 255
        frame[y_coords[~polarities], x_coords[~polarities]] = 128
        return frame
    
    def _filter_background(self, x_coords, y_coords):
        """Filter out background activity"""
        self.background_activity *= self.decay_rate
        for x, y in zip(x_coords, y_coords):
            if 0 <= x < self.width and 0 <= y < self.height:
                self.background_activity[y, x] += 1
        mask = self.background_activity[y_coords, x_coords] < self.activity_threshold
        return x_coords[mask], y_coords[mask]
    
    def _cluster_events_all(self, x_coords, y_coords, eps=20, min_samples=30, max_points=8000):
        """Cluster events using DBSCAN"""
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
    
    def _compute_cluster_info(self, cluster_points):
        """Compute cluster center, size, and bounding box"""
        min_x, min_y = np.min(cluster_points, axis=0)
        max_x, max_y = np.max(cluster_points, axis=0)
        bbox = (int(min_x), int(min_y), int(max_x), int(max_y))
        size = len(cluster_points)
        center = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
        return center, size, bbox
    
    def _select_cluster_with_memory(self, clusters, size_tolerance=0.5):
        """Select the best cluster based on previous tracking info"""
        if not clusters:
            return None, None, None
        cluster_infos = []
        for cx, cy, points in clusters:
            center, size, bbox = self._compute_cluster_info(points)
            cluster_infos.append((center, size, bbox, points))
        if self.previous_cluster_info is None:
            best = max(cluster_infos, key=lambda c: c[1])
            return best[0], best[3], (best[0], best[1], best[2])
        prev_center, prev_size, prev_bbox = self.previous_cluster_info
        best = min(cluster_infos, key=lambda c: np.hypot(c[0][0]-prev_center[0], c[0][1]-prev_center[1]))
        if abs(best[1] - prev_size) / prev_size > size_tolerance:
            return prev_center, None, self.previous_cluster_info
        return best[0], best[3], (best[0], best[1], best[2])
    
    def _predict_future(self, steps=10):
        """Predict future trajectory"""
        future_points = []
        temp_kf = KalmanFilter(dim_x=4, dim_z=2)
        temp_kf.x = self.kf.x.copy()
        temp_kf.F = self.kf.F.copy()
        for _ in range(steps):
            temp_kf.predict()
            future_points.append((int(temp_kf.x[0]), int(temp_kf.x[1])))
        return future_points
    
    def process_frames(self):
        """
        Generator that yields tracking frames as JPEG images
        """
        for batch_range in self.src.ranges():
            window = self._decode_window(self.src.event_words, self.src.order, batch_range.start, batch_range.stop)
            video_frame = self._events_to_frame(window)
            video_frame_bgr = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2BGR)
            
            tracking_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            x_coords, y_coords = self._filter_background(window[0], window[1])
            clusters = self._cluster_events_all(x_coords, y_coords)
            cluster_center, cluster_points, self.previous_cluster_info = self._select_cluster_with_memory(clusters)
            
            # Kalman predict first
            self.kf.predict()
            smooth_x, smooth_y = int(self.kf.x[0]), int(self.kf.x[1])
            
            if cluster_center:
                raw_cx, raw_cy = cluster_center
                
                # Exponential smoothing
                if self.prev_raw_x is not None:
                    raw_cx = int(self.alpha * raw_cx + (1 - self.alpha) * self.prev_raw_x)
                    raw_cy = int(self.alpha * raw_cy + (1 - self.alpha) * self.prev_raw_y)
                
                # Outlier rejection
                if self.prev_raw_x is not None and np.hypot(raw_cx - self.prev_raw_x, raw_cy - self.prev_raw_y) > self.max_jump:
                    raw_cx, raw_cy = smooth_x, smooth_y
                else:
                    self.kf.update((raw_cx, raw_cy))
                
                self.prev_raw_x, self.prev_raw_y = raw_cx, raw_cy
                smooth_x, smooth_y = int(self.kf.x[0]), int(self.kf.x[1])
                self.path_points.append((smooth_x, smooth_y))
                
                # Draw cluster points safely
                if cluster_points is not None:
                    for pt in cluster_points:
                        cv2.circle(tracking_frame, tuple(pt), 2, (255, 255, 0), -1)
                
                # Draw bounding box
                if self.previous_cluster_info:
                    _, _, bbox = self.previous_cluster_info
                    cv2.rectangle(tracking_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            
            # Draw tracking path
            for i in range(1, len(self.path_points)):
                cv2.line(tracking_frame, self.path_points[i-1], self.path_points[i], (0, 255, 0), 2)
            
            # Draw current positions
            cv2.circle(tracking_frame, (smooth_x, smooth_y), 8, (255, 0, 0), -1)  # Kalman
            if self.prev_raw_x and self.prev_raw_y:
                cv2.circle(tracking_frame, (self.prev_raw_x, self.prev_raw_y), 5, (0, 255, 255), -1)  # Smoothed raw
            
            # Predict future trajectory
            future_points = self._predict_future(steps=10)
            for pt in future_points:
                cv2.circle(tracking_frame, pt, 4, (0, 0, 255), -1)
            
            # Combine video feed and tracking view side by side
            combined_frame = np.hstack([video_frame_bgr, tracking_frame])
            
            # Add vertical divider bar between the two views (white line, 3 pixels wide)
            divider_x = self.width
            cv2.line(combined_frame, (divider_x, 0), (divider_x, self.height), (255, 255, 255), 3)
            
            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                yield buffer.tobytes()
