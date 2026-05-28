import copy
import numpy as np
from filterpy.kalman import KalmanFilter

if not hasattr(np, 'asfarray'):
    np.asfarray = lambda x: np.asarray(x, dtype=np.float64)

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.trace = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.state = 0

    def apply_gmc(self, warp_matrix):
        pos = np.array([self.kf.x[0, 0], self.kf.x[1, 0], 1.0])
        new_pos = warp_matrix @ pos
        self.kf.x[0, 0] = new_pos[0]
        self.kf.x[1, 0] = new_pos[1]

    def update(self, bbox, confidence=None):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        if confidence is not None:
            r_factor = (1.0 - confidence) * 20.0
            self.kf.R = np.diag([1.0, 1.0, 10.0, 10.0]) + np.diag([r_factor] * 4)

        self.kf.update(self.convert_bbox_to_z(bbox))

        if self.state == 0 and self.hits >= 3:
            self.state = 1

        bbox_xyxy = self.get_state()[0]
        self.trace.append(self.bbox_center(bbox_xyxy))

    def predict(self):
        if (self.kf.x[6, 0] + self.kf.x[2, 0]) <= 0:
            self.kf.x[6, 0] *= 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        predicted = self.convert_x_to_bbox(self.kf.x)
        self.history.append(predicted)
        return predicted

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    def get_trace(self):
        return list(self.trace)

    def predict_future(self, steps=5):
        """
        Dự báo vị trí tương lai của đối tượng
        
        Args:
            steps: Số bước dự báo
            
        Returns:
            Danh sách các điểm tâm dự báo, loại bỏ các điểm không hợp lệ
        """
        future_centers = []
        future_kf = copy.deepcopy(self.kf)
        for _ in range(steps):
            future_kf.predict()
            
            # Đảm bảo area > 0 để tránh NaN từ sqrt
            if future_kf.x[2, 0] <= 0:
                future_kf.x[2, 0] = max(future_kf.x[2, 0], 1)
            
            try:
                bbox = self.convert_x_to_bbox(future_kf.x)[0]
                
                # Kiểm tra xem bbox có hợp lệ không
                x1, y1, x2, y2 = bbox
                if np.isnan([x1, y1, x2, y2]).any() or x1 >= x2 or y1 >= y2:
                    continue  # Bỏ qua waypoint không hợp lệ
                
                center = self.bbox_center(bbox)
                future_centers.append(center)
            except (ValueError, RuntimeWarning):
                # Bỏ qua nếu không thể tính toán
                continue
        
        return future_centers

    def predict_future_by_time(self, prediction_horizon=3.0, fps=30.0):
        """
        Dự báo quỹ đạo dựa trên thời gian dự báo (time horizon).
        
        Công thức: L = v × t_horizon
        - L: Độ dài đường dự báo (pixel)
        - v: Vận tốc hiện tại (pixel/frame)
        - t_horizon: Khoảng thời gian dự báo (giây)
        
        Số bước dự báo = prediction_horizon(s) × fps(frame/s)
        
        Args:
            prediction_horizon: Khoảng thời gian dự báo (giây), default 3.0s
            fps: Frame per second của video, default 30
            
        Returns:
            Danh sách các điểm tâm dự báo trong time horizon
        """
        # Tính số frame cần dự báo
        steps = int(prediction_horizon * fps)
        return self.predict_future(steps=steps)
    
    def get_prediction_distance(self, prediction_horizon=3.0):
        """
        Tính khoảng cách dự báo dựa trên vận tốc hiện tại.
        
        Công thức: L = v × t_horizon
        
        Args:
            prediction_horizon: Khoảng thời gian dự báo (giây)
            
        Returns:
            Khoảng cách (pixel) sẽ đi được trong time horizon
        """
        vx = self.kf.x[4, 0]  # Vận tốc X (pixel/frame)
        vy = self.kf.x[5, 0]  # Vận tốc Y (pixel/frame)
        
        # Tính độ lớn vận tốc (pixel/frame)
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        
        # Khoảng cách = vận tốc × thời gian dự báo × fps
        # Ở đây ta giả sử 1 frame = 1 đơn vị thời gian
        # Nên: khoảng cách = v(pixel/frame) × time_horizon(s) × fps(frame/s)
        prediction_distance = velocity_magnitude * prediction_horizon
        
        return prediction_distance
    
    def get_motion_info(self, prediction_horizon=3.0):
        """
        Lấy thông tin chuyển động túc thì (hiện tại) và dự báo.
        
        Args:
            prediction_horizon: Khoảng thời gian dự báo (giây)
            
        Returns:
            Dict chứa thông tin chuyển động
        """
        import math
        
        vx = self.kf.x[4, 0]
        vy = self.kf.x[5, 0]
        
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        motion_angle = math.degrees(math.atan2(vy, vx)) if velocity_magnitude > 0 else 0
        
        prediction_distance = self.get_prediction_distance(prediction_horizon)
        
        return {
            'current_position': (self.kf.x[0, 0], self.kf.x[1, 0]),
            'velocity': {
                'vx': vx,
                'vy': vy,
                'magnitude': velocity_magnitude
            },
            'motion_angle': motion_angle,
            'is_moving': velocity_magnitude > 0.5,
            'prediction_horizon': prediction_horizon,
            'predicted_distance': prediction_distance
        }

    @staticmethod
    def bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / (w + 1e-6)
        if score is None:
            return np.array([x[0] - w / 2.0, x[1] - h / 2.0,
                             x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2.0, x[1] - h / 2.0,
                             x[0] + w / 2.0, x[1] + h / 2.0,
                             score]).reshape((1, 5))