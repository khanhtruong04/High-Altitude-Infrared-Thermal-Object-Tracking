"""
State Estimation Analysis Module
Extracts and analyzes Kalman Filter state information for tracking objects.

This module provides utilities to:
1. Extract state vectors from Kalman Filter
2. Analyze motion parameters (velocity, acceleration)
3. Predict future trajectories
4. Export state estimation data for visualization and analysis
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional


class StateEstimationAnalyzer:
    """
    Phân tích State Estimation từ Kalman Filter
    """
    
    def __init__(self):
        """Khởi tạo Analyzer"""
        self.state_history = {}  # {track_id: [(frame_idx, state_vector), ...]}
        self.motion_metrics = {}  # {track_id: motion_info}
    
    def extract_state(self, tracker, frame_idx: int, track_id: int) -> Dict:
        """
        Trích xuất thông tin State từ Kalman Tracker
        
        Args:
            tracker: KalmanBoxTracker instance
            frame_idx: Chỉ số frame hiện tại
            track_id: ID của tracker
            
        Returns:
            Dictionary chứa state estimation info
        """
        state = {
            'frame': frame_idx,
            'track_id': track_id,
            'timestamp': frame_idx / 30.0,  # Assuming 30 FPS
            
            # Position (x, y)
            'position': {
                'x': float(tracker.kf.x[0, 0]),
                'y': float(tracker.kf.x[1, 0])
            },
            
            # Size and aspect ratio
            'size': {
                'area': float(tracker.kf.x[2, 0]),
                'aspect_ratio': float(tracker.kf.x[3, 0])
            },
            
            # Velocity (vx, vy) - trích từ state vector
            'velocity': {
                'vx': float(tracker.kf.x[4, 0]),
                'vy': float(tracker.kf.x[5, 0]),
                'magnitude': float(np.sqrt(tracker.kf.x[4, 0]**2 + tracker.kf.x[5, 0]**2))
            },
            
            # Aspect ratio rate
            'aspect_ratio_rate': float(tracker.kf.x[6, 0]),
            
            # Track quality metrics
            'hits': tracker.hits,
            'hit_streak': tracker.hit_streak,
            'age': tracker.age,
            'time_since_update': tracker.time_since_update,
            'state': tracker.state,  # 0: tentative, 1: confirmed
            
            # Kalman Filter covariance (uncertainty measure)
            'uncertainty': {
                'position_uncertainty': float(np.sqrt(tracker.kf.P[0, 0] + tracker.kf.P[1, 1])),
                'velocity_uncertainty': float(np.sqrt(tracker.kf.P[4, 4] + tracker.kf.P[5, 5]))
            }
        }
        
        # Lưu state vào history
        if track_id not in self.state_history:
            self.state_history[track_id] = []
        self.state_history[track_id].append(state)
        
        return state
    
    def predict_trajectory(self, tracker, steps: int = 5, 
                          dt: float = 1.0) -> List[Tuple[float, float]]:
        """
        Dự báo quỹ đạo tương lai
        
        Args:
            tracker: KalmanBoxTracker instance
            steps: Số bước dự báo
            dt: Khoảng thời gian giữa các bước (default: 1 frame)
            
        Returns:
            Danh sách các điểm dự báo [(x1, y1), (x2, y2), ...]
        """
        import copy
        
        trajectory = []
        future_kf = copy.deepcopy(tracker.kf)
        
        for _ in range(steps):
            future_kf.predict()
            x = future_kf.x[0, 0]
            y = future_kf.x[1, 0]
            trajectory.append((x, y))
        
        return trajectory
    
    def extract_motion_parameters(self, tracker) -> Dict:
        """
        Trích xuất các thông số chuyển động
        
        Args:
            tracker: KalmanBoxTracker instance
            
        Returns:
            Dictionary chứa motion parameters
        """
        vx = float(tracker.kf.x[4, 0])
        vy = float(tracker.kf.x[5, 0])
        
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        
        # Tính góc chuyển động
        if velocity_magnitude > 0:
            angle = np.degrees(np.arctan2(vy, vx))
        else:
            angle = 0
        
        return {
            'velocity_x': vx,
            'velocity_y': vy,
            'velocity_magnitude': velocity_magnitude,
            'motion_angle': angle,
            'is_moving': velocity_magnitude > 0.5,  # Threshold für moving detection
            'trace_length': len(tracker.trace)
        }
    
    def get_state_history(self, track_id: Optional[int] = None) -> Dict:
        """
        Lấy lịch sử state estimation
        
        Args:
            track_id: ID của tracker (nếu None trả về tất cả)
            
        Returns:
            Dictionary chứa state history
        """
        if track_id is not None:
            return {track_id: self.state_history.get(track_id, [])}
        return self.state_history
    
    def export_state_to_json(self, output_path: str, track_id: Optional[int] = None) -> str:
        """
        Xuất state estimation data sang JSON
        
        Args:
            output_path: Đường dẫn file output
            track_id: ID của tracker (nếu None xuất tất cả)
            
        Returns:
            Đường dẫn file đã tạo
        """
        data = {
            'state_estimation_data': self.get_state_history(track_id),
            'summary': {
                'total_tracks': len(self.state_history),
                'total_frames': max([state['frame'] for states in self.state_history.values() 
                                     for state in states], default=0)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    def calculate_motion_statistics(self, track_id: int) -> Dict:
        """
        Tính toán thống kê chuyển động
        
        Args:
            track_id: ID của tracker
            
        Returns:
            Dictionary chứa motion statistics
        """
        if track_id not in self.state_history:
            return {}
        
        states = self.state_history[track_id]
        
        if len(states) < 2:
            return {}
        
        velocities = [np.sqrt(s['velocity']['vx']**2 + s['velocity']['vy']**2) 
                     for s in states]
        
        positions_x = [s['position']['x'] for s in states]
        positions_y = [s['position']['y'] for s in states]
        
        return {
            'average_velocity': float(np.mean(velocities)),
            'max_velocity': float(np.max(velocities)),
            'min_velocity': float(np.min(velocities)),
            'velocity_std': float(np.std(velocities)),
            
            'position_range_x': (float(np.min(positions_x)), float(np.max(positions_x))),
            'position_range_y': (float(np.min(positions_y)), float(np.max(positions_y))),
            
            'total_distance': float(self._calculate_total_distance(positions_x, positions_y)),
            'tracking_duration_frames': len(states)
        }
    
    @staticmethod
    def _calculate_total_distance(x_positions: List[float], 
                                 y_positions: List[float]) -> float:
        """Tính tổng khoảng cách đi được"""
        total_dist = 0
        for i in range(1, len(x_positions)):
            dx = x_positions[i] - x_positions[i-1]
            dy = y_positions[i] - y_positions[i-1]
            total_dist += np.sqrt(dx**2 + dy**2)
        return total_dist
