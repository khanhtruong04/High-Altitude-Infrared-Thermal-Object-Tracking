"""Bounding box drawing utilities for object tracking visualization"""

import cv2


def draw_simple_box(img, box, track_id, distance, is_target=False):
    """Vẽ bounding box đơn giản với hình chữ nhật"""
    x1, y1, x2, y2 = map(int, box)
    color = (0, 0, 255) if is_target else (0, 255, 0)
    thickness = 1
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Vẽ điểm chân
    cv2.circle(img, (int((x1+x2)/2), y2), 4, (0, 0, 255), -1)
    
    # Hiển thị thông tin
    label = f"ID:{track_id} | DST:{distance:.1f}m"
    if is_target:
        label = f"TARGET {track_id} | DST:{distance:.1f}m"
    
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_corners_only(img, box, track_id, distance, is_target=False):
    """Vẽ chỉ các góc của bounding box"""
    x1, y1, x2, y2 = map(int, box)
    color = (0, 0, 255) if is_target else (0, 255, 0)
    thickness = 1
    line_len = min(int((x2-x1) * 0.25), int((y2-y1) * 0.25), 25)
    
    # Vẽ 4 góc
    cv2.line(img, (x1, y1), (x1 + line_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_len), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_len), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_len), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_len), color, thickness)
    
    # Hiển thị thông tin
    label = f"ID:{track_id} | DST:{distance:.1f}m"
    if is_target:
        label = f"TARGET {track_id} | DST:{distance:.1f}m"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_circle_box(img, box, track_id, distance, is_target=False):
    """Vẽ vòng tròn xung quanh đối tượng"""
    x1, y1, x2, y2 = map(int, box)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    radius = int(max(x2 - x1, y2 - y1) / 2) + 5
    color = (0, 0, 255) if is_target else (0, 255, 0)
    thickness = 1
    
    cv2.circle(img, (cx, cy), radius, color, thickness)
    
    # Vẽ dấu tại tâm
    cv2.circle(img, (cx, cy), 1, color, -1)
    
    # Hiển thị thông tin
    label = f"ID:{track_id} | DST:{distance:.1f}m"
    if is_target:
        label = f"TARGET {track_id} | DST:{distance:.1f}m"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (cx - w//2, cy - radius - 25), (cx + w//2, cy - radius - 5), color, -1)
    cv2.putText(img, label, (cx - w//2, cy - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_rounded_box(img, box, track_id, distance, is_target=False):
    """Vẽ bounding box với các góc bo tròn"""
    x1, y1, x2, y2 = map(int, box)
    color = (0, 0, 255) if is_target else (0, 255, 0)
    thickness = 1
    radius = 10
    
    # Vẽ các cạnh
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.line(img, (x2 - radius, y2), (x1 + radius, y2), color, thickness)
    cv2.line(img, (x1, y2 - radius), (x1, y1 + radius), color, thickness)
    
    # Vẽ các góc bo tròn
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 270, 0, 90, color, thickness)
    
    # Hiển thị thông tin
    label = f"ID:{track_id} | DST:{distance:.1f}m"
    if is_target:
        label = f"TARGET {track_id} | DST:{distance:.1f}m"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_tech_callout(img, box, track_id, distance, is_target=False):
    """Vẽ bounding box với style technical callout (góc + đường dẫn)"""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    
    primary_color = (0, 0, 255) if is_target else (0, 255, 0)
    
    line_len = min(int(w * 0.3), int(h * 0.3), 30)
    thickness = 1

    cv2.line(img, (x1, y1), (x1 + line_len, y1), primary_color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_len), primary_color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_len, y1), primary_color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_len), primary_color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_len, y2), primary_color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_len), primary_color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_len, y2), primary_color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_len), primary_color, thickness)

    img_h, img_w = img.shape[:2]
    
    start_pt = (x2, y1)
    elbow_pt = (x2 + 30, y1 - 30)
    end_pt = (x2 + 140, y1 - 30)
    
    text_align_left = True

    if x2 + 150 > img_w:
        start_pt = (x1, y1)
        elbow_pt = (x1 - 30, y1 - 30)
        end_pt = (x1 - 140, y1 - 30)
        text_align_left = False
    
    if y1 - 50 < 0:
        start_pt = (start_pt[0], y2)
        elbow_pt = (elbow_pt[0], y2 + 30)
        end_pt = (end_pt[0], y2 + 30)

    cv2.line(img, start_pt, elbow_pt, primary_color, 2)
    cv2.line(img, elbow_pt, end_pt, primary_color, 2)
    cv2.circle(img, elbow_pt, 3, primary_color, -1)

    label_id = f"ID: {track_id}"
    label_dist = f"DST: {distance:.1f}m"
    if is_target:
        label_id = f"TARGET-{track_id}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thick = 2
    
    (w_id, h_id), _ = cv2.getTextSize(label_id, font, font_scale, font_thick)
    (w_dst, h_dst), _ = cv2.getTextSize(label_dist, font, font_scale, 1)
    
    max_w = max(w_id, w_dst)
    
    if text_align_left:
        txt_x = elbow_pt[0] + 10
    else:
        txt_x = elbow_pt[0] - max_w - 10

    cv2.putText(img, label_id, (txt_x, elbow_pt[1] - 5), font, font_scale, primary_color, font_thick)
    cv2.putText(img, label_dist, (txt_x, elbow_pt[1] + 18), font, font_scale, (200, 200, 200), 1)


def draw_trajectory_vector(img, current_pos, future_pos, velocity_magnitude=None, prediction_horizon=3.0, scale_factor=1.0):
    """
    Vẽ vector dự báo quỹ đạo từ tâm đối tượng hiện tại đến vị trí dự báo.
    Sử dụng mũi tên để biểu thị hướng và độ lớn của vận tốc.
    
    Args:
        img: Hình ảnh input
        current_pos: Tuple (x, y) - vị trí tâm hiện tại
        future_pos: Tuple (x, y) - vị trí dự báo
        velocity_magnitude: Float - độ lớn vận tốc (tùy chọn)
        prediction_horizon: Float - khoảng thời gian dự báo (giây), default 3.0s
        scale_factor: Float - hệ số tỉ lệ vector
    """
    import numpy as np
    import math
    
    try:
        current_pos = tuple(map(int, current_pos))
        future_pos = tuple(map(int, future_pos))
        
        # Kiểm tra giá trị hợp lệ
        if np.isnan([current_pos[0], current_pos[1], future_pos[0], future_pos[1]]).any():
            return
        
        if current_pos == future_pos:  # Bỏ qua nếu không có chuyển động
            return
        
        # Vẽ đường vector từ vị trí hiện tại đến vị trí dự báo (hướng chuyển động)
        cv2.arrowedLine(img, future_pos, current_pos, (0, 255, 255), 2, tipLength=0.3)
        
        # Vẽ đường theo dõi (trajectory) nhẹ
        cv2.circle(img, current_pos, 3, (0, 255, 255), -1)
        cv2.circle(img, future_pos, 2, (255, 255, 0), -1)
        
        # Nếu có thông tin vận tốc, hiển thị độ lớn
        if velocity_magnitude is not None and velocity_magnitude > 0.5:
            # Tính toán vị trí hiển thị
            mid_x = (current_pos[0] + future_pos[0]) // 2
            mid_y = (current_pos[1] + future_pos[1]) // 2
            
            # Hiển thị độ lớn vận tốc
            vel_text = f"v:{velocity_magnitude:.2f}px/f"
            (text_w, text_h), _ = cv2.getTextSize(vel_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (mid_x - text_w//2 - 2, mid_y - 15), 
                         (mid_x + text_w//2 + 2, mid_y - 5), (0, 0, 0), -1)
            cv2.putText(img, vel_text, (mid_x - text_w//2, mid_y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    except Exception:
        # Bỏ qua nếu có lỗi rendering
        pass


def draw_motion_prediction_path(img, current_pos, future_waypoints, color=(0, 255, 255)):
    """
    Vẽ đường dự báo chuyển động qua các waypoint.
    
    Args:
        img: Hình ảnh input
        current_pos: Tuple (x, y) - vị trí hiện tại
        future_waypoints: List of tuples [(x1, y1), (x2, y2), ...] - các điểm dự báo
        color: Tuple RGB - màu sắc (mặc định: vàng lục)
    """
    if len(future_waypoints) < 1:
        return
    
    try:
        # Chuyển đổi tương dô sang int
        current_pos = tuple(map(int, current_pos))
        future_waypoints = [tuple(map(int, pt)) for pt in future_waypoints]
        
        # Vẽ đường poly từ vị trí hiện tại qua tất cả các waypoint
        all_points = [current_pos] + future_waypoints
        points_array = __import__('numpy').array(all_points, dtype=__import__('numpy').int32)
        
        # Vẽ đường liền mét
        cv2.polylines(img, [points_array], False, color, 2)
        
        # Vẽ các waypoint
        for i, pt in enumerate(future_waypoints):
            radius = 4 - i
            cv2.circle(img, pt, max(radius, 2), color, -1)
    except Exception:
        # Bỏ qua nếu có lỗi rendering
        pass


def draw_state_estimation_info(img, track_id, current_pos, velocity_x, velocity_y):
    """
    Hiển thị thông tin State Estimation:
    - Vận tốc X, Y
    - Độ lớn vận tốc
    - Góc chuyển động
    
    Args:
        img: Hình ảnh input
        track_id: ID của tracker
        current_pos: Tuple (x, y) - vị trí tâm
        velocity_x: Vận tốc theo trục X
        velocity_y: Vận tốc theo trục Y
    """
    import math
    
    try:
        current_pos = tuple(map(int, current_pos))
        
        # Tính toán thông số vận tốc
        velocity_magnitude = math.sqrt(velocity_x**2 + velocity_y**2)
        if velocity_magnitude > 0:
            angle = math.degrees(math.atan2(velocity_y, velocity_x))
        else:
            angle = 0
        
        # Vị trí hiển thị thông tin
        info_x = current_pos[0] + 15
        info_y = current_pos[1] - 25
        
        # Hiển thị thông tin State Estimation
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        
        # Vận tốc X (COMMENTED OUT - Đã tắt hiển thị)
        # vel_x_text = f"Vx:{velocity_x:.2f}"
        # cv2.putText(img, vel_x_text, (info_x, info_y), font, font_scale, (255, 100, 0), font_thickness)
        
        # Vận tốc Y
        vel_y_text = f"Vy:{velocity_y:.2f}"
        cv2.putText(img, vel_y_text, (info_x, info_y + 15), font, font_scale, (100, 255, 0), font_thickness)
        
        # Độ lớn vận tốc
        vel_mag_text = f"|V|:{velocity_magnitude:.2f}"
        cv2.putText(img, vel_mag_text, (info_x, info_y + 30), font, font_scale, (0, 255, 255), font_thickness)
        
        # Góc chuyển động
        angle_text = f"θ:{angle:.1f}°"
        cv2.putText(img, angle_text, (info_x, info_y + 45), font, font_scale, (255, 0, 255), font_thickness)
    except Exception:
        # Bỏ qua nếu có lỗi rendering
        pass
