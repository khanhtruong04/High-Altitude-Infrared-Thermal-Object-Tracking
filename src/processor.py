import cv2
import numpy as np
import os
import subprocess
import motmetrics as mm
from ultralytics import YOLO
import gradio as gr

from src.utils.distance import DistanceEstimator
from src.utils.bbox_drawing import (
    draw_simple_box, draw_corners_only, draw_circle_box, draw_rounded_box, draw_tech_callout,
    draw_trajectory_vector, draw_motion_prediction_path, draw_state_estimation_info
)
from src.tracker.gmc import GMC
from src.tracker.kalman import KalmanBoxTracker
from src.tracker.association import associate_detections_to_trackers
from src.utils.gt_loader import load_mot_gt
import torch

if torch.cuda.is_available():
    DEVICE = 0
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

print(f"Đang sử dụng thiết bị: {DEVICE}")

MODEL_OPTIONS = {
    "EfficientNetB0": "models/efficientnetB0-yolov8.pt",
    "EfficientNetB3": "yolov8n.pt",
    "MobileNet": "models/mobilenetv3_yolo8_best.pt",
    "ConvNext-T": "models/convnext_T_best.pt",
    "ConvNext-S": "models/convnext_S_best.pt"
}

# All drawing functions moved to src/utils/bbox_drawing module

# Helper function for IOU calculation
def calculate_iou_single(box1, box2):
    xx1 = max(box1[0], box2[0]); yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2]); yy2 = min(box1[3], box2[3])
    w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
    inter = w * h
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter/union if union > 0 else 0

def process_video(video_path, gt_path, model_selection, conf_threshold, iou_threshold, target_boxes_list, 
                  drone_altitude=130.0, gimbal_pitch=35.0, focal_length=1470.0, bb_style="tech_callout",
                  show_trajectory=True, future_steps=5, prediction_horizon_seconds=3.0, progress=gr.Progress()):
    if video_path is None: 
        return None, "Vui lòng upload video."

    model_path = MODEL_OPTIONS.get(model_selection, "yolov8n.pt")
    
    target_track_ids = set()
    is_selective_mode = target_boxes_list is not None and len(target_boxes_list) > 0

    try:
        model = YOLO(model_path)
    except Exception:
        model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1: 
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = "temp_output.mp4"
    final_output_path = "result_video.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    gt_data = load_mot_gt(gt_path)
    has_gt = len(gt_data) > 0
    acc = mm.MOTAccumulator(auto_id=True)
    
    trackers = []
    gmc = GMC(downscale=2)
    
    dist_estimator = DistanceEstimator(focal_length=focal_length, image_height=height)
    
    KalmanBoxTracker.count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        frame_idx += 1
        
        if frame_idx % 10 == 0:
            progress(frame_idx / max(total_frames, 1), desc=f"Processing {frame_idx}/{total_frames}")

        # YOLO Detection
        results = model(frame, verbose=False, iou=0.45, conf=0.1, device=DEVICE)[0]
        dets = []
        if results.boxes:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0].cpu().numpy())
                dets.append([x1, y1, x2, y2, score])
        dets = np.array(dets) if len(dets) > 0 else np.empty((0, 5))

        # GMC & Prediction
        gmc.apply(frame, trackers)
        trks = np.zeros((len(trackers), 5))
        to_del = []
        for t, tracker in enumerate(trackers):
            pos = tracker.predict()[0]
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)): 
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del): 
            trackers.pop(t)

        # Matching
        if len(dets) > 0:
            inds_high = dets[:, 4] >= conf_threshold
            inds_low = (dets[:, 4] > 0.1) & (dets[:, 4] < conf_threshold)
            dets_high = dets[inds_high]
            dets_low = dets[inds_low]
        else:
            dets_high = np.empty((0, 5))
            dets_low = np.empty((0, 5))

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_high, trks, iou_threshold)

        # IOU Match for low score dets
        trks_remain = trks[unmatched_trks]
        dets_remain = dets_low
        if len(trks_remain) > 0 and len(dets_remain) > 0:
            matched_l, _, _ = associate_detections_to_trackers(dets_remain, trks_remain, 0.1)
            for m in matched_l:
                trackers[unmatched_trks[m[1]]].update(dets_remain[m[0]][:4], dets_remain[m[0]][4])
        
        for m in matched:
            trackers[m[1]].update(dets_high[m[0]][:4], dets_high[m[0]][4])
        
        for i in unmatched_dets:
            trackers.append(KalmanBoxTracker(dets_high[i][:4]))

        # Track management & Output collection
        i = len(trackers)
        ret_trackers = []
        for trk in reversed(trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= 3 or frame_idx <= 3):
                ret_trackers.append(np.concatenate((d,[trk.id])).reshape(1,-1))
            i -= 1
            if(trk.time_since_update > 30): 
                trackers.pop(i)

        # Selective ID Logic (Frame 1)
        if frame_idx == 1 and is_selective_mode and len(ret_trackers) > 0:
            for target_box in target_boxes_list:
                best_iou = 0
                best_id = -1
                for trk_data in ret_trackers:
                    d = trk_data[0]
                    trk_box = [d[0], d[1], d[2], d[3]]
                    iou = calculate_iou_single(target_box, trk_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_id = int(d[4])
                if best_iou > 0.5:
                    target_track_ids.add(best_id)
            if len(target_track_ids) == 0:
                is_selective_mode = False

        # Metrics update
        if has_gt:
            t_ids = []
            t_boxes = []
            for trk_data in ret_trackers:
                d = trk_data[0]
                t_ids.append(int(d[4]))
                t_boxes.append([d[0], d[1], d[2]-d[0], d[3]-d[1]])
            g_ids = []
            g_boxes = []
            if frame_idx in gt_data:
                for item in gt_data[frame_idx]:
                    g_ids.append(int(item[4]))
                    g_boxes.append([item[0], item[1], item[2]-item[0], item[3]-item[1]])
            
            dist = mm.distances.iou_matrix(g_boxes, t_boxes, max_iou=0.5) if (len(g_boxes)>0 and len(t_boxes)>0) else []
            acc.update(g_ids, t_ids, dist)

        # Draw Kalman trajectory and future prediction (State Estimation & Motion Prediction)
        if show_trajectory:
            for tracker in trackers:
                try:
                    # Get historical trace (State Estimation)
                    trace = tracker.get_trace()
                    if len(trace) >= 2:
                        # COMMENTED OUT - Đường vàng (Historical Trace)
                        # pts = np.array(trace, dtype=np.int32)
                        # cv2.polylines(frame, [pts], False, (200, 200, 0), 1)
                        pass

                    if len(trace) > 0:
                        current_center = trace[-1]  # Vị trí tâm hiện tại
                        
                        # Kiểm tra object có bị mất > 1s không (nhiều box)
                        time_since_update_seconds = tracker.time_since_update / fps
                        if time_since_update_seconds > 1.0:
                            # Object bị mất > 1s → skip drawing vector
                            continue
                        
                        # Lấy thông tin chuyển động
                        motion_info = tracker.get_motion_info(
                            prediction_horizon=prediction_horizon_seconds
                        )
                        
                        vx = motion_info['velocity']['vx']
                        vy = motion_info['velocity']['vy']
                        velocity_magnitude = motion_info['velocity']['magnitude']
                        
                        # Kiểm tra object có nằm trong frame không
                        if (0 <= current_center[0] < width and 0 <= current_center[1] < height):
                            # Tính điểm dự báo từ công thức tuyến tính: L = v × t_horizon
                            # distance = velocity(px/frame) × prediction_horizon(s) × fps(frame/s)
                            distance_x = vx * prediction_horizon_seconds * fps
                            distance_y = vy * prediction_horizon_seconds * fps
                            
                            predicted_final_pos = (
                                int(current_center[0] + distance_x),
                                int(current_center[1] + distance_y)
                            )
                            
                            # Vẽ polyline từ hiện tại → điểm dự báo cuối (theo công thức L = v × t_horizon)
                            try:
                                pts = np.array([current_center, predicted_final_pos], dtype=np.int32)
                                cv2.polylines(frame, [pts], False, (0, 255, 255), 2)
                                # Vẽ chấm tại điểm dự báo
                                cv2.circle(frame, predicted_final_pos, 4, (0, 255, 255), -1)
                                
                                # Hiển thị v:X.XXpx/f text
                                if velocity_magnitude > 0.5:
                                    mid_x = (current_center[0] + predicted_final_pos[0]) // 2
                                    mid_y = (current_center[1] + predicted_final_pos[1]) // 2
                                    
                                    vel_text = f"v:{velocity_magnitude:.2f}px/f"
                                    (text_w, text_h), _ = cv2.getTextSize(vel_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                                    cv2.rectangle(frame, (mid_x - text_w//2 - 2, mid_y - 15), 
                                                (mid_x + text_w//2 + 2, mid_y - 5), (0, 0, 0), -1)
                                    cv2.putText(frame, vel_text, (mid_x - text_w//2, mid_y - 8), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                            except Exception:
                                pass
                except Exception:
                    # Bỏ qua lỗi visualization để không làm gián đoạn tracking
                    pass

        # Draw bounding boxes and labels
        for d in ret_trackers:
            d = d[0]
            x1, y1, x2, y2, tid = int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4])
            
            # Tính khoảng cách
            g_dist, s_dist = dist_estimator.estimate([x1, y1, x2, y2], drone_altitude, gimbal_pitch)
            
            is_target = False
            should_draw = True
            
            if is_selective_mode:
                if tid in target_track_ids:
                    is_target = True
                else:
                    should_draw = False
            else:
                is_target = False
            
            if should_draw:
                if bb_style == "simple":
                    draw_simple_box(frame, [x1, y1, x2, y2], tid, g_dist, is_target)
                elif bb_style == "corners":
                    draw_corners_only(frame, [x1, y1, x2, y2], tid, g_dist, is_target)
                elif bb_style == "circle":
                    draw_circle_box(frame, [x1, y1, x2, y2], tid, g_dist, is_target)
                elif bb_style == "rounded":
                    draw_rounded_box(frame, [x1, y1, x2, y2], tid, g_dist, is_target)
                else:  # tech_callout
                    draw_tech_callout(frame, [x1, y1, x2, y2], tid, g_dist, is_target)
        
        out.write(frame)

    cap.release()
    out.release()

    # Finalize Metrics
    metrics_str = ""
    if has_gt:
        mh = mm.metrics.create()
        try:
            summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'idf1', 'mostly_tracked', 'mostly_lost', 'num_switches'], name='acc')
            metrics_str = mm.io.render_summary(summary, formatters=mh.formatters, namemap={'num_frames': 'Frames', 'mota': 'MOTA', 'motp': 'MOTP', 'idf1': 'IDF1', 'mostly_tracked': 'MT', 'mostly_lost': 'ML', 'num_switches': 'ID Sw'})
        except Exception:
            metrics_str = "Error calculating metrics"

    if os.path.exists(final_output_path): 
        os.remove(final_output_path)
    try:
        subprocess.call(args=f"ffmpeg -y -i {output_path} -c:v libx264 {final_output_path} -loglevel quiet", shell=True)
    except Exception:
        final_output_path = output_path
        
    return final_output_path, metrics_str

def process_video_realtime(cap, model_selection, conf_threshold, iou_threshold, target_boxes_list, 
                  drone_altitude=130.0, gimbal_pitch=35.0, focal_length=1470.0, bb_style="tech_callout",
                  prediction_horizon_seconds=3.0):
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1:
        fps = 30.0  # Default FPS
    
    model_path = MODEL_OPTIONS.get(model_selection, "yolov8n.pt")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        model = YOLO("yolov8n.pt")

    target_track_ids = set()
    is_selective_mode = target_boxes_list is not None and len(target_boxes_list) > 0

    trackers = []
    gmc = GMC(downscale=2)
    dist_estimator = DistanceEstimator(focal_length=focal_length, image_height=height)
    
    KalmanBoxTracker.count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # YOLO Detection
        results = model(frame, verbose=False, iou=0.45, conf=0.1, device=DEVICE)[0]
        dets = []
        if results.boxes:
             for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0].cpu().numpy())
                dets.append([x1, y1, x2, y2, score])
        dets = np.array(dets) if len(dets) > 0 else np.empty((0, 5))

        # GMC & Prediction
        gmc.apply(frame, trackers)
        trks = np.zeros((len(trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)): to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del): trackers.pop(t)

        # Matching
        if len(dets) > 0:
            inds_high = dets[:, 4] >= conf_threshold
            inds_low = (dets[:, 4] > 0.1) & (dets[:, 4] < conf_threshold)
            dets_high = dets[inds_high]
            dets_low = dets[inds_low]
        else:
            dets_high = np.empty((0, 5)); dets_low = np.empty((0, 5))

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_high, trks, iou_threshold)

        # IOU Match for low score dets
        trks_remain = trks[unmatched_trks]
        dets_remain = dets_low
        if len(trks_remain) > 0 and len(dets_remain) > 0:
            matched_l, _, _ = associate_detections_to_trackers(dets_remain, trks_remain, 0.1)
            for m in matched_l:
                trackers[unmatched_trks[m[1]]].update(dets_remain[m[0]][:4], dets_remain[m[0]][4])
        
        for m in matched:
            trackers[m[1]].update(dets_high[m[0]][:4], dets_high[m[0]][4])
        
        for i in unmatched_dets:
            trackers.append(KalmanBoxTracker(dets_high[i][:4]))

        # Track management
        i = len(trackers)
        ret_trackers = []
        for trk in reversed(trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= 3 or frame_idx <= 3):
                ret_trackers.append(np.concatenate((d,[trk.id])).reshape(1,-1))
            i -= 1
            if(trk.time_since_update > 30): trackers.pop(i)

        # Selective ID Logic (Frame 1)
        if frame_idx == 1 and is_selective_mode and len(ret_trackers) > 0:
            for target_box in target_boxes_list:
                best_iou = 0
                best_id = -1
                for trk_data in ret_trackers:
                    d = trk_data[0]
                    trk_box = [d[0], d[1], d[2], d[3]]
                    iou = calculate_iou_single(target_box, trk_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_id = int(d[4])
                if best_iou > 0.5:
                    target_track_ids.add(best_id)
            if len(target_track_ids) == 0:
                is_selective_mode = False

        # Draw trajectory and motion prediction (State Estimation & Motion Prediction)
        for tracker in trackers:
            try:
                # Get historical trace (State Estimation)
                trace = tracker.get_trace()
                if len(trace) >= 2:
                    # COMMENTED OUT - Đường vàng (Historical Trace)
                    # pts = np.array(trace, dtype=np.int32)
                    # cv2.polylines(frame, [pts], False, (200, 200, 0), 1)
                    pass

                # Get predicted trajectory (Motion Prediction) dựa trên time horizon
                if len(trace) > 0:
                    current_center = trace[-1]
                    
                    # Kiểm tra object có bị mất > 1s không (nhiều box)
                    time_since_update_seconds = tracker.time_since_update / fps
                    if time_since_update_seconds > 1.0:
                        # Object bị mất > 1s → skip drawing vector
                        continue
                    
                    # Lấy thông tin chuyển động
                    motion_info = tracker.get_motion_info(
                        prediction_horizon=prediction_horizon_seconds
                    )
                    
                    vx = motion_info['velocity']['vx']
                    vy = motion_info['velocity']['vy']
                    velocity_magnitude = motion_info['velocity']['magnitude']
                    
                    # Kiểm tra object có nằm trong frame không
                    if (0 <= current_center[0] < width and 0 <= current_center[1] < height):
                        # Tính điểm dự báo từ công thức tuyến tính: L = v × t_horizon
                        # distance = velocity(px/frame) × prediction_horizon(s) × fps(frame/s)
                        distance_x = vx * prediction_horizon_seconds * fps
                        distance_y = vy * prediction_horizon_seconds * fps
                        
                        predicted_final_pos = (
                            int(current_center[0] + distance_x),
                            int(current_center[1] + distance_y)
                        )
                        
                        # Vẽ polyline từ hiện tại → điểm dự báo cuối (theo công thức L = v × t_horizon)
                        try:
                            pts = np.array([current_center, predicted_final_pos], dtype=np.int32)
                            cv2.polylines(frame, [pts], False, (0, 255, 255), 2)
                            # Vẽ chấm tại điểm dự báo
                            cv2.circle(frame, predicted_final_pos, 4, (0, 255, 255), -1)
                            
                            # Hiển thị v:X.XXpx/f text
                            if velocity_magnitude > 0.5:
                                mid_x = (current_center[0] + predicted_final_pos[0]) // 2
                                mid_y = (current_center[1] + predicted_final_pos[1]) // 2
                                
                                vel_text = f"v:{velocity_magnitude:.2f}px/f"
                                (text_w, text_h), _ = cv2.getTextSize(vel_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                                cv2.rectangle(frame, (mid_x - text_w//2 - 2, mid_y - 15), 
                                            (mid_x + text_w//2 + 2, mid_y - 5), (0, 0, 0), -1)
                                cv2.putText(frame, vel_text, (mid_x - text_w//2, mid_y - 8), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        except Exception:
                            pass
            except Exception as e:
                # Bỏ qua lỗi visualization để không làm gián đoạn tracking
                pass

        # --- DRAWING (Updated with Tech Callout) ---
        for d in ret_trackers:
            d = d[0]
            x1, y1, x2, y2, tid = int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4])
            
            g_dist, s_dist = dist_estimator.estimate([x1, y1, x2, y2], drone_altitude, gimbal_pitch)
            
            is_target = False
            should_draw = True
            
            if is_selective_mode:
                if tid in target_track_ids:
                    is_target = True
                else:
                    should_draw = False
            else:
                is_target = False
            
            if should_draw:
                if bb_style == "simple":
                    draw_simple_box(frame, [x1, y1, x2, y2], tid, g_dist, is_target)
                elif bb_style == "corners":
                    draw_corners_only(frame, [x1, y1, x2, y2], tid, g_dist, is_target)
                elif bb_style == "circle":
                    draw_circle_box(frame, [x1, y1, x2, y2], tid, g_dist, is_target)
                elif bb_style == "rounded":
                    draw_rounded_box(frame, [x1, y1, x2, y2], tid, g_dist, is_target)
                else:  # tech_callout
                    draw_tech_callout(frame, [x1, y1, x2, y2], tid, g_dist, is_target)
        
        yield frame
        
    cap.release()