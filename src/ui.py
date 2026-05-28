from pyexpat import model
import gradio as gr
import cv2
from ultralytics import YOLO
from src.processor import process_video, process_video_realtime, MODEL_OPTIONS
import torch
import time

if torch.cuda.is_available():
    DEVICE = 0
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

def detect_objects_frame_1(video_path, model_name):
    if video_path is None: return [], [], []
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret: return [], [], []

    model_path = MODEL_OPTIONS.get(model_name, "yolov8n.pt")
    try: model = YOLO(model_path)
    except: model = YOLO("yolov8n.pt")
    results = model(frame, verbose=False, iou=0.45, conf=0.1, device=DEVICE)[0]
    gallery_images = []
    detected_boxes = []
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if results.boxes:
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            crop = frame_rgb[y1:y2, x1:x2]
            gallery_images.append((crop, f"Object {i}"))
            detected_boxes.append([x1, y1, x2, y2])

    return gallery_images, detected_boxes, []

def on_select_object(evt: gr.SelectData, detected_boxes, current_selection_indices):
    if current_selection_indices is None: current_selection_indices = []
    index = evt.index
    if index in current_selection_indices:
        current_selection_indices.remove(index)
    else:
        current_selection_indices.append(index)
    current_selection_indices.sort()
    selected_boxes = [detected_boxes[i] for i in current_selection_indices if i < len(detected_boxes)]
    feedback_str = f"Đang chọn các Object: {current_selection_indices}" if current_selection_indices else "Chưa chọn đối tượng nào (Sẽ track tất cả)"
    return feedback_str, current_selection_indices, selected_boxes

def clear_selection():
    return "Đã xóa chọn. Tracking tất cả.", [], []

def start_realtime_tracking(video_path, model_dd, conf_slide, iou_slide, selected_boxes_state, altitude, gimbal_pitch, focal_length, bb_style, prediction_horizon):
    if video_path is None:
        gr.Warning("Vui lòng upload video trước.")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        gr.Warning("Lỗi: Không thể mở file video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1:
        fps = 30
    
    frame_generator = process_video_realtime(
        cap=cap,
        model_selection=model_dd,
        conf_threshold=conf_slide,
        iou_threshold=iou_slide,
        target_boxes_list=selected_boxes_state,
        drone_altitude=altitude,
        gimbal_pitch=gimbal_pitch,
        focal_length=focal_length,
        bb_style=bb_style,
        prediction_horizon_seconds=prediction_horizon
    )

    for frame in frame_generator:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb


def create_ui():
    with gr.Blocks(title="UAV Tracking System") as demo:
        gr.Markdown("# UAV Multi-Target Tracking System")
        gr.Markdown("Tải video lên, quét và chọn đối tượng, sau đó bắt đầu xử lý real-time hoặc batch.")

        detected_boxes_state = gr.State([])
        selected_indices_state = gr.State([])
        selected_boxes_state = gr.State([])

        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                gr.Markdown("### Cài đặt và Xử lý")
                input_video = gr.Video(label="1. Tải lên Video")
                model_dd = gr.Dropdown(choices=list(MODEL_OPTIONS.keys()), value="EfficientNetB0", label="2. Chọn Model")
                
                with gr.Row():
                    conf_slide = gr.Slider(0.1, 0.9, 0.5, label="Confidence Threshold")
                    iou_slide = gr.Slider(0.1, 0.9, 0.2, label="IoU Threshold")

                gr.Markdown("### Tham số Camera/UAV")
                with gr.Row():
                    altitude = gr.Number(value=120, label="Altitude (m)", minimum=0)
                    gimbal_pitch = gr.Number(value=35, label="Gimbal Pitch (°)", minimum=-90, maximum=90)
                    focal_length = gr.Number(value=1470, label="Focal Length (mm)", minimum=0)

                bb_style = gr.Dropdown(
                    choices=["simple", "corners", "circle", "rounded", "tech_callout"],
                    value="tech_callout",
                    label="Bounding Box Style"
                )

                gr.Markdown("### Tham số Dự báo Quỹ đạo")
                with gr.Row():
                    prediction_horizon = gr.Slider(
                        minimum=0.5, maximum=10.0, value=3.0, step=0.5,
                        label="Prediction Horizon (s) - L = v × t_horizon"
                    )

                input_gt = gr.File(label="Tải lên Ground Truth (.txt, tùy chọn cho batch)")

                gr.Markdown("### 3. Quét và Chọn đối tượng")
                with gr.Row():
                    btn_scan = gr.Button("Quét đối tượng", variant="secondary")
                    btn_clear = gr.Button("Xóa lựa chọn", variant="stop")
                
                gallery = gr.Gallery(
                    label="Các đối tượng được phát hiện (click để chọn/bỏ chọn)", show_label=True, elem_id="gallery",
                    columns=4, rows=2, height="auto", object_fit="contain", allow_preview=False
                )
                selection_info = gr.Textbox(label="Trạng thái lựa chọn", value="Chưa chọn đối tượng (sẽ theo dõi tất cả)", interactive=False)

                gr.Markdown("### 4. Bắt đầu Xử lý")
                with gr.Row():
                    btn_run_realtime = gr.Button("Bắt đầu Real-time", variant="primary")
                    btn_run_batch = gr.Button("Bắt đầu Batch", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### Kết quả")
                with gr.Tabs():
                    with gr.TabItem("Xử lý Real-time"):
                        output_image = gr.Image(label="Kết quả Tracking Real-time", type="numpy", height=600)
                    with gr.TabItem("Xử lý Batch"):
                        output_video_batch = gr.Video(label="Kết quả Video sau khi xử lý")
                        output_metrics = gr.Textbox(label="Báo cáo Metrics", lines=10)

        # Actions
        btn_scan.click(detect_objects_frame_1, inputs=[input_video, model_dd], outputs=[gallery, detected_boxes_state, selected_indices_state])
        gallery.select(on_select_object, inputs=[detected_boxes_state, selected_indices_state], outputs=[selection_info, selected_indices_state, selected_boxes_state])
        btn_clear.click(clear_selection, outputs=[selection_info, selected_indices_state, selected_boxes_state])
        
        btn_run_realtime.click(start_realtime_tracking, inputs=[input_video, model_dd, conf_slide, iou_slide, selected_boxes_state, altitude, gimbal_pitch, focal_length, bb_style, prediction_horizon], outputs=[output_image])
        btn_run_batch.click(process_video, inputs=[input_video, input_gt, model_dd, conf_slide, iou_slide, selected_boxes_state, altitude, gimbal_pitch, focal_length, bb_style, prediction_horizon], outputs=[output_video_batch, output_metrics])

    return demo