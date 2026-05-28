import cv2
import os

def rgb_to_grayscale_video(
    input_video_path,
    output_video_path
):
    # Mở video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError("Không thể mở video đầu vào")

    # Lấy thông tin video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Codec (mp4 phổ biến)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # VideoWriter cho grayscale (vẫn dùng 3 channel để tương thích codec)
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height),
        isColor=True
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển sang grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Chuyển lại thành 3 channel để ghi video
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        out.write(gray_3ch)

    cap.release()
    out.release()
    print(" Hoàn tất chuyển video sang grayscale")

# Ví dụ sử dụng
rgb_to_grayscale_video(
    input_video_path="C:\\Users\\MY LAP\\Downloads\\Globhe_SampleData_Thermal_Video.mp4",
    output_video_path="output_gray.mp4"
)
