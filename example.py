import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.3)

# Video 캡처 설정
cap = cv2.VideoCapture("origin_data/detect_5_squart_front.mp4")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'avc1'))

# 원본 영상의 크기와 FPS 확인
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Matplotlib 설정
fig = plt.figure(figsize=(frame_width / 100, frame_height / 100))  # 크기를 원본 영상과 동일하게 설정
ax = fig.add_subplot(111, projection='3d')

# FFmpegWriter 설정
writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
output_file = "pose_3d_animation_test1.mp4"

def plot_3d_landmarks(landmarks):
    ax.cla()  # Clear previous frame
    x_vals, y_vals, z_vals = [], [], []

    # 랜드마크 좌표 리스트 추출
    for landmark in landmarks:
        x_vals.append(landmark.x)  # X 좌표
        y_vals.append(landmark.y)  # Y 좌표
        z_vals.append(landmark.z)  # Z 좌표

    # 스켈레톤 랜드마크 연결 (MediaPipe의 Pose 모델 연결과 유사)
    connections = mp_pose.POSE_CONNECTIONS
    for start_idx, end_idx in connections:
        ax.plot(
            [x_vals[start_idx], x_vals[end_idx]],
            [z_vals[start_idx], z_vals[end_idx]],  # z와 y 축 변환
            [-y_vals[start_idx], -y_vals[end_idx]], color='b'
        )

    # 3D 랜드마크 플로팅
    ax.scatter(x_vals, z_vals, -np.array(y_vals), c='r', marker='o')

    # 축 범위 고정
    ax.set_xlim([0, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 0])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    # 초기 시점 설정 (elev: 높이, azim: 방향)
    ax.view_init(elev=2, azim=-90)

# mp4 저장을 위해 writer 사용
with writer.saving(fig, output_file, dpi=100):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 포즈 추론
        results = pose.process(frame_rgb)

        # 포즈 랜드마크가 인식되면 3D 플롯
        if results.pose_landmarks:
            plot_3d_landmarks(results.pose_landmarks.landmark)
            writer.grab_frame()  # 현재 프레임을 mp4에 저장

        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
pose.close()
plt.close(fig)
