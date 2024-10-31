import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.3)

# Video 캡처 설정

video = 'data/detect_5_squart.mp4'
cap = cv2.VideoCapture(video)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','P','4','S'))

# Matplotlib 설정
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


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
            [-y_vals[start_idx], -y_vals[end_idx]], color='black'
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
    # ax.view_init(elev=2, azim=-90)

    plt.draw()
    plt.pause(0.001)


# 일시 정지 변수 초기화
paused = False

# 프레임 반복
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 RGB로 변환
        frame = cv2.resize(frame, (400, 732), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 포즈 추론
        results = pose.process(frame_rgb)

        # 포즈 랜드마크가 인식되면 3D 플롯
        if results.pose_landmarks:
            plot_3d_landmarks(results.pose_landmarks.landmark)

        # OpenCV로 프레임 보여주기 (원본 영상)
        cv2.imshow('Squat Video', frame)

    # 키 입력 확인
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 'q' 키로 종료
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
pose.close()
plt.ioff()
plt.show()
