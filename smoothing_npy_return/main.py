import os
import cv2
import numpy as np
import torch
import mediapipe as mp
from smoothing_npy_return.SmoothNet.lib.models.smoothnet import SmoothNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VideoPoseSmoother:
    def __init__(self, video_path, checkpoint_path, window_size=32):
        self.video_path = video_path
        self.pose_data_path = f"{os.path.splitext(video_path)[0]}_pose_data.npy"
        self.smoothed_data_path = f"{os.path.splitext(video_path)[0]}_smoothed_pose_data.npy"

        self.window_size = window_size
        self.model = self.load_model(checkpoint_path)

        # MediaPipe Pose 설정
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

        # COCO 포맷 인덱스 (17개 키포인트)
        self.mediapipe_to_coco_indices = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def load_model(self, checkpoint_path):
        output_size = 32
        hidden_size = 512
        res_hidden_size = 128
        num_blocks = 5
        dropout = 0.5

        model = SmoothNet(window_size=self.window_size, output_size=output_size, hidden_size=hidden_size,
                          res_hidden_size=res_hidden_size, num_blocks=num_blocks, dropout=dropout)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    def extract_coco_format(self, results):
        landmarks = results.pose_landmarks.landmark
        xy_coords = [(landmarks[idx].x, landmarks[idx].y) for idx in self.mediapipe_to_coco_indices]
        return xy_coords

    def process_video(self):
        # 비디오에서 포즈 데이터 추출
        cap = cv2.VideoCapture(self.video_path)
        pose_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                coco_format_xy = self.extract_coco_format(results)
                pose_data.append(coco_format_xy)

        cap.release()

        # 포즈 데이터 저장
        pose_data = np.array(pose_data)  # (T, 17, 2)
        np.save(self.pose_data_path, pose_data)
        print(f"원본 포즈 데이터가 {self.pose_data_path}에 저장되었습니다.")

        # SmoothNet 스무딩 적용
        self.apply_smoothing(pose_data)

    def apply_smoothing(self, pose_data):
        if pose_data.shape[-1] == 2:
            input_data = pose_data.reshape(pose_data.shape[0], -1)  # (T, 34)

        smoothed_data = []
        for start in range(0, input_data.shape[0] - self.window_size + 1, self.window_size):
            end = start + self.window_size
            window_data = input_data[start:end]
            input_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)

            with torch.no_grad():
                smoothed_output = self.model(input_tensor)

            smoothed_output_np = smoothed_output.squeeze(0).permute(1, 0).numpy()
            smoothed_data.append(smoothed_output_np)

        smoothed_data = np.concatenate(smoothed_data, axis=0).reshape(-1, 17, 2)  # (T, 17, 2)
        np.save(self.smoothed_data_path, smoothed_data)
        print(f"스무딩된 포즈 데이터가 {self.smoothed_data_path}에 저장되었습니다.")


# 사용 예시
video_path = 'data/detect_5_squart.mp4'
checkpoint_path = 'hrnet_32.pth (1).tar'
processor = VideoPoseSmoother(video_path, checkpoint_path)
processor.process_video()
