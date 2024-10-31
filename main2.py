import cv2
import numpy as np
import json
import util
import Pose_check
import counting_f
import torch
from torch import nn
import os
from smoothing_npy_return.SmoothNet.lib.models.smoothnet import  SmoothNet
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
##########################
import matplotlib.pyplot as plt
import mediapipe as mp
mp_pose = mp.solutions.pose
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
##################
class PoseEstimator:
    def __init__(self, front_video_path, side_video_path, output_front_file, output_side_file):
        self.front_video_path = front_video_path
        self.side_video_path = side_video_path
        self.output_front_file = output_front_file
        self.output_side_file = output_side_file
        self.health_warning = {'frames': []}
        self.camera_matrix1, self.dist_coeffs1, self.camera_matrix2, self.dist_coeffs2 = util.setup_camera()
        self.checkpoint_path = 'smoothing_npy_return/hrnet_32.pth (1).tar'
        self.checkpoint_path = 'smoothing_npy_return/3D_smooth.tar'
        self.mediapipe_to_coco_indices = [
            0,  # nose
            2,  # left_eye (대체할 수 있는 MediaPipe 좌표)
            5,  # right_eye
            7,  # left_ear
            8,  # right_ear
            11,  # left_shoulder
            12,  # right_shoulder
            13,  # left_elbow
            14,  # right_elbow
            15,  # left_wrist
            16,  # right_wrist
            23,  # left_hip
            24,  # right_hip
            25,  # left_knee
            26,  # right_knee
            27,  # left_ankle
            28  # right_ankle
        ]
    def setup_videos(self):
        self.img1 = cv2.VideoCapture(self.front_video_path)
        self.img2 = cv2.VideoCapture(self.side_video_path)
        self.frame_width = int(self.img1.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.img1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps1 = self.img1.get(cv2.CAP_PROP_FPS)
        self.fps2 = self.img2.get(cv2.CAP_PROP_FPS)
        self.pose_model = util.setup_pose_model()
        self.smooth_model = util.setup_smooth_model(self.checkpoint_path)
        self.pose_3d_model = util.setup_3dpose_model()
        self.front_video_writer = cv2.VideoWriter(self.output_front_file, self.fourcc, self.fps1, (self.frame_width, self.frame_height))
        self.side_video_writer = cv2.VideoWriter(self.output_side_file, self.fourcc, self.fps2, (self.frame_width, self.frame_height))


    # 저장되고 반환 json 반환
    def process_video(self):

        body_parts = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                      'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                      'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot', 'right_foot']

        connections = [
            ("nose", "left_eye"),
            ("nose", "right_eye"),
            ("left_eye", "left_ear"),
            ("right_eye", "right_ear"),
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("right_shoulder", "right_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("right_hip", "right_knee"),
            ("left_knee", "left_ankle"),
            ("right_knee", "right_ankle")
        ]

        connections_right = [
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
            ('right_ankle', 'right_heel'), ('right_ankle', 'right_foot'), ('right_foot', 'right_heel'),
        ]

        frame_idx = 0
        stance_list = []
        knee_angle_list = []
        knee_position_list = []
        front_pose_data =[]
        side_pose_data=[]
        y = []
        start_time = None
        img1_landmarks = None
        img2_landmarks = None
        side_coords2 = None
        front_coords1 = None
        front_smoothing_coords = []
        side_smoothing_coords = []


        while True:
            ret1, frame1 = self.img1.read()
            ret2, frame2 = self.img2.read()

            if not ret1 or not ret2:
                print(f"Error: front {ret1}, side {ret2}")
                break

            frame_idx += 1
            if start_time is None:
                start_time = self.img1.get(cv2.CAP_PROP_POS_MSEC) / 1000
                img1_landmarks = frame1
                img2_landmarks = frame2
            current_time = (self.img1.get(cv2.CAP_PROP_POS_MSEC) / 1000) - start_time

            # 이미지 보정
            frame1_undistorted = util.undistort_image(frame1, self.camera_matrix1, self.dist_coeffs1)
            frame2_undistorted = util.undistort_image(frame2, self.camera_matrix2, self.dist_coeffs2)

            ####################### 포즈 감지
            pose_results1 = self.pose_model.process(cv2.cvtColor(frame1_undistorted, cv2.COLOR_BGR2RGB))
            pose_results2 = self.pose_model.process(cv2.cvtColor(frame2_undistorted, cv2.COLOR_BGR2RGB))

            if pose_results1.pose_landmarks and pose_results2.pose_landmarks:
                # front_side 관절좌표
                #front_coords1 = util.extract_camera_coords(pose_results1.pose_landmarks.landmark, frame1_undistorted)
                #side_coords2 = util.extract_camera_coords(pose_results2.pose_landmarks.landmark, frame2_undistorted)
                ################### smoothnet에 맞춰서 데이터 뽑기
                front_coords = util.extract_coco_format(pose_results1, self.mediapipe_to_coco_indices)
                side_coords = util.extract_coco_format(pose_results2, self.mediapipe_to_coco_indices)
                front_pose_data.append(front_coords)
                side_pose_data.append(side_coords)
                y.append(front_coords[0][1])

            else:
                # 사람이 감지되지 않은 경우 빈 좌표 추가
                if front_pose_data:
                    # 이전 프레임의 좌표 복사
                    front_pose_data.append(front_pose_data[-1])
                    side_pose_data.append(side_pose_data[-1])
                    y.append(0)
                else:
                    # 초기 상태일 경우 빈 좌표 추가
                    front_pose_data.append([(0, 0)] * len(self.mediapipe_to_coco_indices))
                    side_pose_data.append([(0, 0)] * len(self.mediapipe_to_coco_indices))
                    y.append(0)

        #################후처리 시작
        ##### 스무딩적용

        front_smoothed_data = util.apply_smoothing(front_pose_data, self.smooth_model, False,'front_smoothed_pose_data.npy')
        side_smoothed_data = util.apply_smoothing(side_pose_data, self.smooth_model, False,'side_smoothed_pose_data.npy')

        #######임계값 기반 체크 ###########################################근데 여기에 모델추가해서 모델이 부상이라하면 1차 필터링
        stance_list, knee_position_list, knee_angle_list = [], [], []

        # 이미지와 스무딩된 좌표 시각화 및 저장 ################문제점
        ################################################위에 하나라도 인간 디텍트 안되면 아마 프레임이 안맞을꺼임
        self.img1 = cv2.VideoCapture(self.front_video_path)
        self.img2 = cv2.VideoCapture(self.side_video_path)
        # 반복문 front_smoothed_data로해야하나>
        for frame_idx in range(len(front_smoothed_data)):
            ret1, frame1 = self.img1.read()
            ret2, frame2 = self.img2.read()

            if not ret1 or not ret2:
                break

            front_coords_dict = util.convert_smoothed_to_dict(front_smoothed_data[frame_idx],
                                                              (self.frame_width, self.frame_height))
            side_coords_dict = util.convert_smoothed_to_dict(side_smoothed_data[frame_idx],
                                                             (self.frame_width, self.frame_height))
            #########################################3
            #########################################
            #여기에 학습 모델 들어가야 할듯 ..
            ###########################################
            ###########################################

            stance_list.append(Pose_check.check_stance(front_coords_dict))
            knee_position_list.append(Pose_check.check_knee_position(side_coords_dict, side='right'))
            knee_angle_list.append(Pose_check.calculate_knee_angle(side_coords_dict, side='right'))


            result_3d = self.pose_3d_model.process(frame1)

            if result_3d.pose_landmarks:
                plot_3d_landmarks(result_3d.pose_landmarks.landmark)

            frame1_smoothed = frame1.copy()
            frame2_smoothed = frame2.copy()
            # frame1_smoothed = util.draw_2d_landmarks(frame1_smoothed, front_coords_dict, connections, body_parts,
            #                                          (self.frame_height, self.frame_width))
            # frame2_smoothed = util.draw_2d_landmarks(frame2_smoothed, side_coords_dict, connections_right, body_parts,
            #                                          (self.frame_height, self.frame_width))

            ### 한솔 Test draw
            frame1_smoothed = util.draw_pose(front_smoothed_data[frame_idx],frame1_smoothed,(self.frame_width, self.frame_height))
            frame2_smoothed = util.draw_pose(side_smoothed_data[frame_idx],frame2_smoothed,(self.frame_width, self.frame_height))

            cv2.imshow("Front Camera - Smoothed", frame1_smoothed)
            cv2.imshow("Side Camera - Smoothed", frame2_smoothed)

            self.front_video_writer.write(frame1_smoothed)
            self.side_video_writer.write(frame2_smoothed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        count_list, squat_count = counting_f.squart_count(y)
        util.count_injury(stance_list, knee_position_list, knee_angle_list, count_list, self.health_warning)

        with open("data/smooth.json", 'w') as f:
            json.dump(self.health_warning, f, indent=5)
        print(len(front_pose_data))
        print(len(front_smoothed_data))
        print(len(y))
        self.front_video_writer.release()
        self.side_video_writer.release()
        self.img1.release()
        self.img2.release()
        plt.ioff()
        plt.show()
        cv2.destroyAllWindows()
        return self.health_warning






if __name__ == "__main__":
    front_video = 'data/wrong_squart_front.mp4'
    side_video = 'data/wrong_squart_side.mp4'
    output_front_file = 'data/delay_check.mp4'
    output_side_file = 'data/smooth_detect_5_squart_class.mp4'

    # 클래스 초기화로 파일위치, 저장위치 매개변수로 받음
    estimator = PoseEstimator(front_video, side_video, output_front_file, output_side_file)
    estimator.setup_videos()
    json_data = estimator.process_video()
