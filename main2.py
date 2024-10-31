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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MultiClassTransformer(nn.Module):
    def __init__(self, num_points=17, d_model=64, num_heads=8, num_layers=3, num_classes=3):
        super(MultiClassTransformer, self).__init__()
        
        self.embedding = nn.Linear(2, d_model)  # 각 (x, y) 포인트를 d_model 임베딩으로 변환
        self.positional_encoding = self._generate_positional_encoding(num_points, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(num_points * d_model, num_classes)  # 최종 출력 레이어
        
    def _generate_positional_encoding(self, num_points, d_model):
        position = torch.arange(0, num_points, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pos_encoding = torch.zeros(num_points, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)
    
    def forward(self, x):
        # x shape: (batch_size, num_points, 2)
        x = self.embedding(x)  # 임베딩
        x = x + self.positional_encoding  # 위치 인코딩 추가
        x = x.permute(1, 0, 2)  # Transformer 인코더에 맞게 차원 변경
        
        encoded = self.transformer_encoder(x)  # Transformer 인코더에 입력
        encoded = encoded.permute(1, 0, 2).reshape(x.shape[1], -1)  # 평탄화
        
        out = self.fc_out(encoded)  # 분류 레이어
        return out

class PoseEstimator:
    def __init__(self, front_video_path, side_video_path, output_front_file, output_side_file):
        self.front_video_path = front_video_path
        self.side_video_path = side_video_path
        self.output_front_file = output_front_file
        self.output_side_file = output_side_file
        self.health_warning = {'frames': []}
        self.camera_matrix1, self.dist_coeffs1, self.camera_matrix2, self.dist_coeffs2 = util.setup_camera()
        self.checkpoint_path = 'smoothing_npy_return/hrnet_32.pth (1).tar'
        self.checkpoint_path_3D = 'smoothing_npy_return/3D_smooth.tar'
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
        self.smooth_model_3d = util.setup_smooth_model(self.checkpoint_path_3D)
        self.front_video_writer = cv2.VideoWriter(self.output_front_file, self.fourcc, self.fps1, (self.frame_width, self.frame_height))
        self.side_video_writer = cv2.VideoWriter(self.output_side_file, self.fourcc, self.fps2, (self.frame_width, self.frame_height))
        self.P1,self.P2 = util.P1P2(self.camera_matrix1,self.camera_matrix2)

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
        coords_3d_data=[]
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
                ################################################3D
                front_smoothed_data_abs = util.abs_xy(front_coords,(self.frame_width, self.frame_height))
                side_smoothed_data_abs = util.abs_xy(side_coords,(self.frame_width, self.frame_height))
                coords_3d_data.append(util.scale_3d_coords(util.triangulate_3d_points(front_smoothed_data_abs,side_smoothed_data_abs,self.P1,self.P2)))


        front_smoothed_data = util.apply_smoothing(front_pose_data, self.smooth_model, False,'front_smoothed_pose_data.npy')
        side_smoothed_data = util.apply_smoothing(side_pose_data, self.smooth_model, False,'side_smoothed_pose_data.npy')
        smooth_coords_3d = util.apply_3Dsmoothing(coords_3d_data,self.smooth_model_3d, False,'smoothed_3D_pose_data.npy')
        #######임계값 기반 체크 ###########################################근데 여기에 모델추가해서 모델이 부상이라하면 1차 필터링
        stance_list, knee_position_list, knee_angle_list = [], [], []

        # 이미지와 스무딩된 좌표 시각화 및 저장 ################문제점
        ################################################위에 하나라도 인간 디텍트 안되면 아마 프레임이 안맞을꺼임
        self.img1 = cv2.VideoCapture(self.front_video_path)
        self.img2 = cv2.VideoCapture(self.side_video_path)
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
            # #여기에 학습 모델 들어가야 할듯 ..
            # model = MultiClassTransformer(num_points=17, d_model=64, num_heads=8, num_layers=3, num_classes=3)

            # # 저장된 모델 가중치 로드
            # checkpoint_path_class = 'transformer_lr0199.pth'
            # model.load_state_dict(torch.load(checkpoint_path_class, map_location=torch.device('cpu')))  # CPU로 로드 가능

            # # 모델을 평가 모드로 설정 (학습이 끝난 후라면 필요)
            # model.eval()

            # # 예시 데이터로 모델 추론
            # # 예시 입력 데이터 생성 (batch_size, num_points, 2) 형태로, 이 경우 (1, 17, 2)
            # example_input = torch.randn(1, 17, 2)

            # # 추론 수행
            # output = model(example_input)
            # print(output)  # 모델의 출력 확인

            ###########################################
            ###########################################

            stance_list.append(Pose_check.check_stance(front_coords_dict))
            knee_position_list.append(Pose_check.check_knee_position(side_coords_dict, side='right'))
            knee_angle_list.append(Pose_check.calculate_knee_angle(side_coords_dict, side='right'))

            frame1_smoothed = frame1.copy()
            frame2_smoothed = frame2.copy()
            # frame1_smoothed = util.draw_2d_landmarks(frame1_smoothed, front_coords_dict, connections, body_parts,
            #                                          (self.frame_height, self.frame_width))
            # frame2_smoothed = util.draw_2d_landmarks(frame2_smoothed, side_coords_dict, connections_right, body_parts,
            #                                          (self.frame_height, self.frame_width))

            ### 한솔 Test draw
            front_smoothed_data_abs = util.abs_xy(front_smoothed_data[frame_idx],(self.frame_width, self.frame_height))
            side_smoothed_data_abs = util.abs_xy(side_smoothed_data[frame_idx],(self.frame_width, self.frame_height))
            frame1_smoothed = util.draw_pose(front_smoothed_data[frame_idx],frame1_smoothed,(self.frame_width, self.frame_height))
            frame2_smoothed = util.draw_pose(side_smoothed_data[frame_idx],frame2_smoothed,(self.frame_width, self.frame_height))
            # print(coords_3d)
            # print("3D좌표")
            # projection_coords_3d = coords_3d
            # for i in range(len(projection_coords_3d)):
                
            #     projection_coords_3d[]
#
            smooth_coords_3d=util.scale_3d_coords(util.triangulate_3d_points(front_smoothed_data_abs,side_smoothed_data_abs,self.P1,self.P2))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.ion()

            util.draw_3d_landmarks(ax,smooth_coords_3d)
            plt.show()

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

        self.front_video_writer.release()
        self.side_video_writer.release()
        self.img1.release()
        self.img2.release()
        cv2.destroyAllWindows()
        return self.health_warning






if __name__ == "__main__":
    front_video = 'data/detect_5_squart_front.mp4'
    side_video = 'data/detect_5_squart.mp4'
    output_front_file = 'data/smooth_detect_5_squart_front_class.mp4'
    output_side_file = 'data/smooth_detect_5_squart_class.mp4'

    # 클래스 초기화로 파일위치, 저장위치 매개변수로 받음
    estimator = PoseEstimator(front_video, side_video, output_front_file, output_side_file)
    estimator.setup_videos()
    json_data = estimator.process_video()
