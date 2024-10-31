import cv2
import mediapipe as mp
import numpy as np
import json
import Pose_check
import counting_f
import util

class PoseEstimator:
    def __init__(self, front_video_path, side_video_path, output_front_file, output_side_file):
        self.front_video_path = front_video_path
        self.side_video_path = side_video_path
        self.output_front_file = output_front_file
        self.output_side_file = output_side_file
        self.health_warning = {'frames': []}
        self.camera_matrix1, self.dist_coeffs1, self.camera_matrix2, self.dist_coeffs2 = util.setup_camera()

    def setup_videos(self):
        self.img1 = cv2.VideoCapture(self.front_video_path)
        self.img2 = cv2.VideoCapture(self.side_video_path)
        self.frame_width = int(self.img1.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.img1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.fps1 = self.img1.get(cv2.CAP_PROP_FPS)
        self.fps2 = self.img2.get(cv2.CAP_PROP_FPS)
        self.front_video_writer = cv2.VideoWriter(self.output_front_file, self.fourcc, self.fps1, (self.frame_width, self.frame_height))
        self.side_video_writer = cv2.VideoWriter(self.output_side_file, self.fourcc, self.fps2, (self.frame_width, self.frame_height))

    # 저장되고 반환 json 반환
    def process_video(self):
        self.pose_model = util.setup_pose_model()
        body_parts = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                      'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                      'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot', 'right_foot']

        connections = [
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
            ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
            ('left_ankle', 'left_heel'), ('left_ankle', 'left_foot'), ('left_foot', 'left_heel'), ('right_ankle', 'right_heel'),
            ('right_ankle', 'right_foot'), ('right_foot', 'right_heel'),
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
        y = []
        start_time = None
        img1_landmarks = None
        img2_landmarks = None
        side_coords2 = None
        front_coords1 = None

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
            frame1_undistorted = util.undistort_image(frame1, self.camera_matrix1, self.dist_coeffs1)
            frame2_undistorted = util.undistort_image(frame2, self.camera_matrix2, self.dist_coeffs2)
            # 포즈 감지
            pose_results1 = self.pose_model.process(cv2.cvtColor(frame1_undistorted, cv2.COLOR_BGR2RGB))
            pose_results2 = self.pose_model.process(cv2.cvtColor(frame2_undistorted, cv2.COLOR_BGR2RGB))

            if pose_results1.pose_landmarks and pose_results2.pose_landmarks:
                # front_side 관절좌표
                front_coords1 = util.extract_camera_coords(pose_results1.pose_landmarks.landmark, frame1_undistorted)
                side_coords2 = util.extract_camera_coords(pose_results2.pose_landmarks.landmark, frame2_undistorted)
                # 운동 카운팅
                y.append((front_coords1['nose'][1]))
                # 부상위험요인
                stance_list.append(Pose_check.check_stance(front_coords1))
                knee_position_list.append(bool(Pose_check.check_knee_position(side_coords2, side='right')))
                knee_angle_list.append(Pose_check.calculate_knee_angle(side_coords2, side='right'))
                # 이미지 그리기 추후 3D 있으면 여기서 삼각측량
                img1_landmarks = util.draw_2d_landmarks(frame1_undistorted, front_coords1, connections, body_parts)
                img2_landmarks = util.draw_2d_landmarks(frame2_undistorted, side_coords2, connections_right, body_parts)

            if img1_landmarks is not None and img2_landmarks is not None:
                img1_landmarks = util.draw_2d_landmarks(frame1_undistorted, front_coords1, connections, body_parts)
                img2_landmarks = util.draw_2d_landmarks(frame2_undistorted, side_coords2, connections_right, body_parts)
            # 저장 및 imshow
            # cv2.imshow('front_cam',img1_landmarks)
            # cv2.imshow('side_cam',img2_landmarks)
            self.front_video_writer.write(img1_landmarks)
            self.side_video_writer.write(img2_landmarks)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 

        # 후처리 동영상 다 돌고 부상판단
        count_list, squart_count = counting_f.squart_count(y)
        util.count_injury(stance_list, knee_position_list, knee_angle_list, count_list, self.health_warning)

        with open("data/test.json", 'w') as f:
            json.dump(self.health_warning, f, indent=5)

        self.front_video_writer.release()
        self.side_video_writer.release()
        self.img1.release()
        self.img2.release()
        cv2.destroyAllWindows()

        return self.health_warning


# if __name__ == "__main__":
#     front_video = 'data/detect_5_squart_front.mp4'
#     side_video = 'data/detect_5_squart.mp4'
#     output_front_file = 'data/detect_5_squart_front_class.mp4'
#     output_side_file = 'data/detect_5_squart_class.mp4'

#     # 클래스 초기화로 파일위치, 저장위치 매개변수로 받음
#     estimator = PoseEstimator(front_video, side_video, output_front_file, output_side_file)
#     estimator.setup_videos()
#     json_data = estimator.process_video()
