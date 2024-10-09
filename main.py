import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import Pose_check
import counting_f
import  util


mp_pose = mp.solutions.pose
# 카메라 및 MediaPipe 설정

def main():
    camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2 = util.setup_camera()

    # # 투영 행렬
    # R = np.array([[0.88033205, -0.0700383, -0.46915894],
    #               [0.1120291, 0.99175901, 0.06s215738],
    #               [0.46093921, -0.10727859, 0.88092358]])
    # T = np.array([[10.97330599], [-0.43874374], [0.15791984]])
    #
    # P1 = np.dot(camera_matrix1, np.hstack((np.eye(3), np.zeros((3, 1)))))
    # P2 = np.dot(camera_matrix2, np.hstack((R, T)))

    # 동영상 로드해서하기 no cam
    file_path ='data/detect_5_squart_front.mp4'
    file_path2 = 'data/detect_5_squart.mp4'

    # cam 2개 사용시
    img1 =cv2.VideoCapture(file_path)
    img2 =cv2.VideoCapture(file_path2)
    frame_width = int(img1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(img1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps1 =img1.get(cv2.CAP_PROP_FPS)
    fps2 = img2.get(cv2.CAP_PROP_FPS)
    # pose model 선언#
    pose_model = util.setup_pose_model()



    ## 표시할 부분
    body_parts = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                  'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                  'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot', 'right_foot']
    body_parts_right = ['nose', 'right_shoulder',  'right_elbow',
                  'right_wrist',  'right_hip',
                  'right_knee',  'right_ankle', 'right_heel', 'right_foot']
    ## 각관절 연결 집합
    connections = [
        ("left_shoulder", "left_elbow"),("left_elbow", "left_wrist"),("right_shoulder", "right_elbow"),("right_elbow", "right_wrist"),
        ("left_shoulder", "right_shoulder"),("left_shoulder", "left_hip"),("right_shoulder", "right_hip"),("left_hip", "right_hip"),
        ("left_hip", "left_knee"),("left_knee", "left_ankle"),("right_hip", "right_knee"),("right_knee", "right_ankle"),
        ('left_ankle', 'left_heel'),('left_ankle', 'left_foot'),('left_foot', 'left_heel'),('right_ankle', 'right_heel'),
        ('right_ankle', 'right_foot'),('right_foot', 'right_heel'),
    ]
    connections_right = [
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
         ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
         ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ('right_ankle', 'right_heel'),
        ('right_ankle', 'right_foot'), ('right_foot', 'right_heel'),
    ]

    # 여기에 넣어서 한번에 각도 관절 뽑기
    check_angle =[('left_shoulder','left_elbow','left_wrist'),('right_shoulder','right_elbow','right_wrist'),
                  ('left_hip','left_knee','left_ankle'),('right_hip','right_knee','right_ankle')]
    recording = False
    front_output_file = 'front_downsampling_10fps.mp4'
    side_output_file = 'side_downsampling_10fps.mp4'
    health_warning = { 'frames':[]
    }
    front_video_writer = cv2.VideoWriter(front_output_file, fourcc, 30, (frame_width, frame_height))
    side_video_writer = cv2.VideoWriter(side_output_file, fourcc, 30, (frame_width, frame_height))

    frame_idx = 0
    stance_list=[]
    knee_angle_list=[]
    knee_position_list=[]
    waist_list=[]
    y=[]
    start_time=None
    threshold=False
    img1_landmarks=None
    img2_landmarks =None
    side_coords2=None
    front_coords1=None
    while True:
        ret1, frame1 = img1.read()
        ret2, frame2 = img2.read()

        if not ret1 or not ret2:
            print(f"Error: front{ret1} ,side{ret2}")
            break

        frame_idx +=1
        if start_time is None:
            start_time = img1.get(cv2.CAP_PROP_POS_MSEC)/1000
            img1_landmarks = frame1
            img2_landmarks = frame2

        current_time = (img1.get(cv2.CAP_PROP_POS_MSEC) / 1000) -start_time
        # 왜곡 보정된 이미지
        frame1_undistorted = util.undistort_image(frame1, camera_matrix1, dist_coeffs1)
        frame2_undistorted = util.undistort_image(frame2, camera_matrix2, dist_coeffs2)

        # 포즈 감지
        pose_results1 = pose_model.process(cv2.cvtColor(frame1_undistorted, cv2.COLOR_BGR2RGB))
        pose_results2 = pose_model.process(cv2.cvtColor(frame2_undistorted, cv2.COLOR_BGR2RGB))

        #if frame_idx % 3 == 0: # 30프레임마다
        if pose_results1.pose_landmarks and pose_results2.pose_landmarks:
                # 각 관절 좌표
                front_coords1 = util.extract_camera_coords(pose_results1.pose_landmarks.landmark, frame1_undistorted)
                side_coords2 = util.extract_camera_coords(pose_results2.pose_landmarks.landmark, frame2_undistorted)

                y.append((front_coords1['nose'][1]))
                ###################################자세 체크 부분 ################################################

                ########## 정면 카메라 feature########
                stance_list.append(Pose_check.check_stance(front_coords1))
                #neck_warning,neck_angle = Pose_check.check_neck_angle(front_coords1)

                ########## 측면 카메라 feature########
                knee_position_list.append(bool(Pose_check.check_knee_position(side_coords2, side='right')))
                knee_angle_list.append(Pose_check.calculate_knee_angle(side_coords2, side='right'))

                # 2D 좌표 찍기
                img1_landmarks = util.draw_2d_landmarks(frame1_undistorted,front_coords1,connections,body_parts)
                img2_landmarks = util.draw_2d_landmarks(frame2_undistorted,side_coords2,connections_right,body_parts_right)
        else:
            print(frame_idx)

    #else:
        if front_coords1 is not None and side_coords2 is not None:
            # 이전 2D 좌표 사용해서 그리기
            img1_landmarks = util.draw_2d_landmarks(frame1_undistorted, front_coords1, connections, body_parts)
            img2_landmarks = util.draw_2d_landmarks(frame2_undistorted, side_coords2, connections_right,
                                               body_parts_right)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('muran1',img1_landmarks)
        cv2.imshow('muran2', img2_landmarks)
        # front_video_writer.write(img1_landmarks)
        # side_video_writer.write(img2_landmarks)
    ## ########################## ###################################################### 후처리 부분
    count_list,squart_count = counting_f.squart_count(y)
    # print(f'stancelen{len(stance_list)}  knee_position_list{len(knee_position_list)}  knee_angle_list{len(knee_angle_list)}  stance_list{len(stance_list)} 실제 frame수 {frame_idx},')
    # print(f'횟수{squart_count}, 횟수 프레임 idx{count_list}')
    print(knee_position_list)
    for i in range(len(count_list)):
        print(util.get_time_in_seconds(count_list[i],fps1))
    # 스쿼트 횟수별 부상여부 감지
    util.count_injury(stance_list, knee_position_list, knee_angle_list, count_list, health_warning)

    with open("data/test.json", 'w') as f:
        json.dump(health_warning, f,indent=5)
    #print(health_warning)
    # front_video_writer.release()
    # side_video_writer.release()
    img1.release()
    img2.release()
    cv2.destroyAllWindows()


    return health_warning



if __name__ == "__main__":
    plt.ion()
    json_data =main()
