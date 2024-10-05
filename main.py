import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import json
import time

mp_pose = mp.solutions.pose
# 카메라 및 MediaPipe 설정
def setup_camera():
    camera_matrix1 = np.array([[497.39900884, 0, 323.96380382], [0, 496.67512836, 250.05988172], [0, 0, 1]])
    dist_coeffs1 = np.array([[0.02950153, 0.0615235, -0.00102225, -0.00196549, -0.15609333]])
    camera_matrix2 = np.array([[506.35817686, 0, 330.64228386], [0, 506.09559386, 238.95273757], [0, 0, 1]])
    dist_coeffs2 = np.array([0.05938374, -0.05269976, 0.00271985, -0.00217994, -0.07622173])
    return camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2



# 2D estimation 모델 선언
def setup_pose_model():
    return mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 왜곡 보정 이미지 반환
def undistort_image(frame, camera_matrix, dist_coeffs):
    return cv2.undistort(frame, camera_matrix, dist_coeffs)


# Triangulation을 통한 3D 좌표 계산
def triangulate_3d_points(camera_coords1, camera_coords2, P1, P2, body_parts):
    coords_3d = {}
    for part in body_parts:
        points_4d = cv2.triangulatePoints(P1, P2, camera_coords1[part].reshape(2, 1),
                                          camera_coords2[part].reshape(2, 1))
        coords_3d[part] = (points_4d[:3] / points_4d[3])  # homogeneous coordinates to 3D
    return coords_3d


# 스케일링 함수
def scale_value(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value) if max_value - min_value != 0 else 0


# 스케일링된 3D 좌표 반환
def scale_3d_coords(coords_3d):
    x_coords = np.array([coord[0][0] for coord in coords_3d.values()])
    y_coords = np.array([coord[1][0] for coord in coords_3d.values()])
    z_coords = np.array([coord[2][0] for coord in coords_3d.values()])

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()

    scaled_coords_3d = {}
    for part, coord in coords_3d.items():
        scaled_x = scale_value(coord[0][0], x_min, x_max)
        scaled_y = scale_value(coord[1][0], y_min, y_max)
        scaled_z = scale_value(coord[2][0], z_min, z_max)
        scaled_coords_3d[part] = np.array([[scaled_x], [scaled_y], [scaled_z]], dtype=float)

    return scaled_coords_3d


# joint 각도 추출 이미지 영상 y축이 반대이기때문에 180.0 - angle해서 반환
def calculate_angle(start, joint, end, coords):
    v1 =coords[joint]-coords[start]
    v2 =coords[end]-coords[joint]
    dot_product = np.dot(v1.flatten(), v2.flatten())
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # cos_theta가 -1에서 1 사이에 있도록 클리핑

    return 180.0- np.degrees(angle)

def calculate_2d_angle(v1, v2):
    dot_product = np.dot(v1.flatten(), v2.flatten())
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:  # 벡터의 길이가 0인 경우를 체크
        return 0  # 또는 다른 적절한 값을 반환

    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # cos_theta가 -1에서 1 사이에 있도록 클리핑
    return np.degrees(angle)

# 이건 그냥 좌표찍기
def draw_2d_landmarks(frame,coords,connections):
    for part,coord in coords.items():
        cv2.circle(frame,(int(coord[0]),int(coord[1])),10,(255,0,0),-1)

    for connection in connections:
        start, end = connection
        start_coord = coords[start]
        end_coord = coords[end]
        cv2.line(frame, (int(start_coord[0]), int(start_coord[1])), (int(end_coord[0]), int(end_coord[1])), (0, 255, 0),
                 2)

    return frame
# 3d 좌표찍기
def draw_3d_landmarks(ax,coords,connections):
    ax.clear()
    # 좌표찍기
    for part,coord in coords.items():
        ax.scatter(coord[0],coord[1],coord[2],)

    for connection in connections:
        start, end = connection
        start_coord = coords[start].flatten()
        end_coord = coords[end].flatten()
        ax.plot([start_coord[0], end_coord[0]],
                [start_coord[1], end_coord[1]],
                [start_coord[2], end_coord[2]],
                color='gray')

    plt.draw()
    plt.pause(0.01)

    # 모델이 뽑은 좌표 * 이미지 shape으로 실제 픽셀값 추출
def extract_camera_coords(landmarks, frame):
    camera_coords = {
                "nose": np.array(
                    [landmarks[mp_pose.PoseLandmark.NOSE.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.NOSE.value].y * frame.shape[0]],
                    dtype=np.float32),

                "left_shoulder": np.array(
                    [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]],
                    dtype=np.float32),

                "right_shoulder": np.array(
                    [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]],
                    dtype=np.float32),

                "left_elbow": np.array(
                    [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame.shape[0]],
                    dtype=np.float32),

                "right_elbow": np.array(
                    [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]],
                    dtype=np.float32),

                "left_wrist": np.array(
                    [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame.shape[0]],
                    dtype=np.float32),

                "right_wrist": np.array(
                    [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]],
                    dtype=np.float32),

                "left_hip": np.array(
                    [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]],
                    dtype=np.float32),

                "right_hip": np.array(
                    [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0]],
                    dtype=np.float32),

                "left_knee": np.array(
                    [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * frame.shape[0]],
                    dtype=np.float32),

                "right_knee": np.array(
                    [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * frame.shape[0]],
                    dtype=np.float32),

                "left_ankle": np.array(
                    [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame.shape[0]],
                    dtype=np.float32),

                "right_ankle": np.array(
                    [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * frame.shape[0]],
                    dtype=np.float32),

                "left_heel": np.array(
                    [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y * frame.shape[0]],
                    dtype=np.float32),

                "right_heel": np.array(
                    [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y * frame.shape[0]],
                    dtype=np.float32),

                "left_foot": np.array(
                    [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * frame.shape[0]],
                    dtype=np.float32),

                "right_foot": np.array(
                    [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x * frame.shape[1],
                     landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * frame.shape[0]],
                    dtype=np.float32),

            }
    return camera_coords
# 주 루프


# 수정이 필요함 함수 이함수안에 우리가 봐야할 check포인트 넣고 해줘야함
def record_timestamp(angle,time,stamp,threshold):
    if angle> 120 and not threshold:
        stamp.append({
            'angle':angle,
            'time':time})
        threshold =True
    elif angle <= 120 and threshold:
        threshold = False
    return stamp ,threshold

def distance_shoudler_knee(a):
    # 정면 cap에서 양 어깨 ,양 무릎 좌표받아와서 체크하기 근데 어느정도면 문제인지
    return a
def main():
    # camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2 = setup_camera()
    #
    # # 투영 행렬
    # R = np.array([[0.88033205, -0.0700383, -0.46915894],
    #               [0.1120291, 0.99175901, 0.06s215738],
    #               [0.46093921, -0.10727859, 0.88092358]])
    # T = np.array([[10.97330599], [-0.43874374], [0.15791984]])
    #
    # P1 = np.dot(camera_matrix1, np.hstack((np.eye(3), np.zeros((3, 1)))))
    # P2 = np.dot(camera_matrix2, np.hstack((R, T)))

    # 동영상 로드해서하기 no cam
    #file_path ='output_test.mp4'
    #file_path2 = 'squart.mp4'

    # cam 2개 사용시
    img1 =cv2.VideoCapture(0)
    #img2 =cv2.VideoCapture(file_path2)

    output_file = f"data/elbow_120_json_test.mp4"
    #output_file2 = f"squart.mp4"
    frame_width = int(img1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(img1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, 30,(frame_width, frame_height))

    #img2 = cv2.VideoCapture(1)
    #fig =plt.figure()
    #ax = fig.add_subplot(111,projection='3d')

    pose_model = setup_pose_model()

    ## 표시할 부분
    body_parts = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                  'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                  'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot', 'right_foot']
    ## 각관절 연결 집합
    connections = [
        ("left_shoulder", "left_elbow"),("left_elbow", "left_wrist"),("right_shoulder", "right_elbow"),("right_elbow", "right_wrist"),
        ("left_shoulder", "right_shoulder"),("left_shoulder", "left_hip"),("right_shoulder", "right_hip"),("left_hip", "right_hip"),
        ("left_hip", "left_knee"),("left_knee", "left_ankle"),("right_hip", "right_knee"),("right_knee", "right_ankle"),
        ('left_ankle', 'left_heel'),('left_ankle', 'left_foot'),('left_foot', 'left_heel'),('right_ankle', 'right_heel'),
        ('right_ankle', 'right_foot'),('right_foot', 'right_heel'),
    ]

    # 여기에 넣어서 한번에 각도 관절 뽑기
    check_angle =[('left_shoulder','left_elbow','left_wrist'),('right_shoulder','right_elbow','right_wrist'),
                  ('left_hip','left_knee','left_ankle'),('right_hip','right_knee','right_ankle')]
    recording = False
    output_file = None
    #video_writer = None
    frame_idx = 0
    angles = []
    timestamps = []
    save_count=0
    start_time=None
    threshold=False
    while True:
        ret1, frame1 = img1.read()
        #ret2, frame2 = img2.read()

        if not ret1: #or not ret2:
            print("Error: Could not read image fromqq webcams.")
            break

        if start_time is None:
            start_time = img1.get(cv2.CAP_PROP_POS_MSEC)/1000

        current_time = (img1.get(cv2.CAP_PROP_POS_MSEC) / 1000) -start_time
        # 왜곡 보정된 이미지
        frame1_undistorted= frame1

        #frame2_undistorted= frame2
        # frame1_undistorted = undistort_image(frame1, camera_matrix1, dist_coeffs1)
        # frame2_undistorted = undistort_image(frame2, camera_matrix2, dist_coeffs2)

        # 포즈 감지
        pose_results1 = pose_model.process(cv2.cvtColor(frame1_undistorted, cv2.COLOR_BGR2RGB))
        #pose_results2 = pose_model.process(cv2.cvtColor(frame2_undistorted, cv2.COLOR_BGR2RGB))

        if pose_results1.pose_landmarks: #and pose_results2.pose_landmarks:
            # 각 관절 좌표
            camera_coords1 = extract_camera_coords(pose_results1.pose_landmarks.landmark, frame1_undistorted)
            #camera_coords2 = extract_camera_coords(pose_results2.pose_landmarks.landmark, frame2_undistorted)

            # 2D 좌표 찍기
            img1_landmarks = draw_2d_landmarks(frame1,camera_coords1,connections)
            elbow_angle = calculate_angle('left_shoulder','left_elbow','left_wrist',camera_coords1)

            timestamps,threshold = record_timestamp(elbow_angle,current_time,timestamps,threshold)

            print(elbow_angle,current_time)
            # if elbow_angle> 85.0 and elbow_angle<95.0:
            #     frame_filename = f'frame_{save_count}_{int(elbow_angle)}.jpg'
            #     cv2.imwrite(frame_filename, frame1)
            #     save_count += 1
            #     print(f'Frame saved ')

            #img2_landmarks = draw_2d_landmarks(frame2, camera_coords2,connections)
            # 3d좌표추출
            #coords_3d = triangulate_3d_points(camera_coords1, camera_coords2, P1, P2, body_parts)
            #scaled_coords_3d = scale_3d_coords(coords_3d)
            # 3d 좌표찍기
            #draw_3d_landmarks(ax,scaled_coords_3d,connections)

            #print('elbow_angle :',calculate_angle('left_shoulder','left_elbow','left_wrist',camera_coords1))
        else:
            img1_landmarks =frame1
            #img2_landmarks = frame2

        video_writer.write(img1_landmarks)
        ################### s 녹화 시작 / e 녹화 종료
        # if cv2.waitKey(1) & 0xFF == ord('s'):  # 's' 키를 눌러 동영상 녹화 시작
        #     if not recording:
        #         recording = True
        #         output_file = f"elbow_120_json_test.mp4"
        #         fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        #         video_writer = cv2.VideoWriter(output_file, fourcc, 30, (frame1.shape[1], frame1.shape[0]))
        #         print("시작")
        #
        # if recording:
        #     video_writer.write(img1_landmarks)
        #
        # if cv2.waitKey(1) & 0xFF == ord('e'):  # 'e' 키를 눌러 동영상 녹화 종료
        #     if recording:
        #         recording = False
        #         video_writer.release()
        #         print("녹화종료")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('muran1',img1_landmarks)
        #cv2.imshow('muran2', img2_landmarks)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    with open("data/elbow_120_json_test.json", 'w') as f:
        json.dump(timestamps, f)
    print(timestamps)
    video_writer.release()
    img1.release()
    #img2.release()
    cv2.destroyAllWindows()





# 2D 카메라 좌표 추출 함수



if __name__ == "__main__":
    plt.ion()
    main()
