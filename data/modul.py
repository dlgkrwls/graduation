import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
# 카메라 및 MediaPipe 설정
def setup_camera():
    camera_matrix1 = np.array([[497.39900884, 0, 323.96380382], [0, 496.67512836, 250.05988172], [0, 0, 1]])
    dist_coeffs1 = np.array([[0.02950153, 0.0615235, -0.00102225, -0.00196549, -0.15609333]])
    camera_matrix2 = np.array([[506.35817686, 0, 330.64228386], [0, 506.09559386, 238.95273757], [0, 0, 1]])
    dist_coeffs2 = np.array([0.05938374, -0.05269976, 0.00271985, -0.00217994, -0.07622173])
    return camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2


def setup_pose_model():
    return mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


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


# 벡터 간 각도 계산
def calculate_angle(v1, v2):
    dot_product = np.dot(v1.flatten(), v2.flatten())
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(cos_theta)
    return np.degrees(angle)
def draw_2d_landmarks(frame,coords,connections):
    for part,coord in coords.items():
        cv2.circle(frame,(int(coord[0]),int(coord[1])),10,(0,255,0),-1)

    for connection in connections:
        start, end = connection
        start_coord = coords[start]
        end_coord = coords[end]
        cv2.line(frame, (int(start_coord[0]), int(start_coord[1])), (int(end_coord[0]), int(end_coord[1])), (0, 255, 0),
                 2)

    return frame

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
def main():
    camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2 = setup_camera()

    # 투영 행렬
    R = np.array([[0.88033205, -0.0700383, -0.46915894],
                  [0.1120291, 0.99175901, 0.06215738],
                  [0.46093921, -0.10727859, 0.88092358]])
    T = np.array([[10.97330599], [-0.43874374], [0.15791984]])

    P1 = np.dot(camera_matrix1, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(camera_matrix2, np.hstack((R, T)))

    img1 = cv2.VideoCapture(0)
    img2 = cv2.VideoCapture(1)

    fig =plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    pose_model = setup_pose_model()

    body_parts = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                  'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                  'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot', 'right_foot']
    connections = [
        ("left_shoulder", "left_elbow"),("left_elbow", "left_wrist"),("right_shoulder", "right_elbow"),("right_elbow", "right_wrist"),
        ("left_shoulder", "right_shoulder"),("left_shoulder", "left_hip"),("right_shoulder", "right_hip"),("left_hip", "right_hip"),
        ("left_hip", "left_knee"),("left_knee", "left_ankle"),("right_hip", "right_knee"),("right_knee", "right_ankle"),
        ('left_ankle', 'left_heel'),('left_ankle', 'left_foot'),('left_foot', 'left_heel'),('right_ankle', 'right_heel'),
        ('right_ankle', 'right_foot'),('right_foot', 'right_heel'),
    ]
    while True:
        ret1, frame1 = img1.read()
        ret2, frame2 = img2.read()

        if not ret1 or not ret2:
            print("Error: Could not read image from webcams.")
            break

        # 왜곡 보정된 이미지
        frame1_undistorted = undistort_image(frame1, camera_matrix1, dist_coeffs1)
        frame2_undistorted = undistort_image(frame2, camera_matrix2, dist_coeffs2)

        # 포즈 감지
        pose_results1 = pose_model.process(cv2.cvtColor(frame1_undistorted, cv2.COLOR_BGR2RGB))
        pose_results2 = pose_model.process(cv2.cvtColor(frame2_undistorted, cv2.COLOR_BGR2RGB))

        if pose_results1.pose_landmarks and pose_results2.pose_landmarks:
            camera_coords1 = extract_camera_coords(pose_results1.pose_landmark.landmarks, frame1_undistorted)
            camera_coords2 = extract_camera_coords(pose_results2.pose_landmark.landmarks, frame2_undistorted)

            # 2D 좌표 찍기
            img1_landmarks = draw_2d_landmarks(frame1,camera_coords1,connections)
            img2_landmarks = draw_2d_landmarks(frame2, camera_coords2)
            # 3d좌표추출
            coords_3d = triangulate_3d_points(camera_coords1, camera_coords2, P1, P2, body_parts)
            scaled_coords_3d = scale_3d_coords(coords_3d)
            # 3d 좌표찍기
            draw_3d_landmarks(ax,scaled_coords_3d,connections)
        else:
            img1_landmarks =frame1
            img2_landmarks = frame2

        cv2.imshow('muran1',img1_landmarks)
        cv2.imshow('muran1', img2_landmarks)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    img1.release()
    img2.release()
    cv2.destroyAllWindows()





# 2D 카메라 좌표 추출 함수



if __name__ == "__main__":
    plt.ion()
    main()
