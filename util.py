import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import Pose_check
import counting_f
mp_pose = mp.solutions.pose
def setup_camera():
    camera_matrix1 = np.array([[497.39900884, 0, 323.96380382], [0, 496.67512836, 250.05988172], [0, 0, 1]])
    dist_coeffs1 = np.array([[0.02950153, 0.0615235, -0.00102225, -0.00196549, -0.15609333]])
    camera_matrix2 = np.array([[506.35817686, 0, 330.64228386], [0, 506.09559386, 238.95273757], [0, 0, 1]])
    dist_coeffs2 = np.array([0.05938374, -0.05269976, 0.00271985, -0.00217994, -0.07622173])
    return camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2



def setup_pose_model():
    return mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 왜곡 보정 이미지 반환
def undistort_image(frame, camera_matrix, dist_coeffs):
    return cv2.undistort(frame, camera_matrix, dist_coeffs)
def apply_smoothing(pose_data, model, issave=False, save_path=None):
    # Convert pose data to a numpy array and ensure it has the correct shape
    pose_data = np.array(pose_data)
    input_data = pose_data.reshape(len(pose_data), -1) if pose_data.ndim == 3 else pose_data

    smoothed_data = []
    window_size = model.window_size

    # Process data in sliding windows
    for start in range(0, input_data.shape[0] - window_size + 1, window_size):
        end = start + window_size
        window_data = input_data[start:end]
        input_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)

        # Use the model to smooth the data
        with torch.no_grad():
            smoothed_output = model(input_tensor).squeeze(0).permute(1, 0).numpy()
            smoothed_data.append(smoothed_output)

    # Concatenate all smoothed windows and reshape to the original format
    smoothed_data = np.concatenate(smoothed_data, axis=0).reshape(-1, 17, 2)

    if issave and save_path:
        np.save(save_path, smoothed_data)

    return smoothed_data


def convert_smoothed_to_dict(smoothed_coords, frame_shape):
    """Converts smoothed coordinates to a dictionary format for easy access."""
    return {
        "nose": smoothed_coords[0] * frame_shape,
        "left_shoulder": smoothed_coords[1] * frame_shape,
        "right_shoulder": smoothed_coords[2] * frame_shape,
        "left_elbow": smoothed_coords[3] * frame_shape,
        "right_elbow": smoothed_coords[4] * frame_shape,
        "left_wrist": smoothed_coords[5] * frame_shape,
        "right_wrist": smoothed_coords[6] * frame_shape,
        "left_hip": smoothed_coords[7] * frame_shape,
        "right_hip": smoothed_coords[8] * frame_shape,
        "left_knee": smoothed_coords[9] * frame_shape,
        "right_knee": smoothed_coords[10] * frame_shape,
        "left_ankle": smoothed_coords[11] * frame_shape,
        "right_ankle": smoothed_coords[12] * frame_shape,
        "left_heel": smoothed_coords[13] * frame_shape,
        "right_heel": smoothed_coords[14] * frame_shape,
        "left_foot": smoothed_coords[15] * frame_shape,
        "right_foot": smoothed_coords[16] * frame_shape
    }


def extract_coco_format(results, coco_indices):
    """Extracts landmarks in COCO format."""
    landmarks = results.pose_landmarks.landmark
    xy_coords = [(landmarks[idx].x, landmarks[idx].y) for idx in coco_indices]
    return xy_coords

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
def draw_2d_landmarks(frame,coords,connections,body_part):
    for part,coord in coords.items():
        if part in body_part:
            cv2.circle(frame,(int(coord[0]),int(coord[1])),5,(255,0,0),-1)

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

def get_time_in_seconds(frame_count, fps):
    return frame_count / fps



def count_injury(stance,knee_angle,knee_position,count_list,health_warning):

    for i in range(len(count_list)):
        frame_data={}
        start = count_list[i]
        end= count_list[i+1] if i+1<len(count_list) else len(stance)
        frame_data['count']=int(i+1)
        frame_data['start_indx']=int(start)
        frame_data['end_indx'] = int(end)

        # stance 구간 내에 0이 있으면 0으로 설정, 없으면 1
        if 0 in stance[start:end]:
            frame_data['stance'] = bool(0)
        else:
            frame_data['stance'] = bool(1)

        # knee_angle 구간 내에 0이 있으면 0으로 설정, 없으면 1
        if 0 in knee_angle[start:end]:
            frame_data['knee_angle'] = bool(0)
        else:
            frame_data['knee_angle'] = bool(1)

        # knee_position 구간 내에 0이 있으면 0으로 설정, 없으면 1
        if 0 in knee_position[start:end]:
            frame_data['knee_position'] = bool(0)
        else:
            frame_data['knee_position'] = bool(1)




        # if stance[start:end].count(0) >= 20:
        #     frame_data['stance'] = 0
        # else:
        #     frame_data['stance'] = 1
        #
        # if knee_angle[start:end].count(0) >= 20:
        #     frame_data['knee_angle'] = 0
        # else:
        #     frame_data['knee_angle'] = 1
        #
        #     # knee_position 구간 내에 0이 20개 이상이면 0으로 설정, 그렇지 않으면 1
        # if knee_position[start:end].count(0) >= 20:
        #     frame_data['knee_position'] = 0
        # else:
        #     frame_data['knee_position'] = 1

        health_warning['frames'].append(frame_data)
