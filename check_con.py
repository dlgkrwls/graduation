import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from scipy.signal import find_peaks


mp_pose = mp.solutions.pose
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


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    angle = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    return np.degrees(angle + 360) % 360



count_list =[]
def main():
    img1 = cv2.VideoCapture('squart_front.mp4')
    if not img1.isOpened():
        print("Cannot open camera")
        exit()

    fps = img1.get(cv2.CAP_PROP_FPS)
    if fps ==0:
        fps =30
    print(f"Camera FPS: {fps}")
    outfile = f"./data/fps30.mp4"
    frame_width = int(img1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(img1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #video_writer = cv2.VideoWriter(outfile, fourcc, 30, (frame_width, frame_height))

    # 모델 선언
    pose_model = setup_pose_model()
    connections = [
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ('left_ankle', 'left_heel'), ('left_ankle', 'left_foot'), ('left_foot', 'left_heel'),
        ('right_ankle', 'right_heel'),
        ('right_ankle', 'right_foot'), ('right_foot', 'right_heel'),
    ]

    is_recording =False
    video_writer =None
    frame_idx = 0
    frame_count =0
    last_squat_frame = 0  # 여기서 초기화
    min_squat_interval = int(fps * 0.5)  # 피크 탐지 후 스쿼트 탐지까지 최소 프레임 간격 (1.5초)

    while True:

        ret1,frame1 =img1.read()
        frame_count +=1
        if not ret1:
            print("카메라 안열림")
            break

        frame_idx +=1

        #포즈감지
        pose_result =pose_model.process(cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB))

        if frame_idx % 3 == 0: # 30프레임마다

            if pose_result.pose_landmarks:
                coord = extract_camera_coords(pose_result.pose_landmarks.landmark,frame1)
                count_list.append(coord["nose"][1])
            #print(coord["nose"][1])
            #img1_landmarks = draw_2d_landmarks(frame1, coord, connections)
            else:
                img1_landmarks=frame1
        #cv2.imshow('muran',frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
        # # 스쿼트 카운트: 이전 피크 탐지 이후 최소 프레임 간격을 확인
        # if frame_count - last_squat_frame > min_squat_interval:
        #     smoothed_y = smooth(count_list)
        #     min_peaks, _ = find_peaks(-smoothed_y, distance=15)

        #     # 중앙값 계산
        #     median_y = np.median(smoothed_y)

        #     # 중앙값보다 큰 극점 제외
        #     filtered_peaks = [peak for peak in min_peaks if smoothed_y[peak] < median_y]

        #     # 필터링된 극점이 존재할 경우, 마지막 피크 업데이트
        #     if filtered_peaks:
        #         last_peak = filtered_peaks[-1]
        #         last_squat_frame = frame_count  # 마지막 피크가 감지된 프레임 번호 업데이트

        #     # Write the frame to video file if recording
        # if is_recording:
        #     video_writer.write(frame1)

            # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    img1.release()
    #cv2.destroyAllWindows()



# 이동 평균을 이용한 smoothing
def smooth(y, window_size=10):
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')


if __name__ == "__main__":
    #plt.ion()
    main()

    smoothed_y = smooth(count_list)

    # 원하는 최소 프레임 간격
    desired_frame_difference = 15  # 예: 15프레임 차이

    # 극점 탐지 (내려가는 지점 찾기)q
    
    min_peaks, _ = find_peaks(-smoothed_y, distance=15)
    len_peaks = len(min_peaks)

    updated_peaks = []
    i = 0

    while i < len(min_peaks):
        if i < len(min_peaks) - 1:  # 마지막 피크를 제외한 경우
            frame_diff = min_peaks[i + 1] - min_peaks[i]

            if frame_diff < desired_frame_difference:
                # 두 피크의 중앙값 계산
                new_peak = (min_peaks[i] + min_peaks[i + 1]) // 2
                updated_peaks.append(new_peak)
                i += 2  # 두 개의 피크를 건너뜁니다
                len_peaks -= 1
            else:
                updated_peaks.append(min_peaks[i])
                i += 1
        else:
            updated_peaks.append(min_peaks[i])
            i += 1
        print(updated_peaks)
    


    # 스쿼트 횟수 계산
    squat_count = len_peaks



    # 결과 시각화
    plt.plot(count_list, label="Original Y-coords", alpha=0.5)
    plt.plot(range(len(smoothed_y)), smoothed_y, label="Smoothed Y-coords", color='green')
    plt.plot(updated_peaks, smoothed_y[updated_peaks], "rx", label="Squat Bottom")
    plt.legend()
    plt.title(f'Squat Count: {squat_count}')
    plt.show()

    

    print(f'스쿼트 횟수: {squat_count}')


    # plt.figure(figsize=(10, 5))
    # plt.plot(count_list, label='Nose Y-Coordinate')
    # plt.title('Nose Y-Coordinate over Time')
    # plt.xlabel('Frame Index')
    # plt.ylabel('Y-Coordinate')
    # plt.legend()
    # plt.grid()

    # Calculate and plot the derivative
    # count_list_diff = np.diff(count_list)
    # plt.figure(figsize=(10, 5))
    # plt.plot(count_list_diff, label='Derivative of Nose Y-Coordinate', color='red')
    # plt.title('Derivative of Nose Y-Coordinate over Time')
    # plt.xlabel('Frame Index (derivative)')
    # plt.ylabel('Derivative Value')
    # plt.legend()
    # plt.grid()

    plt.show()