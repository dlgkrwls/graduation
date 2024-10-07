import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import json
import time
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

count_list =[]
def main():
    img1 = cv2.VideoCapture('data/count_test.mp4')
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

    frame_count =0

    while True:

        ret1,frame1 =img1.read()
        frame_count +=1
        if not ret1:
            print("카메라 안열림")
            break

        #포즈감지
        pose_result =pose_model.process(cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB))

        if pose_result.pose_landmarks:
            coord = extract_camera_coords(pose_result.pose_landmarks.landmark,frame1)
            count_list.append(coord["nose"][1])
            print(coord["nose"][1])
            #img1_landmarks = draw_2d_landmarks(frame1, coord, connections)
        else:
            img1_landmarks=frame1
        cv2.imshow('muran',frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('s'):
            if not is_recording:
                is_recording = True
                video_writer = cv2.VideoWriter(outfile, fourcc, fps, (frame_width, frame_height))
                print("Recording started...")
                print(f'시작:프레임',frame_count)
                print("현재시간",time.asctime(time.localtime()))

            # Stop recording and save when 'e' is pressed
        if cv2.waitKey(1) & 0xFF == ord('e'):
            if is_recording:
                is_recording = False
                video_writer.release()
                video_writer = None
                print("Recording saved.")
                print(f"저장 프레임",frame_count)
                print("현재시간", time.asctime(time.localtime()))

            # Write the frame to video file if recording
        if is_recording:
            video_writer.write(frame1)

            # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    img1.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    #plt.ion()
    main()

    plt.figure(figsize=(10, 5))
    plt.plot(count_list, label='Nose Y-Coordinate')
    plt.title('Nose Y-Coordinate over Time')
    plt.xlabel('Frame Index')
    plt.ylabel('Y-Coordinate')
    plt.legend()
    plt.grid()

    # Calculate and plot the derivative
    count_list_diff = np.diff(count_list)
    plt.figure(figsize=(10, 5))
    plt.plot(count_list_diff, label='Derivative of Nose Y-Coordinate', color='red')
    plt.title('Derivative of Nose Y-Coordinate over Time')
    plt.xlabel('Frame Index (derivative)')
    plt.ylabel('Derivative Value')
    plt.legend()
    plt.grid()

    plt.show()
