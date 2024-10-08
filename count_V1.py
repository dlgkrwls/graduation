import cv2

def display_video_with_frames(video_path):
    # 비디오 파일을 읽기 위한 VideoCapture 객체 생성
    cap = cv2.VideoCapture(video_path)

    # 비디오 파일이 열렸는지 확인
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return

    # 비디오의 프레임을 하나씩 읽어와서 출력
    while cap.isOpened():
        ret, frame = cap.read()

        # 더 이상 읽을 프레임이 없을 때
        if not ret:
            print("비디오 파일이 끝났습니다.")
            break

        # 현재 프레임 번호 출력
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"현재 프레임 번호: {current_frame}")

        # 프레임에 현재 프레임 번호를 표시
        cv2.putText(frame, f"Frame: {current_frame}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 영상을 화면에 표시
        cv2.imshow('Video', frame)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

# 사용 예시: 비디오 파일 경로를 입력하세요
video_path = 'sdata/squat_cam2_45.mp4'  # 파일 경로에 맞게 수정
display_video_with_frames(video_path)
