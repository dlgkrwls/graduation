from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
import cv2
import threading
import os
#from model.main import PoseEstimator

# backend_main = PoseEstimator()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


UPLOAD_FOLDER = './uploaded_videos'
PROCESSED_FOLDER = './processed_videos'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# 두 개의 웹캠 스트림을 위한 글로벌 변수
cap1 = cv2.VideoCapture(0)  # 첫 번째 웹캠
cap2 = cv2.VideoCapture(1)  # 두 번째 웹캠
recording = False  # 녹화 상태
out1 = None  # 첫 번째 웹캠 녹화 파일
out2 = None  # 두 번째 웹캠 녹화 파일

def generate_frames(camera):
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames(cap1), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames(cap2), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_recording', methods=['POST'])
def start_recording():
    global out1, out2, recording
    
    if recording:
        return jsonify({"message": "이미 녹화 중입니다."}), 400

    recording = True

    # 첫 번째 웹캠 녹화 설정
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # fps1 = cap1.get(cv2.CAP_PROP_FPS)
    # fps2 = cap2.get(cv2.CAP_PROP_FPS)
    out1 = cv2.VideoWriter(os.path.join(UPLOAD_FOLDER, 'output1.mp4'), fourcc, 5.0, (640, 480))

    # 두 번째 웹캠 녹화 설정
    out2 = cv2.VideoWriter(os.path.join(UPLOAD_FOLDER, 'output2.mp4'), fourcc, 5.0, (640, 480))

    def record():
        frame_idx = 0
        while recording:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            frame_idx+=1
            if frame_idx % 3 == 0: 
                if ret1:
                    out1.write(frame1)
                if ret2:
                    out2.write(frame2)

    t = threading.Thread(target=record)
    t.start()

    return jsonify({"message": "녹화를 시작했습니다."})


@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording, out1, out2

    if not recording:
        return jsonify({"message": "녹화가 진행 중이 아닙니다."}), 400

    recording = False
    out1.release()
    out2.release()

    return jsonify({"message": "녹화를 중지했습니다."})


# 영상 흑백 변환 및 업로드
@app.route('/upload_and_convert', methods=['POST'])
def upload_and_convert():
    video_file1 = os.path.join(UPLOAD_FOLDER, 'output1.mp4')
    video_file2 = os.path.join(UPLOAD_FOLDER, 'output2.mp4')
    

    if not os.path.exists(video_file1) or not os.path.exists(video_file2):
        return jsonify({'error': '녹화된 영상이 없습니다.'}), 400
    
    # 흑백 변환
    processed_video1 = convert_to_grayscale(video_file1, 'output1.mp4')
    processed_video2 = convert_to_grayscale(video_file2, 'output2.mp4')
    print(processed_video1)
    
    # 흑백 변환된 영상의 경로를 반환 (send_file로 제공)
    return jsonify({
        'message': '흑백 변환 완료',
        'video1': f'./processed/{os.path.basename(processed_video1)}',
        'video2': f'/processed/{os.path.basename(processed_video2)}'
    })

# 변환된 동영상을 클라이언트로 전송하는 엔드포인트
@app.route('/processed/<filename>')
def get_processed_video(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename), as_attachment=False, mimetype='video/mp4')




# 영상 흑백 변환 함수
def convert_to_grayscale(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    
    processed_video_path = os.path.join(PROCESSED_FOLDER, f'convert_{filename}')
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # MP4 코덱 사용
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray_frame)

    cap.release()
    out.release()


    return processed_video_path


@app.route('/video/<path:filename>')
def serve_video(filename):
    video_path = f"C:/Users/NHJ/Downloads/졸프_UI/UI/data/{filename}" 
    #video_path = f"C:/Users/NHJ/Downloads/졸프_UI/UI/processed_videos/{filename}" 
    mimetype = 'video/mp4'
    return send_file(video_path, mimetype=mimetype)

if __name__ == '__main__':
    app.run(debug=True)