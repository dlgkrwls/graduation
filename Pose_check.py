#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# 정면 캠
# 스탠스 발 어깨 너비 벌리기
def check_stance(coords_2d_1, tolerance=10):
    left_foot = coords_2d_1['left_wrist']
    right_foot = coords_2d_1['right_wrist']
    left_shoulder = coords_2d_1['left_shoulder']
    right_shoulder = coords_2d_1['right_shoulder']
    
    #유클리드 거리가 안정적이라고 하더라
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    foot_width = np.linalg.norm(left_foot - right_foot)
    
    # 어깨너비와 발너비 비교
    stance_correct = abs(foot_width - shoulder_width) / shoulder_width <= tolerance
    
    return "어깨 너비 맞음." if stance_correct else "어깨너비 아님."

# 이건 목 각도인데 완전 지피티 쓴거
# 너무 위나 아래를 보지 않도록 해야되는데 그게 어느정도인지 모름
# 어깨 좌표랑 코 좌표로 구하는거임
def check_neck_angle(coords_2d_1):
    # 좌표 가져오기
    left_shoulder = coords_2d_1['left_shoulder']
    right_shoulder = coords_2d_1['right_shoulder']
    nose = coords_2d_1['nose']
    
    # 왼쪽 어깨에서 코로 향하는 벡터
    vector_left_shoulder = nose - left_shoulder
    
    # 오른쪽 어깨에서 코로 향하는 벡터
    vector_right_shoulder = nose - right_shoulder
    
    # 벡터 간의 내적을 이용해 각도 계산
    cos_theta = np.dot(vector_left_shoulder, vector_right_shoulder) / (np.linalg.norm(vector_left_shoulder) * np.linalg.norm(vector_right_shoulder))
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 라디안으로 변환
    neck_angle = np.degrees(angle_rad)  # 도(degree)로 변환
    
    return neck_angle

# 허리 굽힘 측정을 위한 허리 길이 계산
# 다른 방법도 생각해야됨
# 이때 길이의 비교는 이전 프레임과의 오차로 계산해야될듯? ex)이전 허리길이 10 현재 허리길이 5 <= 굽힘 발생
def check_stance(coords_2d_2, tolerance=0.1):
    left_hip = coords_2d_2['left_hip']
    right_hip = coords_2d_2['right_hip']
    left_shoulder = coords_2d_2['left_shoulder']
    right_shoulder = coords_2d_2['right_shoulder']
    
    #유클리드 거리가 안정적이라고 하더라
    left_width = np.linalg.norm(left_shoulder - left_hip)
    right_width = np.linalg.norm(right_shoulder - right_hip)
    
    # 어깨너비와 발너비 비교
    stance_correct = abs(left_width + right_width) / 2
    
    return stance_correct

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# 측면 캠
# 무릎이 앞발끝 넘는지
def check_knee_position(coords_2d_2, side='left'):
    # 측면에서의 무릎과 발끝 비교 (x 좌표만 사용)
    if side == 'left':
        knee_x = coords_2d_2['left_knee'][0] 
        foot_x = coords_2d_2['left_foot'][0] 
    else:
        knee_x = coords_2d_2['right_knee'][0] 
        foot_x = coords_2d_2['right_foot'][0] 
    
    # 무릎이 발끝을 넘었는지 확인
    # 신체 가동범위에 따라 일부 넘을 수 있기 때문에 약간의 오차를 줘야될듯
    # if 안넘은: Good, else if 조금 넘음: 자신의 가동 범위를 확인해보셈 , else: 자세 이상함
    if knee_x > foot_x:
        return "무릎이 발끝 넘음."
    else:
        return "무릎이 발끝 안넘음 잘했음."
    
# 무릎 각도
def calculate_knee_angle(coords_2d_2, side='left'):
    # 측면에서의 엉덩이, 무릎, 발목 좌표
    if side == 'left':
        hip = coords_2d_2['left_hip']
        knee = coords_2d_2['left_knee']
        ankle = coords_2d_2['left_ankle']
    else:
        hip = coords_2d_2['right_hip']
        knee = coords_2d_2['right_knee']
        ankle = coords_2d_2['right_ankle']
    
    # 벡터 계산 (2D 평면에서 x, y 좌표만 사용)
    v1 = hip - knee  # 엉덩이 -> 무릎 벡터
    v2 = ankle - knee  # 발목 -> 무릎 벡터
    
    # 내적(dot product)과 벡터 크기를 이용한 각도 계산
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_rad = np.arccos(cos_theta)  # 라디안 값의 각도
    angle_deg = np.degrees(angle_rad)  # 각도를 도(degree)로 변환
    
    if angle_deg > 60 and angle_deg < 90 :
        print("하프 스퀏")
    elif angle_deg > 90 and angle_deg < 120 :
        print("스퀏")
    elif angle_deg > 120:
        print("너무 깊음")
    else :
        print("너무 안 앉음")
    
    return f"무릎의 굽힘 각도: {angle_deg:.2f}"


# 허리 굽힘 측정을 위한 허리 길이 계산
# 측면 버전
def check_stance(coords_2d_2, side='left'):
    if side == 'left':
        left_hip = coords_2d_2['left_hip']
        left_shoulder = coords_2d_2['left_shoulder']
        left_width = np.linalg.norm(left_shoulder - left_hip)
        waist_length = abs(left_width)
    else:
        right_hip = coords_2d_2['right_hip']
        right_shoulder = coords_2d_2['right_shoulder']
        right_width = np.linalg.norm(right_shoulder - right_hip)
        waist_length = abs(right_width)
        
    return waist_length
