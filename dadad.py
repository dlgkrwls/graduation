import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from scipy.signal import find_peaks

def calculate_angle(a, b, c):
     ba = a - b
     bc = c - b
     angle = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
     return np.degrees(angle + 360) % 360

# 이동 평균을 이용한 smoothing
#필요함수
def smooth(y, window_size=35):
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')


def squart_count(count_list):
    #plt.ion()
    smoothed_y = smooth(count_list)

    # 원하는 최소 프레임 간격
    desired_frame_difference = 30  # 예: 15프레임 차이

    # 극점 탐지 (내려가는 지점 찾기)
    
    min_peaks, _ = find_peaks(-smoothed_y, distance=40)
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

    return updated_peaks, squat_count
