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
# 필요함수
def smooth(y, window_size=15):
    hamming_window = np.hamming(window_size)
    smoothed_y = np.convolve(y, hamming_window/np.sum(hamming_window), mode='valid')
    return smoothed_y


def squart_count(count_list,knee_angle):
    updated_peaks = []
    # plt.ion()
    smoothed_y = smooth(count_list)
    
    # 원하는 최소 프레임 간격
    desired_frame_difference = 15  # 예: 15프레임 차이

    # 극점 탐지 (내려가는 지점 찾기)

    min_peaks, _ = find_peaks(smoothed_y, distance=40,height=100)
    max_peaks, _ = find_peaks(-smoothed_y, distance=30,prominence=1)

    len_peaks = len(min_peaks)
    print(min_peaks)
    print(min_peaks.dtype)
    for i in min_peaks:
        print("TLQKF 김시진",knee_angle[i])
        if knee_angle[i] < 171:
            updated_peaks.append(i)

    # 스쿼트 횟수 계산
    squat_count = len_peaks
    plt.plot(smoothed_y)
    plt.scatter(updated_peaks, smoothed_y[updated_peaks], color='red')
    plt.scatter(max_peaks, smoothed_y[max_peaks], color='blue')

    plt.show()

    return updated_peaks, squat_count
