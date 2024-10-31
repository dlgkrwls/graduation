import cv2 as cv
import numpy as np
import pandas as pd
import os
#D는 프레임 버리기, S는 시작점, 1은 TRUE, 2는 허리, 다리오무림, 3은 무릎, 다리벌림

labeling= []
save_csv_dir = './train_data_side/label_csv/knee_side.csv'
video_path = './raw_data/knee_side.mp4'
save_dir = './train_data_side/knee_side'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_dir = save_dir+'/'
video = cv.VideoCapture(video_path)
count = 0
start = 0
while True:
    ret,frame = video.read()
    if count % 5 ==0:
        if not ret:
            break
        print(frame.shape)
        frame = cv.resize(frame,(400,732),interpolation=cv.INTER_AREA)
        cv.imshow('a',frame)
        key = cv.waitKey() & 0xFF

        if key == ord('1'):  # true
            labeling.append(1)
            cv.imwrite(save_dir+str(count)+'.jpg',frame)
        elif key == ord('2'):  #false
            labeling.append(2) # 다리 오무림
            cv.imwrite(save_dir+str(count)+'.jpg',frame)
        elif key == ord('3'): # 다리벌림
            labeling.append(3)
            cv.imwrite(save_dir+str(count)+'.jpg',frame)
        elif key == ord('s'):
            labeling.append('start')
            start += 1
            print(start)
            cv.imwrite(save_dir+str(count)+'.jpg',frame)
        elif key == ord('d'):
            labeling.append('dump')

    count+=1

cv.destroyAllWindows()

print(labeling)

labeling = np.array(labeling)

df = pd.DataFrame(labeling)
df.to_csv(save_csv_dir)
