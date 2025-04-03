import cv2

video = cv2.VideoCapture('/home/parth/CV_for_viatouch_media/data_for_Mapping_logic/office_transaciton_1/media.mp4')
frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    frame_count+=1
print(frame_count)