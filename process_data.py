import cv2
import glob


def keyframe_extraction(video,frame_path):
    cap = cv2.VideoCapture(video)

    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧速

    fWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    fHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fNums = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # print(fps, fWidth, fHeight, fNums)

    success, frame = cap.read()
    i = 0
    while success:
        print('%s,%s' % (video, i))
        success, arr = cv2.imencode('.jpg', frame)
        a = arr.tostring()
        fp = open(frame_path + video.split('.')[0].split('/')[4] + '_' + str(i) + '.jpg', 'wb')
        fp.write(a)
        fp.close()
        i = i + 1
        success, frame = cap.read()


def get_video(path):
    return glob.glob(path + 'train/' + '*.mp4')


if __name__ == '__main__':
    path = 'D:/spwd/VQADatasetA_20180815/'
    frame_path = 'D:/spwd/VQADatasetA_20180815/frame/'
    for i in get_video(path):
        keyframe_extraction(video=i.replace('\\', '/'), frame_path=frame_path)
