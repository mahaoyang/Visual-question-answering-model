import cv2


def keyframe_extraction(path, video):
    cap = cv2.VideoCapture(path + 'test/' + video)

    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧速

    fWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    fHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fNums = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print(fps, fWidth, fHeight, fNums)

    success, frame = cap.read()
    i = 0
    while success:
        success, arr = cv2.imencode('.jpg', frame)
        a = arr.tostring()
        fp = open(path + 'frame/' + str(i) + '.jpg', 'wb')
        fp.write(a)
        fp.close()
        i = i + 1
        success, frame = cap.read()


if __name__ == '__main__':
    path = 'D:/spwd/VQADatasetA_20180815/'
    keyframe_extraction(path, 'ZJL6.mp4')
