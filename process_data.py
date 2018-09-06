import cv2
import glob


def keyframe_extraction(video, frame_path):
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


def get_name_index(path):
    files = glob.glob(path + '*.jpg')
    indexs = []
    lables = []
    f_paths = []
    for i in files:
        file = i.split('/')[5]
        index = file.split('.')[0].split('_')[1]
        lable = file.split('.')[0].split('_')[1]
        f_path = i
        indexs.append(index)
        lables.append(lable)
        f_paths.append(f_path)
    data = {'index': indexs, 'lable': lables, 'path': f_paths}
    return data


def uniform_sampling(length, nums):
    mod = length % (nums)
    new_length = length - mod
    div = (length - mod) / nums
    print(mod, div)
    for i in range(0, new_length, int(div)):
        yield i


if __name__ == '__main__':
    path = 'D:/spwd/VQADatasetA_20180815/'
    frame_path = 'D:/spwd/VQADatasetA_20180815/frame/'
    for i in get_video(path):
        keyframe_extraction(video=i.replace('\\', '/'), frame_path=frame_path)
