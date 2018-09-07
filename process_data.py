import cv2
import glob
import json


def keyframe_extraction(video, frame_path):
    cap = cv2.VideoCapture(video)

    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧速

    fWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    fHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fNums = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # print(fps, fWidth, fHeight, fNums)

    success, frame = cap.read()
    i = 0
    path_last = []
    while success:
        print('%s,%s' % (video, i))
        success, arr = cv2.imencode('.jpg', frame)
        a = arr.tostring()
        path_temp = frame_path + video.split('.')[0].split('/')[4] + '_' + str(i) + '.jpg', 'wb'
        path_last.append(path_temp[0])
        fp = open(path_temp[0], 'wb')
        fp.write(a)
        fp.close()
        i = i + 1
        success, frame = cap.read()
    return i, path_last


def get_video(path):
    return glob.glob(path + 'train/' + '*.mp4')


def get_name_index(path):
    files = glob.glob(path + '*.jpg')
    indexs = []
    lables = []
    f_paths = []
    for i in files:
        i = i.replace('\\', '/')
        file = i.split('/')[-1]
        index = file.split('.')[0].split('_')[1]
        lable = file.split('.')[0].split('_')[0]
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
    div = int(div)
    print(mod, div)
    for i in range(0, new_length, div):
        yield i


def sampling():
    with open('data.json', 'r') as f:
        data = json.load(f)
    for i in data:
        max_index = data[i]['max_index']
        data[i]['sample_index'] = []
        for ii in uniform_sampling(max_index, 10):
            data[i]['sample_index'].append(ii)
    with open('data.json', 'w') as f:
        json.dump(data, f)


def video2pic(path, frame_path):
    data = dict()
    for i in get_video(path):
        i = i.replace('\\', '/')
        kf = keyframe_extraction(video=i, frame_path=frame_path)
        data[i.split('.')[0].split('/')[4]] = dict()
        data[i.split('.')[0].split('/')[4]]['max_index'] = kf[0]
        data[i.split('.')[0].split('/')[4]]['path_list'] = kf[1]
    with open('data.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    path = 'D:/spwd/VQADatasetA_20180815/'
    frame_path = 'D:/spwd/VQADatasetA_20180815/frame/'
