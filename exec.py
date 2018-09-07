import json
import pickle
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

from model import vqa_model

with open('data.json', 'r') as f:
    data = json.load(f)

labels = []
sample_path = []
for i in data:
    sample_index = data[i]['sample_index']
    for ii in sample_index:
        sample_path.append(ii)

sample_array = []
for i in sample_path:
    pic = load_img(i, target_size=(64, 64))
    pic = img_to_array(pic)
    sample_array.append(pic)
sample_array = np.array(sample_array)
with open('sample_array.pickle', 'wb') as f:
    pickle.dump(sample_array, f)
