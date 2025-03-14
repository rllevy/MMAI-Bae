import numpy as np
import os
from tqdm import tqdm
# data = np.load('train_image_features.npy',allow_pickle=True).item()

train_image_directory = '/Users/liamkopp/Downloads/scene_img_abstract_v002_train2015'

for img_name in tqdm(os.listdir(train_image_directory), desc='Processing Training Images', leave=False, miniters=100):
    print(img_name)