import pandas as pd
import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_mask(dataframe,path_mask):

    for image_id in dataframe['image_id'].unique():
        print(image_id)
        name_img = str(image_id) + ".png"
        path_save = os.path.join(path_mask, name_img)

        point = dataframe[dataframe['image_id'] == image_id]

        boxes = point[['x', 'y', 'w', 'h']].values.astype(np.int)

        classes = point[['class']].values

        h = point['height'].values[0]
        w = point['width'].values[0]

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        numpy_image = np.zeros((h,w))
        # image = cv2.imread("./traffic_train/images/3.png")
        for key, bbox in enumerate(boxes):
            for idy in range(bbox[0], bbox[2] + 1):
                for idx in range(bbox[1], bbox[3] + 1):
                    numpy_image[idx][idy] = classes[key]

            # image = cv2.rectangle(image, (bbox[0], bbox[1]),
            #         (bbox[2],bbox[3]), (0, 0, 255),
            #         thickness=1)
        PIL_image = Image.fromarray(np.uint8(numpy_image)).convert('L')
        # plt.figure(figsize=(h/10,w/10))
        # PIL_image.show()
        # arr = np.asarray(image)
        # plt.imshow(image)
        # plt.show()
        PIL_image.save(path_save)
        # break
    return

df_main = pd.read_csv("./train.csv")
path_mask = "./traffic_train/mask/"
if not os.path.exists(path_mask):
    os.mkdir(path_mask)
make_mask(df_main,path_mask)
