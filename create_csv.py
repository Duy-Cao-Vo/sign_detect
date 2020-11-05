import json
import cv2
import pandas as pd
def make_csv(path_data, path_data_json, save_path ):
    json_train = json.load(open(path_data_json,'r'))

    image_id = []
    width    = []
    height   = []
    label    = []
    classes  = []

    x, y, w, h, area = [], [], [], [], []
    mp = dict()

    for img in json_train['annotations']:
      image_id.append(img['image_id'])
      x.append(img['bbox'][0])
      y.append(img['bbox'][1])
      w.append(img['bbox'][2])
      h.append(img['bbox'][3])
      area.append(img['area'])
      label.append(img['category_id'])
      # print(path_data+str(img['image_id'])+".png")
      image = cv2.imread(path_data+str(img['image_id'])+".png")
      width.append(image.shape[1])
      height.append(image.shape[0])
      # image = cv2.rectangle(image, (img['bbox'][0],img['bbox'][1]), (img['bbox'][0]+img['bbox'][2],img['bbox'][1]+img['bbox'][3]), (0,0,255), thickness=1)
      # cv2_imshow(image)
      classes.append(img['category_id'])
      print(img['category_id'])
      if img['category_id'] not in mp.keys():
        mp[img['category_id']] =0
      else:
        mp[img['category_id']] +=1

    print(mp)
    import pandas as pd
    data_train = pd.DataFrame({
        'image_id':image_id,
        'width':width,
        'height': height,
        'x':x,
        'y':y,
        'w':w,
        'h':h,
        'class': classes
    }).to_csv(save_path)


path_train = './traffic_train/images/''
path_train_json = "./traffic_train/train_traffic_sign_dataset.json"
make_csv(path_train, path_train_json, save_path='train.csv')