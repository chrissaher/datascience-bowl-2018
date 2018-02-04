import utils.constants as const
import utils.pathbuilder as upb
import os
from PIL import Image
import numpy as np
FOLDER_NAME = './train-img/'
IGNORED_FILES = ['.DS_Store']

if os.path.isdir(FOLDER_NAME) == False:
    os.makedirs(FOLDER_NAME)
    print("Directory created correctly")
else:
    print("Directory alrerady exists")


train_set = [x for x in os.listdir(const.STAGE1_TRAIN_PATH)]
cont = 0
for train_id in train_set:
    if train_id in IGNORED_FILES:
        continue
    images_path = upb.images_path(train_id)
    images = [x for x in os.listdir(images_path)]
    image_id = images[0]
    if image_id == IGNORED_FILES:
        continue
    image_path = upb.image_path(train_id, image_id)
    print(image_id)
    with Image.open(image_path,) as img:
        img.save(FOLDER_NAME + str(cont) + '.png', 'PNG')
    cont += 1

print("Total of images analyzed: ", cont)
