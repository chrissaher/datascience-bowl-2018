import utils.constants as const
import utils.pathbuilder as upb
import os
from PIL import Image
import numpy as np

IGNORED_FILES = ['.DS_Store']
train_set = [x for x in os.listdir(const.STAGE1_TRAIN_PATH)]
train_id = train_set[0]

images_path = upb.images_path(train_id)
images = [x for x in os.listdir(images_path)]
image_id = images[0]
image_path = upb.image_path(train_id, image_id)

with Image.open(image_path) as img:
    img = img.convert('LA').convert('RGB')
    arr = np.array(img)
    print(arr.shape)
    print(arr[0][0])


print("END")
