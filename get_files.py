import utils.constants as const
import utils.pathbuilder as upb
import os
from PIL import Image
import numpy as np

IGNORED_FILES = ['.DS_Store']
train_set = [x for x in os.listdir(const.STAGE1_TRAIN_PATH)]

for train_id in train_set:

    if train_id in IGNORED_FILES:
        continue

    # Get the raw image for training
    images_path = upb.images_path(train_id)
    images = [x for x in os.listdir(images_path)]
    image_id = images[0]
    if image_id == IGNORED_FILES:
        continue

    image_path = upb.image_path(train_id, image_id)
    with Image.open(image_path) as img:
        # Dimensions are (256, 256, 4)
        # Need to drop the alpha column to become (256, 256, 3)
        # Check PIL documentation
        arr = np.array(img)

    # Get the raw masks for training
    mask_path = upb.masks_path(train_id)
    masks = [x for x in os.listdir(mask_path)]
    for mask_id in masks:
        mask_path = upb.mask_path(train_id, mask_id)
    with Image.open(mask_path) as img:
        # Pending for checking
        arr = np.array(img)
