import utils.constants as const
import re

def images_path(train_id):
    return re.sub(const.RE_TRAIN_ID, train_id, const.IMAGES_PATH)

def masks_path(train_id):
    return re.sub(const.RE_TRAIN_ID, train_id, const.MASKS_PATH)

def image_path(train_id, image_id):
    path = re.sub(const.RE_TRAIN_ID, train_id, const.IMAGE_PATH)
    path = re.sub(const.RE_IMAGE_ID, image_id, path)
    return path

def maks_path(train_id, mask_id):
    path = re.sub(const.RE_TRAIN_ID, train_id, const.IMAGE_PATH)
    path = re.sub(const.RE_MASK_ID, mask_id, path)
    return path
