import re

STAGE1_TRAIN_PATH = "./data-raw/stage1_train"
STAGE1_TEST_PATH = "./data-raw/stage1_test"
IMAGES_PATH = "./data-raw/stage1_train/:id/images"
MASKS_PATH = "./data-raw/stage1_train/:id/masks"
IMAGE_PATH = "./data-raw/stage1_train/:id/images/:image_id"
MASK_PATH = "./data-raw/stage1_train/:id/images/:mask_id"
RE_TRAIN_ID = re.compile(":id")
RE_IMAGE_ID = re.compile(":image_id")
RE_IMAGE_ID = re.compile(":mask_id")
