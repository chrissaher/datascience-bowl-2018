import numpy as np
import skimage.io
from skimage.color import rgba2rgb, rgb2gray, gray2rgb
import utils.constants as const
import os
import random

def get_image_mask_from_id(img_id):
    images_id = [x for x in os.listdir(const.STAGE1_TRAIN_PATH)]
    image_id = images_id[img_id]

    image_path = os.path.join(const.STAGE1_TRAIN_PATH, image_id, 'images', '{}.png'.format(image_id))
    mask_paths = os.path.join(const.STAGE1_TRAIN_PATH, image_id, 'masks', '*.png')

    image = gray2rgb(rgb2gray(skimage.io.imread(image_path)))
    masks = skimage.io.imread_collection(mask_paths).concatenate()
    mask = np.zeros(image.shape[:2], np.uint16)

    for mask_idx in range(masks.shape[0]):
        mask[masks[mask_idx] > 0] = 1

    image = np.uint8(image * 255)
    if image.mean() > 255 / 2:
        image = 255 - image

    return image, mask

def get_train_and_labels_from_image(image, mask, rate = 0.5, padding = 50, verbose = False):
    # Rate is the rate between positive and negative examples. Expected a value [0,1]
    # Rate = positive / negative 
    # In algorithm, we use [0-1000] to be more precise
    # It's assumed that image and mask are in range [0,1]

    w, h, _ = image.shape
    all_zeros_added = False
    image_padded = apply_padding(image, padding)
    mask_padded = apply_padding(mask, padding)
    training_examples = []
    label_examples = []

    pos = 0
    neg = 1
    
    for x in range(w):
        for y in range(h):
            nx = x + 2 * padding + 1
            ny = y + 2 * padding + 1
            sample = mask_padded[x: nx ,y : ny]
            if np.sum(sample) > 0:
                pos += (mask_padded[x + padding + 1][y + padding + 1] == 1)
                neg += (mask_padded[x + padding + 1][y + padding + 1] == 0)
    
    real_rate = pos * 100. / neg
    real_rate *= (rate * 10)
    
    pos = 0
    neg = 1
    
    for x in range(w):
        for y in range(h):
            nx = x + 2 * padding + 1
            ny = y + 2 * padding + 1
            sample = mask_padded[x: nx ,y : ny]

            if np.sum(sample) > 0:
                if (mask_padded[x + padding + 1][y + padding + 1] == 0):
                    if random.randint(0, 1000) < real_rate:
                        img_sample = image_padded[x: nx, y : ny, :]
                        training_examples.append(img_sample)
                        label_examples.append(np.array([0,1]))
                        neg += 1
                else:
                    img_sample = image_padded[x: nx, y : ny, :]
                    training_examples.append(img_sample)
                    label_examples.append(np.array([1,0]))
                    pos += 1
            else:
                if all_zeros_added == False:
                    all_zeros_added = True
                    img_sample = image_padded[x: nx, y : ny, :]
                    training_examples.append(img_sample)
                    label_examples.append(np.array([0,1]))

    train = np.array(training_examples)
    labels = np.array(label_examples)

    if verbose: 
        print("---------------------")
        print("Positive examples taken : ", pos)
        print("Negative examples taken : ", neg)
        print("---------------------")
        
    return train, labels

def get_train_and_labels_from_image_3_classes(image, mask, rate = 0.5, padding = 50, verbose = False):
    # Rate is the rate between positive and negative examples. Expected a value [0,1]
    # Rate = positive / negative 
    # In algorithm, we use [0-1000] to be more precise
    # It's assumed that image and mask are in range [0,1]

    w, h, _ = image.shape
    all_zeros_added = False
    image_padded = apply_padding(image, padding)
    mask_padded = apply_padding(mask, padding)
    training_examples = []
    label_examples = []

    pos = 0
    neg = 1
    
    for x in range(w):
        for y in range(h):
            nx = x + 2 * padding + 1
            ny = y + 2 * padding + 1
            sample = mask_padded[x: nx ,y : ny]
            if np.sum(sample) > 0:
                pos += (mask_padded[x + padding + 1][y + padding + 1] == 1)
                neg += (mask_padded[x + padding + 1][y + padding + 1] == 0)
    
    real_rate = pos * 100. / neg
    real_rate *= (rate * 10)
    
    pos = 0
    neg = 1
    
    for x in range(w):
        for y in range(h):
            nx = x + 2 * padding + 1
            ny = y + 2 * padding + 1
            sample = mask_padded[x: nx ,y : ny]

            if np.sum(sample) > 0:
                if (mask_padded[x + padding + 1][y + padding + 1] == 0):
                    if random.randint(0, 1000) < real_rate:
                        img_sample = image_padded[x: nx, y : ny, :]
                        training_examples.append(img_sample)
                        label_examples.append(np.array([1,0,0]))
                        neg += 1
                else:
                    img_sample = image_padded[x: nx, y : ny, :]
                    neighbors = np.sum(mask_padded[x + padding: x + padding + 2][y + padding: y + padding + 2])
                    if neighbors == 9:
                        label_examples.append(np.array([0,1,0])) # Is not border
                    else: label_examples.append(np.array([0,0,1])) # Is border
                    training_examples.append(img_sample)
                    
                    pos += 1
            else:
                if all_zeros_added == False:
                    all_zeros_added = True
                    img_sample = image_padded[x: nx, y : ny, :]
                    training_examples.append(img_sample)
                    label_examples.append(np.array([1,0,0]))

    train = np.array(training_examples)
    labels = np.array(label_examples)

    if verbose: 
        print("---------------------")
        print("Positive examples taken : ", pos)
        print("Negative examples taken : ", neg)
        print("---------------------")
        
    return train, labels

def get_test_from_image(image, fixed_w =0, fixed_h = 0, padding =50):
    # It's assumed that image is in range [0,1]
    
    w, h, _ = image.shape
    w = fixed_w if fixed_w > 0 else w
    h = fixed_h if fixed_h > 0 else h
    image_padded = apply_padding(image, padding)
    to_predict = []

    for x in range(w):
        for y in range(h):
            nx = x + 2 * padding + 1
            ny = y + 2 * padding + 1
            img_sample = image_padded[x: nx, y : ny, :]
            to_predict.append(img_sample)
            
    return np.array(to_predict)

def from_2_mask_to_3_mask(mask):
    w, h = mask.shape
    new_mask = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            neighbors = np.sum(mask[i - 1: i + 2, j - 1: j + 2])
            if neighbors == 9:
                new_mask[i][j] = 2
            elif neighbors > 0:
                new_mask[i][j] = 3
    return new_mask

def apply_padding(image_array, padding = 50):
    if len(image_array.shape) == 2:
        w, h = image_array.shape
        padded_array = np.zeros((w + padding * 2, h + padding * 2))
    else:
        w, h, c = image_array.shape
        padded_array = np.zeros((w + padding * 2, h + padding * 2, c))
        
    # Copy all content
    for i in range(w):
        for j in range (h):
            padded_array[i + padding][j + padding] = image_array[i][j]
            
    # For left and right
    for i in range(w):
        for j in range(padding):
            # Left
            padded_array[padding + i][padding - j] = image_array[i][j + 1]
            # Right
            padded_array[padding + i][h + padding + j] = image_array[i][h - j - 1]
            
    # For up and down
    for i in range(h):
        for j in range(padding):
            # Up
            padded_array[padding - j][padding + i] = image_array[j + 1][i]
            # Down
            padded_array[w + padding + j][padding + i] = image_array[w - j - 1][i]
            
    # For diagonal
    for i in range(padding):
        for j in range(padding):
            # Up Left
            padded_array[i][j] = image_array[padding - i][padding - j]
            # Up Right
            padded_array[i][h + padding + j] = image_array[padding - i][h - j - 1]
            # Downn Left
            padded_array[w + padding + i][j] = image_array[w - i - 1][padding - j]
            # Down Right
            padded_array[w + padding + i][h + padding + j] = image_array[w - i - 1][h - j - 1]
    
    return padded_array

def from_normilzed_to_image_array(normalized_array):
    return np.uint8(normalized_array * 255)

def IoU(real, pred):
    # It's expected that real and pred are in range [0,1]
    # Get comparisson matrix (cmp)
    # This matrix contains the following information:
    # 0 -> 0 in mask and 0 in prediction
    # 1 -> 0 is mask and 1 in prediction - False positive
    # 2 -> 1 in mask and 0 in prediction - False negative
    # 3 -> 1 in mask and 1 in prediction - True Positive
    pred = (pred > 0) * 1
    cmp = 2 * real + pred
    cant = []
    for i in range(4):
        cant.append(np.sum((cmp == i) * 1))
    intersection = cant[3]
    union = np.sum(cant[1:])
    return intersection / union

def create_folder_if_not_exists(folder_name):
    if os.path.isdir(folder_name) == False:
        os.makedirs(folder_name)