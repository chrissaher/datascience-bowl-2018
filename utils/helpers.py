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

def sum_mat(matrix):
    H, W = matrix.shape

    acum = np.zeros(matrix.shape)

    for i in range(H):
        if i == 0:
            acum[i][0] = matrix[0][0]
        else:
            acum[i][0] = acum[i - 1][0] + matrix[i][0]

    for i in range(W):
        if i == 0:
            acum[0][i] = matrix[0][0]
        else:
            acum[0][i] = acum[0][i - 1] + matrix[0][i]

    for i in range(2, H):
        for j in range(2, W):
            acum[i][j] = acum[i - 1][j] + acum[i][j - 1] - acum[i - 1][j - 1] + matrix[i][j]
    return acum

def get_pos_neg_examples(mask, padding):
    w, h = mask.shape
    mask_padded = apply_padding(mask, padding)
    pos = 0
    neg = 0
    pos_positions = []
    neg_positions = []

    acum = sum_mat(mask_padded)

    for x in range(w):
        for y in range(h):
            if mask_padded[x + padding + 1][y + padding + 1] >= 1:
                pos += 1
                pos_positions.append((x,y))
            else:

                nx = x + 2 * padding
                ny = y + 2 * padding

                value = acum[nx][ny] + acum[x][y]
                value -= (acum[nx - x][ny] + acum[nx][ny - y])
                #sample = mask_padded[x: nx ,y : ny]

                if value > 0:
                    neg += 0
                    neg_positions.append((x,y))

    return pos, neg, pos_positions, neg_positions

def get_train_and_labels_from_image_N_classes(image, mask, num_classes = 2, rate = 1, padding = 50, verbose = False):
    w, h, _ = image.shape
    all_zeros_added = False
    image_padded = apply_padding(image, padding)
    mask_padded = apply_padding(mask, padding)
    training_examples = []
    label_examples = []
    ident = np.identity(max(num_classes, 2))

    pos,neg, pos_positions, neg_positions = get_pos_neg_examples(mask, padding)

    real_rate = pos * rate
    real_rate = min(real_rate, neg)

    newpos = 0
    newneg = 1

    for (x,y) in pos_positions:
        nx = x + 2 * padding + 1
        ny = y + 2 * padding + 1
        img_sample = image_padded[x: nx, y : ny, :]
        training_examples.append(img_sample)
        out_class = int(mask_padded[x + padding + 1][y + padding + 1])
        label_examples.append(ident[out_class])
        newpos += 1

    random.shuffle(neg_positions)
    for i in range(real_rate):
        (x,y) = neg_positions[i]
        if random.randint(0, neg) < real_rate:
            nx = x + 2 * padding + 1
            ny = y + 2 * padding + 1
            img_sample = image_padded[x: nx, y : ny, :]
            training_examples.append(img_sample)
            label_examples.append(ident[0])
            newneg += 1

    # Add extra sample for all zeros
    zero_sample = np.zeros((2 * padding + 1,2 * padding + 1,3))
    training_examples.append(zero_sample)
    label_examples.append(ident[0])

    train = np.array(training_examples)
    labels = np.array(label_examples)

    if verbose:
        print("---------------------")
        print("Positive examples taken : ", newpos)
        print("Negative examples taken : ", newneg)
        print("---------------------")

    return train, labels

def get_train_test(image, mask, num_classes = 2, rate = 1, padding = 50, verbose = False):
    train, labels = get_train_and_labels_from_image_N_classes(image, mask, num_classes, rate, padding, verbose)
    if num_classes == 1:
        train = np.argmax(train, axis = -1)
        labels = np.argmax(labels, axis = -1)
    return train, labels

def get_train_and_labels_from_image(image, mask, rate = 1, padding = 50, verbose = False):
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

    pos,neg = get_pos_neg_examples(mask, padding)

    real_rate = pos * rate

    newpos = 0
    newneg = 1

    for x in range(w):
        for y in range(h):
            nx = x + 2 * padding + 1
            ny = y + 2 * padding + 1
            sample = mask_padded[x: nx ,y : ny]

            if np.sum(sample) > 0:
                if (mask_padded[x + padding + 1][y + padding + 1] == 0):
                    if random.randint(0, neg) < real_rate:
                        img_sample = image_padded[x: nx, y : ny, :]
                        training_examples.append(img_sample)
                        label_examples.append(np.array([0,1]))
                        newneg += 1
                else:
                    img_sample = image_padded[x: nx, y : ny, :]
                    training_examples.append(img_sample)
                    label_examples.append(np.array([1,0]))
                    newpos += 1
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
        print("Positive examples taken : ", newpos)
        print("Negative examples taken : ", newneg)
        print("---------------------")

    return train, labels

def get_train_and_labels_from_image_1_class(image, mask, rate = 1, padding = 50, verbose = False):
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

    pos,neg = get_pos_neg_examples(mask, padding)

    real_rate = pos * rate

    newpos = 0
    newneg = 1

    for x in range(w):
        for y in range(h):
            nx = x + 2 * padding + 1
            ny = y + 2 * padding + 1
            sample = mask_padded[x: nx ,y : ny]

            if np.sum(sample) > 0:
                if (mask_padded[x + padding + 1][y + padding + 1] == 0):
                    if random.randint(0, neg) < real_rate:
                        img_sample = image_padded[x: nx, y : ny, :]
                        training_examples.append(img_sample)
                        label_examples.append(0)
                        newneg += 1
                else:
                    img_sample = image_padded[x: nx, y : ny, :]
                    training_examples.append(img_sample)
                    label_examples.append(1)
                    newpos += 1
            else:
                if all_zeros_added == False:
                    all_zeros_added = True
                    img_sample = image_padded[x: nx, y : ny, :]
                    training_examples.append(img_sample)
                    label_examples.append(0)

    train = np.array(training_examples)
    labels = np.array(label_examples)

    if verbose:
        print("---------------------")
        print("Positive examples taken : ", newpos)
        print("Negative examples taken : ", newneg)
        print("---------------------")

    return train, labels

def get_train_and_labels_from_image_2_classes(image, mask, rate = 1, padding = 50, verbose = False):
    return get_train_and_labels_from_image(image, mask, rate, padding, verbose)

def get_train_and_labels_from_image_3_classes(image, mask, rate = 1, padding = 50, verbose = False):
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

    pos,neg = get_pos_neg_examples(mask, padding)

    real_rate = pos * rate

    newpos = 0
    newneg = 1

    for x in range(w):
        for y in range(h):
            nx = x + 2 * padding + 1
            ny = y + 2 * padding + 1
            sample = mask_padded[x: nx ,y : ny]

            if np.sum(sample) > 0:
                if (mask_padded[x + padding + 1][y + padding + 1] == 0):
                    if random.randint(0, neg) < real_rate:
                        img_sample = image_padded[x: nx, y : ny, :]
                        training_examples.append(img_sample)
                        label_examples.append(np.array([1,0,0]))
                        newneg += 1
                else:
                    img_sample = image_padded[x: nx, y : ny, :]
                    neighbors = np.sum(mask_padded[x + padding: x + padding + 2][y + padding: y + padding + 2])
                    if neighbors == 9:
                        label_examples.append(np.array([0,1,0])) # Is not border
                    else: label_examples.append(np.array([0,0,1])) # Is border
                    training_examples.append(img_sample)

                    newpos += 1
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
        print("Positive examples taken : ", newpos)
        print("Negative examples taken : ", newneg)
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

def from_2_mask_to_3_mask(mask, border = 1):
    w, h = mask.shape
    new_mask = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            neighbors = np.sum(mask[i - 1: i + 2, j - 1: j + 2])
            if neighbors == 9:
                new_mask[i][j] = 1
            elif neighbors > 0:
                new_mask[i][j] = 3

    dx = [-1, 0, 1, 1, 1, 0,-1,-1]
    dy = [-1,-1,-1, 0, 1, 1, 1, 0]

    level = 3
    for t in range(border):
        for i in range(w):
            for j in range(h):
                if new_mask[i][j] == level:
                    for v in range(8):
                        nx = i + dx[v]
                        ny = j + dy[v]
                        if nx > 0 and ny > 0 and nx < w and ny < h and new_mask[nx][ny] == 0:
                            new_mask[nx][ny] = level + 1
        level += 1

    for i in range(w):
        for j in range(h):
            if new_mask[i][j] > 3:
                new_mask[i][j] = 2
            if new_mask[i][j] == 3:
                new_mask[i][j] = 1

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
