import cv2
import glob
import numpy as np

pos_imgs = glob.glob('dataset/train/pos/*.jpg')
neg_imgs = glob.glob('dataset/train/neg/*.jpg')


def get_train_data():
    return pos_imgs[:int(0.7 * len(pos_imgs))], neg_imgs[:int(0.7 * len(neg_imgs))]


def get_test_data():
    return pos_imgs[int(0.7 * len(pos_imgs)):], neg_imgs[int(0.7 * len(neg_imgs)):]


def get_predict_data():
    pos_imgs = glob.glob('dataset/test/pos/*.jpg')
    neg_imgs = glob.glob('dataset/test/neg/*.jpg')

    return pos_imgs, neg_imgs


def get_imgs_feature(imgs, hog, hog_params):
    data = []
    pos, neg = imgs

    for img in pos:
        data.append(np.append(get_img_feature(img, hog, hog_params), 1))

    for img in neg:
        data.append(np.append(get_img_feature(img, hog, hog_params), 0))

    data = np.array(data)

    labels = data[:, -1]
    features = data[:, :-1]

    return features, labels


def get_img_feature(img, hog, hog_params):
    win_stride, padding, locations = hog_params

    if type(img) == str:
        img = cv2.imread(img)

    feature = hog.compute(img, win_stride, padding, locations)
    feature = feature.reshape(-1)

    return feature


def visualize(image, rects):

    image = cv2.imread(image)

    for rect in rects:
        image = cv2.rectangle(image, (rect[0], rect[1]),
                              (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0))

    cv2.imshow('image', image)
    cv2.waitKey(0)
