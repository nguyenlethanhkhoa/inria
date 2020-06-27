import cv2
import dataset
import numpy as np


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def sliding_window(image, ratio, rects, hog, hog_params, model):
    img_width = image.shape[1]
    img_height = image.shape[0]

    window_width = 96
    window_height = 160

    window_x = 0
    window_y = 0

    result = False
    while window_y + window_height <= img_height:
        while window_x + window_width <= img_width:
            img = image[
                window_y:int(window_y + window_height),
                window_x:int(window_x + window_width)
            ]
            feature = dataset.get_img_feature(img, hog, hog_params)
            if int(model.predict(np.array(feature.reshape(-1)).reshape(1, -1))) == 1:
                rect_x = window_x / ratio
                rect_y = window_y / ratio
                rect_width = window_width / ratio
                rect_height = window_height / ratio
                rects.append([rect_x, rect_y, rect_width, rect_height])

            # cv2.imwrite('sliding/'+str(img_width)+'x'+str(img_height)+'_'+str(window_x)+'x'+str(window_y)+'.jpg', img)

            result = True
            window_x = int(window_x + 4)

        window_x = 0
        window_y += int(window_y + 4)

    return result


def detect_person(img_path, hog, hog_params, model):
    is_resize = True
    has_image = False
    ratio = 1
    rects = []
    image = cv2.imread(img_path)
    while is_resize:
        if has_image:
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            ratio = ratio * 0.5

        is_resize = sliding_window(image, ratio, rects, hog, hog_params, model)
        has_image = True

    return rects


def histogram_of_gradient():
    win_size = (64, 128)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    deriv_aperture = 1
    win_sigma = 4.
    histogram_norm_type = 0
    l2_hys_threshold = 2.0000000000000001e-01
    gamma_correction = 0
    n_levels = 64
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins, deriv_aperture,
                            win_sigma, histogram_norm_type, l2_hys_threshold, gamma_correction, n_levels)

    return hog


def precision_and_recall_evaluate(results, labels):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(len(results)):
        if results[i] == labels[i] and int(results[i]) == 1:
            true_pos += 1

        if results[i] == labels[i] and int(results[i]) == 0:
            true_neg += 1

        if results[i] != labels[i] and int(results[i]) == 1:
            false_pos += 1

        if results[i] != labels[i] and int(results[i]) == 0:
            false_neg += 1

    print('accuracy: ' + str((true_pos + true_neg)/len(results)))
    print('precision: ' + str(true_pos/(true_pos + false_pos)))
    print('recall: ' + str(true_pos/(true_pos + false_neg)))
