import util
import dataset
import numpy as np

from PIL import Image
from sklearn import svm

win_stride = (8, 8)
padding = (8, 8)
locations = ((10, 20),)

hog = util.histogram_of_gradient()
hog_params = (win_stride, padding, locations)

train_data = dataset.get_train_data()
features, labels = dataset.get_imgs_feature(train_data, hog, hog_params)

linear_svm_model = svm.LinearSVC()
linear_svm_model.fit(features, labels)

test_data = dataset.get_test_data()
features, labels = dataset.get_imgs_feature(test_data, hog, hog_params)
results = linear_svm_model.predict(features)

util.precision_and_recall_evaluate(results, labels)

pos, neg = dataset.get_predict_data()
for i in range(5):
    rects = util.detect_person(pos[i], hog, hog_params, linear_svm_model)
    rects = util.non_max_suppression_fast(np.array(rects), 0.5)
    print(pos[i])
    print(rects)
    dataset.visualize(pos[i], rects)


# # imgs = []
# # img_paths = []
# rects = util.detect_person(
#     'dataset/test/pos/crop_000001.jpg', hog, hog_params, linear_svm_model)
# print(np.array(rects))
# rect = util.non_max_suppression_fast(np.array(rects), 0.5)
# print(rect)
# # images = glob.glob('inria/sliding/*.jpg')
# # for image in images:
# #     feature = hog.compute(cv2.imread(image),winStride,padding,locations)
# #     feature = feature.reshape(-1)
# #     imgs.append(feature)
# #     img_paths.append(image)

# # results = linear_svm_model.predict(np.array(imgs))
# # for i in range(len(results)):
# #     print(img_paths[i] + ': ' + str(int(results[i])))
