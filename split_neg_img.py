import os
import cv2
import glob
import math


imgs = glob.glob('inria/test/pos/*.jpg')
for img in imgs:
    image = cv2.imread(img)
    filename_w_ext = os.path.basename(img)
    filename, file_extension = os.path.splitext(filename_w_ext)

    for x in range(math.floor(image.shape[0]/160)):
        for y in range(math.floor(image.shape[1]/96)):
            width = (y + 1) * 96
            height = (x + 1) * 160
            cv2.imwrite('inria/train/_neg/' + filename + '_' + str(x) + '_' + str(y) + file_extension, image[(height - 160):height, (width - 96):width])
            print(filename + '_' + str(x) + '_' + str(y) + file_extension)



