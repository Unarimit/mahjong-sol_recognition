import os
import numpy as np
import cv2
from ImgDetector import MahjongDetector


def get_mahjongs_position(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = gray
    # 白色区域为255， 将图像二值化 +- 10 误差
    thresh = np.zeros_like(gray)
    for i in range(19, 25, 2):
        thresh += cv2.inRange(img, (i*10, i*10, i*10), (i*10+20, i*10+20, i*10+20))
    cv2.imshow("123", thresh)

    # 画轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 得到轮廓信息
    boxes = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 300:  # 小于300像素面积不保存
            continue
        if hierarchy[0][i][3] != -1:
            continue
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        boxes.append(box)
    return boxes

# make data to do machine learning
import uuid
def save_img(the_class : str, img):
    data_src = 'data'
    if not os.path.exists(data_src):
        os.mkdir(data_src)
    dest = data_src+'/'+the_class
    if not os.path.exists(dest):
        os.mkdir(dest)
    cv2.imwrite(dest+'/'+str(uuid.uuid1())+'.png', img)


if __name__ == '__main__':
    img = cv2.imread('aim/30.png')
    boxes = get_mahjongs_position(img)
    # TODO: 第一次取所有麻将均值
    detector = MahjongDetector(max(max(boxes[0][0][1], boxes[0][1][1], boxes[0][2][1], boxes[0][3][1]) - min(boxes[0][0][1], boxes[0][1][1], boxes[0][2][1], boxes[0][3][1]),
                                   max(boxes[0][0][0], boxes[0][1][0], boxes[0][2][0], boxes[0][1][0]) - min(boxes[0][0][0], boxes[0][1][0], boxes[0][2][0], boxes[0][1][0]))
                               , 220)
    result = []

    # 测试框的准不准
    '''
    for box in boxes:
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    if len(boxes) != 0:
        cv2.imshow('convexHull', img)
    cv2.waitKey(0)
    '''


    padding = 1
    for i, box in enumerate(boxes):
        max_x = max(box[0][1], box[1][1], box[2][1], box[3][1])
        min_x = min(box[0][1], box[1][1], box[2][1], box[3][1])
        min_y = min(box[0][0], box[1][0], box[2][0], box[1][0])
        max_y = max(box[0][0], box[1][0], box[2][0], box[1][0])
        text = detector.detect(img[(min_x+padding):(max_x-padding), (min_y+padding):(max_y-padding)])
        if text == '':
            text = 'no'
        # save_img(text, img[(min_x+padding):(max_x-padding), (min_y+padding):(max_y-padding)])
        result.append([[min_y, min_x], text])

    detector.draw_down('123')
    for re in result:
        cv2.putText(img, re[1], re[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))
    cv2.imshow('convexHull', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



