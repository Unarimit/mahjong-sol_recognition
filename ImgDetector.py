import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
PATH = 'pic/mahjong'


class MahjongDetector:
    def __init__(self, img_height, white_center):
        self.white_center = white_center
        self.height = img_height
        self.output_imgs = []
        detector = cv2.SIFT_create(nfeatures=20)  # can adjust
        computer = cv2.SIFT_create()
        # detector = cv2.ORB_create()
        files = os.listdir(PATH)
        self.pfeatures = dict()
        self.cfeatures = dict()
        for file in files:
            img = cv2.imread(PATH + '/' + file)
            ratio = min(1, img_height / max(len(img[0]), len(img)))
            img = cv2.resize(img, None, fx=ratio, fy=ratio)
            self.cfeatures[file.split('.')[0]] = self._compute_hist(img)

            img = self._pre_process_img(img)
            temp = detector.detect(img)
            self.pfeatures[file.split('.')[0]] = computer.compute(img, temp, None)

            if True:
                self._draw_keypoints(img, temp)
        self.draw_down('source')
        self.detector = detector
        self.computer = computer
        self.debug_i = 0

    def detect(self, img):
        # debug
        ''''''
        self.debug_i += 1
        if len(img) < 5 or len(img[0]) < 5:
            cv2.imshow('what', img)
            cv2.waitKey(0)
            return ''

        if self.debug_i == 5:
            print('koko')

        # hist filter
        max_value = 0
        max_name = ''
        poss = self._detect_by_hist(img)
        if self.debug_i != 0:
            print(str(self.debug_i) + '\t hist: ' + str(poss))
        if len(poss) == 0:
            poss = self.pfeatures.keys()
        elif poss[0] == '5z':
            max_name = '5z'
            max_value = 5

        img = self._pre_process_img(img)
        fp = self.detector.detect(img, None)
        if len(fp) == 0:
            return '5z'
        des = self.computer.compute(img, fp)  # des 可能为none
        # match = cv2.BFMatcher(cv2.NORM_L2, False)
        self._draw_keypoints(img, fp)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        for key in poss:
            name = key
            descriptors = self.pfeatures[key]
            if name == '5z':
                continue
            # matched = match.match(descriptors[1], des[1])
            matches = flann.knnMatch(des[1], descriptors[1], k=2)
            useful_len = 0
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.6 * n.distance:  # default is 0.4
                    useful_len += 1
            matched_points = useful_len / len(matches)
            if matched_points > max_value:
                max_value = matched_points
                max_name = name
        return max_name

    def draw_down(self, frame_name='draw_down'):
        for i in range(len(self.output_imgs)):
            self.output_imgs[i] = cv2.resize(self.output_imgs[i], (self.height, self.height))
        cv2.imshow(frame_name, cv2.hconcat(self.output_imgs))
        self.output_imgs = []

    def _detect_by_hist(self, img):
        '''

        :param img:
        :return: [name, ...]
        '''
        cur_hist = self._compute_hist(img)
        possible = []
        for key in self.cfeatures:
            feature = self.cfeatures[key]
            sum = cv2.compareHist(np.array(cur_hist), np.array(feature), cv2.HISTCMP_BHATTACHARYYA)
            if sum < 0.3:
                possible.append([key, sum])
        possible.sort(key=lambda x: x[1])
        if self.debug_i == -1:
            '''test 直方图
            '''
            cv2.imshow('test', img)
            self._draw_hist(cur_hist, 'aim')
            self._draw_hist(self.cfeatures['1p'], '1p')
            '''
            self._draw_hist(self.cfeatures[possible[0][0]], possible[0][0])
            self._draw_hist(self.cfeatures[possible[1][0]], possible[1][0])
            self._draw_hist(self.cfeatures[possible[2][0]], possible[2][0])
            '''
            plt.show()
            cv2.waitKey(0)
        result = []
        for i in range(min(10, len(possible))):
            result.append(possible[i][0])

        return result

    def _compute_hist(self, img):
        # hist is ndarray(256, 1)
        # b_hist = cv2.calcHist(img, [0], None, [256], [0, 255])
        # 使用函数计算得到的直方图因为数据集和识别结果差异大（光照），所以不能用
        # 手动绘制特征直方图
        b_hist = np.zeros(shape=(256, 1), dtype=np.float32)
        g_hist = np.zeros(shape=(256, 1), dtype=np.float32)
        r_hist = np.zeros(shape=(256, 1), dtype=np.float32)

        # count black red green BGR!!
        thresh = cv2.inRange(img, (210, 210, 210), (230, 230, 230))
        black = 0
        for i in range(1, 9):
            black += cv2.inRange(img, (i*10, i*10, i*10), (i*10+10, i*10+10, i*10+10))
        red = cv2.inRange(img, (25, 25, 140), (80, 80, 180))
        green = cv2.inRange(img, (55, 95, 15), (65, 115, 35))
        white_cnt = np.count_nonzero(thresh == 255)
        black_cnt = np.count_nonzero(black == 255)
        red_cnt = np.count_nonzero(red == 255)
        green_cnt = np.count_nonzero(green == 255)
        r_hist[0] = red_cnt
        g_hist[60] = green_cnt
        b_hist[180] = black_cnt
        b_hist[250] = max(white_cnt / (len(img) * len(img[0])) * (red_cnt + green_cnt + black_cnt) / 3, 10)  # 小心白板
        tu = (r_hist, g_hist, b_hist)
        return tu

    def _pre_process_img(self, img):
        '''

        :param img:  a bgr img
        :return:
        '''
        ''' useless
        img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img0, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return thresh
        '''
        return img

    # test method
    def _draw_keypoints(self, img, kp):
        img0 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
        self.output_imgs.append(img0)

    def _draw_hist(self, hists, name):
        '''显示三个通道的颜色直方图'''
        plt.figure(num=name)
        plt.plot(hists[0], label='R', color='red')
        plt.plot(hists[1], label='G', color='green')
        plt.plot(hists[2], label='B', color='blue')
        plt.legend(loc='best')
        plt.xlim([0, 256])
        plt.draw()


if __name__ == '__main__':
    detector = MahjongDetector(50, 220)
    print(detector.detect(cv2.imread(PATH+'/1z.png')))
