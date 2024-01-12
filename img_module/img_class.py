import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

flag = cv.COLOR_BGR2RGB


class ImageTasks:
    def __init__(self, img_src):
        self.img_src = img_src
        self.img = cv.imread(self.img_src)

    def reread(self):
        self.img = cv.imread(self.img_src)

    def show(self):
        plt.imshow(cv.cvtColor(self.img, flag))

    # task 1
    def orb_features(self):
        orb = cv.ORB_create()
        kp, des = orb.detectAndCompute(self.img, None)
        res = cv.drawKeypoints(self.img, kp, None, flags=0)

        plt.imshow(cv.cvtColor(res, flag))

    # task 2
    def sift_features(self):
        sift = cv.SIFT_create()
        kp = sift.detect(self.img, None)
        res = cv.drawKeypoints(self.img, kp, None, flags=0)

        plt.imshow(cv.cvtColor(res, flag))

    # task 3
    def canny_edges(self):
        res = cv.Canny(self.img, 100, 200)
        plt.imshow(cv.cvtColor(res, flag))

    # task 4
    def convert_to_hsv(self):
        res = cv.cvtColor(self.img, cv.COLOR_RGB2HSV)
        plt.imshow(cv.cvtColor(res, flag))

    # task 5
    def convert_to_grayscale(self):
        res = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        plt.imshow(cv.cvtColor(res, flag))

    # task 6
    def horizontal_flip(self):
        res = cv.flip(self.img, 1)
        plt.imshow(cv.cvtColor(res, flag))

    # task 7
    def vertical_flip(self):
        res = cv.flip(self.img, 0)
        plt.imshow(cv.cvtColor(res, flag))

    # task 8
    def rotate_45_degree(self):
        (h, w, d) = self.img.shape
        center = (w // 2, h // 2)
        res = cv.warpAffine(self.img, cv.getRotationMatrix2D(center, 45, 1.0), (w, h))
        plt.imshow(cv.cvtColor(res, flag))

    # task 9
    def rotate_30_degree_around_point(self, point=(0, 0)):
        (h, w, d) = self.img.shape
        res = cv.warpAffine(self.img, cv.getRotationMatrix2D(point, 30, 1.0), (w, h))
        plt.imshow(cv.cvtColor(res, flag))

    # task 10
    def shift_10_pixels(self):
        (h, w, d) = self.img.shape
        cutted_matrix = np.float32([[1, 0, 10], [0, 1, 0]])
        res = cv.warpAffine(self.img, cutted_matrix, (w, h))
        plt.imshow(cv.cvtColor(res, flag))

    # task 11
    def change_brightness(self, value=0):
        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
        final_hsv = cv.merge((h, s, v))
        res = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
        plt.imshow(cv.cvtColor(res, flag))

    # task 12
    def change_contrast(self, value=0):
        res = cv.convertScaleAbs(self.img, alpha=value)
        plt.imshow(cv.cvtColor(res, flag))

    # task 13
    def change_gamma(self, value=0):
        matrix2d = [((i / 255) ** (1 / value)) * 255 for i in range(256)]
        res = cv.LUT(self.img, np.array(matrix2d, np.uint8))
        plt.imshow(cv.cvtColor(res, flag))

    # task 14
    def histogram_equalize(self):
        res = cv.equalizeHist(cv.cvtColor(self.img, cv.COLOR_BGR2GRAY))
        plt.imshow(cv.cvtColor(res, flag))

    # task 15
    def white_balance_make_warmer(self, value=0):
        blue, green, red = cv.split(self.img)
        blue = np.clip((blue.astype(np.int16) - value), 0, 255).astype(np.uint8)
        red = np.clip((red.astype(np.int16) + value), 0, 255).astype(np.uint8)
        res = cv.merge((blue, green, red))
        plt.imshow(cv.cvtColor(res, flag))

    # task 16
    def white_balance_make_colder(self, value=0):
        blue, green, red = cv.split(self.img)
        blue = np.clip((blue.astype(np.int16) + value), 0, 255).astype(np.uint8)
        red = np.clip((red.astype(np.int16) - value), 0, 255).astype(np.uint8)
        res = cv.merge((blue, green, red))
        plt.imshow(cv.cvtColor(res, flag))

    # task 17
    def change_palette(self, value=(0.0, 0.0, 0.0)):
        blue, green, red = cv.split(self.img)
        res = cv.merge((
            np.clip((blue.astype(np.int16) + value[2]), 0, 255).astype(np.uint8),
            np.clip((green.astype(np.int16) + value[1]), 0, 255).astype(np.uint8),
            np.clip((red.astype(np.int16) + value[0]), 0, 255).astype(np.uint8)
        ))
        plt.imshow(cv.cvtColor(res, flag))

    # task 18
    def binarize_image(self):
        _, res = cv.threshold(self.img, 127, 255, 0)
        plt.imshow(cv.cvtColor(res, flag))

    # task 19
    def found_contours(self):
        _, img_bin = cv.threshold(cv.cvtColor(self.img, cv.COLOR_BGR2GRAY), 127, 255, 0)
        contours, hierarchy = cv.findContours(img_bin.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        res = cv.drawContours(self.img, contours, -1, (0, 0, 255), 2, cv.LINE_AA, hierarchy, 1)
        plt.imshow(cv.cvtColor(res, flag))
        self.reread()

    # task 20
    def found_contour_by_sobel(self):
        res = cv.filter2D(self.img, -1, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
        plt.imshow(cv.cvtColor(res, flag))

    # task 21
    def blur(self):
        res = cv.blur(self.img, ksize=(10, 10))
        plt.imshow(cv.cvtColor(res, flag))

    # task 22
    def fourier_filter_fast_freq(self):
        dft_shift = np.fft.fftshift(np.fft.fft2(self.img, axes=(0, 1)))

        mask = np.zeros_like(self.img)
        y = mask.shape[0] // 2
        x = mask.shape[1] // 2

        cv.circle(mask, (x, y), 16, (255, 255, 255), -1)[0]
        mask = cv.GaussianBlur((255 - mask), (19, 19), 0)

        filtered = np.fft.ifft2(np.fft.ifftshift(np.multiply(dft_shift, mask) / 255), axes=(0, 1))
        res = np.abs(3 * filtered).clip(0, 255).astype(np.uint8)

        plt.imshow(cv.cvtColor(res, flag))

    # task 23
    def fourier_filter_low_freq(self):
        dft_shift = np.fft.fftshift(np.fft.fft2(self.img, axes=(0, 1)))

        mask = np.zeros_like(self.img)
        y = mask.shape[0] // 2
        x = mask.shape[1] // 2

        cv.circle(mask, (x, y), 64, (255, 255, 255), -1)[0]
        mask = cv.GaussianBlur(mask, (19, 19), 0)

        filtered = np.fft.ifft2(np.fft.ifftshift(np.multiply(dft_shift, mask) / 255), axes=(0, 1))
        res = np.abs(filtered).clip(0, 255).astype(np.uint8)

        plt.imshow(cv.cvtColor(res, flag))

    # task 24
    def erose_image(self):
        res = cv.erode(self.img, np.ones((5, 5), 'uint8'), iterations=1)
        plt.imshow(cv.cvtColor(res, flag))

    # task 25
    def dilate_image(self):
        res = cv.dilate(self.img, np.ones((5, 5), 'uint8'), iterations=1)
        plt.imshow(cv.cvtColor(res, flag))
