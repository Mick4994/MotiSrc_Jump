#标定命令 python3 cal.py --image_size 1280x720 --corner 11x8 --square 30

import os
import cv2 as cv
import numpy as np
import glob
import xml.etree.ElementTree as ET
import argparse

class CameraCalibrator(object):
    def __init__(self, image_size:tuple):
        super(CameraCalibrator, self).__init__()
        self.image_size = image_size
        self.matrix = np.zeros((3, 3), np.float32)
        self.new_camera_matrix = np.zeros((3, 3), np.float32)
        self.dist = np.zeros((1, 5))
        self.roi = np.zeros(4, np.int32)

    def save_params(self, save_path='camera_params.xml'):
        root = ET.Element('root')
        tree = ET.ElementTree(root)

        mat_node = ET.Element('camera_matrix')
        root.append(mat_node)
        for i, elem in enumerate(self.matrix.flatten()):
            child = ET.Element('data{}'.format(i))
            child.text = str(elem)
            mat_node.append(child)

        dist_node = ET.Element('camera_distortion')
        root.append(dist_node)
        for i, elem in enumerate(self.dist.flatten()):
            child = ET.Element('data{}'.format(i))
            child.text = str(elem)
            dist_node.append(child)

        tree.write(save_path, 'UTF-8')
        print("Saved params in {}.".format(save_path))


    def cal_real_corner(self, corner_height, corner_width, square_size):
        obj_corner = np.zeros([corner_height * corner_width, 3], np.float32)
        obj_corner[:, :2] = np.mgrid[0:corner_height, 0:corner_width].T.reshape(-1, 2)  # (w*h)*2
        return obj_corner * square_size

    def calibration(self, corner_height:int, corner_width:int, square_size:float):
        file_names = glob.glob('./chess_img/*.JPG') + glob.glob('./chess_img/*.jpg') + glob.glob('./chess_img/*.png')
        objs_corner = []
        imgs_corner = []
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        obj_corner = self.cal_real_corner(corner_height, corner_width, square_size)
        for file_name in file_names:
            # read image
            chess_img = cv.imread(file_name)
            assert (chess_img.shape[0] == self.image_size[1] and chess_img.shape[1] == self.image_size[0]), \
                "Image size does not match the given value {}.".format(self.image_size)
            # to gray
            gray = cv.cvtColor(chess_img, cv.COLOR_BGR2GRAY)
            # find chessboard corners
            ret, img_corners = cv.findChessboardCorners(gray, (corner_height, corner_width))

            # append to img_corners
            if ret:
                objs_corner.append(obj_corner)
                img_corners = cv.cornerSubPix(gray, img_corners, winSize=(square_size//2, square_size//2),
                                              zeroZone=(-1, -1), criteria=criteria)
                imgs_corner.append(img_corners)
            else:
                print("Fail to find corners in {}.".format(file_name))

        # calibration
        ret, self.matrix, self.dist, rvecs, tveces = cv.calibrateCamera(objs_corner, imgs_corner, self.image_size, None, None)
        print(self.matrix)
        return ret


    def rectify_image(self, img):
        if not isinstance(img, np.ndarray):
            AssertionError("Image type '{}' is not numpy.ndarray.".format(type(img)))
        dst = cv.undistort(img, self.matrix, self.dist, self.new_camera_matrix)
        x, y, w, h = self.roi
        dst = dst[y:y + h, x:x + w]
        dst = cv.resize(dst, (self.image_size[0], self.image_size[1]))
        return dst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type= str, help='width*height of image')
    parser.add_argument('--square', type=int, help='size of chessboard square, by mm')
    parser.add_argument('--corner', type=str, help='width*height of chessboard corner')
    args = parser.parse_args()
    calibrator = None

    try:

        image_size = tuple(int(i) for i in args.image_size.split('x'))
        print(image_size)
        calibrator = CameraCalibrator(image_size)
    except:
        print("Invalid/Missing parameter: --image_size. Sample: \n\n"
              "    --image_size 1920*1080\n")
        exit(-1)

    if not args.corner or not args.square:
        print("Missing parameters of corner/square. Using: \n\n"
                "    --corner <width>x<height>\n\n"
                "    --square <length of square>\n")
        exit(-1)
    corner = tuple(int(i) for i in args.corner.split('x'))
    if calibrator.calibration(corner[1], corner[0], args.square):
        calibrator.save_params()
    else:
        print("Calibration failed.")

