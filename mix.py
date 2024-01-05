import serial
import threading
import time
import numpy as np
import cv2
import traceback
from ultralytics import YOLO
from math import sin, cos, radians

# 截取的边长转换成和像素边长的比例关系，5m = 1000pixel
border_m = 5
border_p = 1000
init_x = 46
init_y = 68
init_z = 39
init_p = 48

cut_bound = 100
CAMERA_INDEX = 1


def imgSplit():
    yellow_hsv_min = np.array([14, 43, 46], dtype=np.uint8)
    yellow_hsv_max = np.array([20, 255, 255], dtype=np.uint8)
    img = cv2.imread('lands_img/1.png')
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    inrange_img = cv2.inRange(hsv_img, yellow_hsv_min, yellow_hsv_max)
    inrange_img = cv2.cvtColor(inrange_img, cv2.COLOR_GRAY2BGR)
    merge = np.hstack((img, inrange_img))
    cv2.imshow('test', merge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class VisionSolution:
    def __init__(self) -> None:
        # self.SCREEN_W = 1280
        # self.SCREEN_H = 720
        self.SCREEN_W = 1920
        self.SCREEN_H = 1080
        self.camera_x = 0
        self.camera_y = -0.55
        self.camera_z = -0.5
        self.pitch = 20
        test_cloud = []
        x = -2
        while x < 2:
            z = -0.2
            while z < 0.2:
                test_cloud.append([x, 0, z])
                z += 0.002
            # z = -0.2
            x += 0.5
        self.test_cloud = test_cloud

        map_cloud = []
        x = -1.5
        while x < 1.5:
            z = -1
            map_cloud.append([x, 0, z])
            map_cloud.append([x, 0, z + 2])
            map_cloud.append([x + 0.01, 0, z + 2])
            map_cloud.append([x + 0.01, 0, z])
            # z = -0.2
            x += 0.01
        self.map_cloud = map_cloud

    def projectMethod(self, camera_img):
        test_cloud = np.array(self.test_cloud, dtype=np.float32)
        map_cloud = np.array(self.map_cloud, dtype=np.float32)

        camera_position = np.array([self.camera_x, self.camera_y, self.camera_z], dtype=np.float32)
        camera_euler_angles = np.array([radians(self.pitch), radians(0), radians(0)], dtype=np.float32)
        img = np.zeros((self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8)
        img_3d = camera_img.copy()

        # 内参
        # K = np.array([[669.83598624, 0, 642.06902362],
        #               [0, 669.94827046, 363.98259922],
        #               [0,  0, 1]], dtype=np.float32)
        # K = np.array([[671.15035972, 0, 638.50053821],
        #               [0, 670.68343126, 362.61330926],
        #               [0, 0, 1]], dtype=np.float32)
        # K = np.array([[670, 0, 640],
        #             [0, 670, 360],
        #             [0,   0,   1]], dtype=np.float32)
        # K = np.array([[1.67634347e+03, 0.00000000e+00, 9.49802868e+02],
        #               [0.00000000e+00, 1.67448209e+03, 5.38285316e+02],
        #               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
        K = np.array([[1.49217024e+03, 0.00000000e+00, 9.55260745e+02],
                         [0.00000000e+00, 1.49520977e+03, 5.39343430e+02],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
        # 计算被旋转后的平移向量
        R, _ = cv2.Rodrigues(camera_euler_angles)

        # R 叉乘 camera_position  输出的R为旋转矩阵，这里求出R是为了后面进行旋转变换，
        # 输入的camera_euler_angles为旋转向量
        tvec = R @ camera_position

        points_2d, _ = cv2.projectPoints(
            test_cloud, camera_euler_angles, tvec, K, None
        )

        map_2d, _ = cv2.projectPoints(
            map_cloud, camera_euler_angles, tvec, K, None
        )

        points_2d = np.array(points_2d, dtype=np.int32)

        map_2d = np.array(map_2d, dtype=np.int32)

        for i in range(len(map_2d))[::4]:
            temp_points = [map_2d[i + j][0] for j in range(4)]
            distance = 300 - (map_cloud[i][0] * 100 + 150)
            if distance > 255:
                color = [255, distance - 255, 0]
            else:
                color = [distance, 0, 0]
            # print(color)
            img_3d = cv2.fillConvexPoly(img_3d, np.array(temp_points), color)
        # for p in points_2d:
        #     x, y = p[0]
        #     # cv2.circle(img, p[0], 5, (0, 255, 0), -1)
        #     try:
        #         img[y][x] = (0, 255, 0)
        #     except:
        #         pass
        #
        # cv2.imshow('test', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return points_2d, img_3d

    def get_distance_seg(self, model: YOLO, img_3d: np.ndarray, src_img: np.ndarray):
        result = model(src_img)
        # print(src_img)
        np_mask = result[0].masks[0].cpu().data.numpy() * 255
        img_mask = np_mask[0]
        img_mask = cv2.resize(img_mask, (self.SCREEN_W, self.SCREEN_H))
        rq = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))
        # print(img_mask.sh)

        up_bound = 0
        distance = 0
        x, y = 0, 0
        img_y_len = len(img_mask)

        for i in range(img_y_len):
            index = len(img_mask) - 1 - i
            if img_mask[index].any():
                if index - cut_bound > 0:
                    up_bound = index - cut_bound
                cut_mask = img_mask[up_bound:index]

                T_cut_mask = cut_mask.T
                for k in range(len(T_cut_mask)):
                    if T_cut_mask[k].any():
                        x = k
                        for s in range(len(T_cut_mask[k])):
                            if not T_cut_mask[k][s].any():
                                y = index - (cut_bound - 1 - s)
                                # print(s, img_mask.shape, T_cut_mask.shape)

                                color = img_3d[y][x]
                                # print(color)
                                distance = (color[0] + color[1]) / 100
                                img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
                                cv2.circle(img_mask, (x, y), 3, (0, 255, 0), 1)
                                cv2.imwrite(f'lands/img_mask_{rq}.jpg', img_mask)
                                return distance

                        break

        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
        cv2.circle(img_mask, (x, y), 3, (0, 255, 0), 1)
        cv2.imwrite(f'lands/img_mask_{rq}.jpg', img_mask)
        return distance


class Lidar:
    def __init__(self, com: str, file=None):
        self.file = file
        if not file:
            self.ser = serial.Serial(com, 512000, timeout=5)
            print('connect serial')
        self.jump_img = np.zeros((border_p, border_p), dtype=np.uint8)
        self.out_img = np.zeros((border_p, border_p), dtype=np.uint8)
        self.all_img = np.zeros((border_p, border_p), dtype=np.uint8)
        self.merge_img = []
        self.tick = 0
        self.cut_x_start = -1
        self.cut_x_end = -1
        self.cut_y_start = -1
        self.cut_y_end = -1

    def readlidar2img(self):
        last_angle = 0
        img = np.zeros((border_p, border_p), dtype=np.uint8)

        index = 0
        data = []
        dataset = []

        # 如果是文件
        if self.file:
            dataset = [bytes.fromhex(i[:-1]) for i in open(self.file, 'r').readlines()]

        while True:

            if self.file is None:
                # 读取2个字节数据
                data = self.ser.read(4)

            else:

                if index == len(dataset):
                    break

                data = dataset[index]
                index += 1

            # 判断是否为数据帧头
            if data[0] == 0xA5 and data[1] == 0x5A and data[2] == 0x00 and data[3] == 0xa0:

                # print('yes')

                if not self.file:
                    data = self.ser.read(156)

                else:
                    # 截去帧头
                    data = data[4:]

                # 起始角度：高字节在前，低字节在后，原始角度为方便传输放大了100倍，这里要除回去
                start_angle = (data[0] * 256 + data[1]) / 100.0

                # 从第四个单位数据开始，第141个结束
                start_unit, end_unit = 4, 141

                # 声明一个内部函数
                def toImgPoint(real_pos):
                    return int(border_p * (real_pos / border_m) + border_p / 2)

                # 解析点云串遍历开始
                j = 5

                # 2个字节的距离数据，步长为2
                for x in range(start_unit, end_unit, 2):
                    distance = 0
                    if data[x] & 0x80:
                        distance = (data[x] & 0x7F) << 8 | data[x + 1]
                        if distance & 0x7f:
                            continue
                    else:
                        distance = data[x] * 256 + data[x + 1]
                    distance /= 1000

                    # 转化为xy
                    now_angle = start_angle + 15 / 68 * (j - 5)
                    x = toImgPoint(distance * sin(radians(now_angle)))
                    y = toImgPoint(distance * cos(radians(now_angle)))

                    # 绘制点云
                    try:
                        img[y][x] = 255
                    except:
                        pass

                    # 每轮循环结束后
                    j += 1

                if last_angle - start_angle > 100:

                    # 对目标区域切片截取ROI
                    self.jump_img = img[toImgPoint(0.2):toImgPoint(0.8), toImgPoint(-2):toImgPoint(-1.5)]
                    self.out_img = img[toImgPoint(0.2):toImgPoint(0.8), toImgPoint(-1):toImgPoint(1)]
                    self.all_img = img

                    try:
                        merge = np.hstack([self.jump_img, self.out_img])
                        merge = cv2.cvtColor(merge, cv2.COLOR_GRAY2BGR)
                        self.merge_img = cv2.resize(merge, (1280, 720))
                        # cv2.imshow('merge', merge)
                        # if ord('b') == cv2.waitKey(1):
                        #     break
                    except:
                        traceback.print_exc()

                    img = np.zeros((border_p, border_p), dtype=np.uint8)
                    self.tick += 1

                last_angle = start_angle
            else:
                if self.file is None:
                    self.ser.read(155)


def nothing(x):
    pass


def com_lidar():
    lidar = Lidar(com="COM43")
    lidar_thread = threading.Thread(target=lidar.readlidar2img, daemon=True)

    lidar_thread.start()
    print('lidar started')

    cap = cv2.VideoCapture(CAMERA_INDEX)
    visionSolution = VisionSolution()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, visionSolution.SCREEN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, visionSolution.SCREEN_H)

    cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
    cv2.namedWindow('stand', cv2.WINDOW_NORMAL)
    cv2.namedWindow('land', cv2.WINDOW_NORMAL)
    cv2.namedWindow('all_img', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('x', 'camera', 53, 100, nothing)
    cv2.createTrackbar('y', 'camera', 49, 100, nothing)
    cv2.createTrackbar('z', 'camera', 13, 100, nothing)
    cv2.createTrackbar('p', 'camera', 43, 90, nothing)

    while True:
        _, img = cap.read()

        camera_x = cv2.getTrackbarPos('x', 'camera')
        visionSolution.camera_x = (camera_x - 50) / 100
        camera_y = cv2.getTrackbarPos('y', 'camera')
        # -0.84 -0.6
        visionSolution.camera_y = - 1.5 + (camera_y - 50) / 100
        camera_z = cv2.getTrackbarPos('z', 'camera')
        visionSolution.camera_z = - 1 + (camera_z - 50) / 100
        pitch = cv2.getTrackbarPos('p', 'camera')
        visionSolution.pitch = pitch
        print(camera_x, camera_y, camera_z, pitch, end='\r')
        points_2d, img_3d = visionSolution.projectMethod(img)
        show_img = img
        for p in points_2d:
            x, y = p[0]
            try:
                cv2.circle(img, p[0], 1, (0, 255, 0), -1)
                # img[y][x] = (0, 255, 0)
            except:
                pass
        # show_img = np.vstack((show_img, img_3d))
        cv2.imshow('camera', show_img)
        cv2.imshow('stand', lidar.jump_img)
        cv2.imshow('land', lidar.out_img)
        try:
            cv2.imshow('all_img', lidar.all_img)
        except:
            pass

        if cv2.waitKey(1) == ord('b'):
            cv2.destroyAllWindows()
            break

    cap.release()

def main2():
    lidar = Lidar(com="COM4")
    lidar_thread = threading.Thread(target=lidar.readlidar2img, daemon=True)

    lidar_thread.start()
    tick = lidar.tick
    print('lidar started')

    model = YOLO('yolov8n-seg.pt')
    print('yolo_seg loaded')

    cap = cv2.VideoCapture(CAMERA_INDEX)
    visionSolution = VisionSolution()
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    rq = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))

    # out = cv2.VideoWriter(f'out_{rq}.mp4', fourcc, 15, (visionSolution.SCREEN_W, visionSolution.SCREEN_H))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, visionSolution.SCREEN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, visionSolution.SCREEN_H)

    cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
    cv2.namedWindow('stand', cv2.WINDOW_NORMAL)
    cv2.namedWindow('land', cv2.WINDOW_NORMAL)
    cv2.namedWindow('all_img', cv2.WINDOW_NORMAL)
    cv2.namedWindow('img_3d', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('x', 'camera', init_x, 100, nothing)
    cv2.createTrackbar('y', 'camera', init_y, 100, nothing)
    cv2.createTrackbar('z', 'camera', init_z, 100, nothing)
    cv2.createTrackbar('p', 'camera', init_p, 90, nothing)

    is_land = False
    is_jump = False
    is_stand = False
    is_start = False

    while True:
        _, img = cap.read()
        camera_x = cv2.getTrackbarPos('x', 'camera')
        visionSolution.camera_x = (camera_x - 50) / 100
        camera_y = cv2.getTrackbarPos('y', 'camera')
        # -0.84 -0.6
        visionSolution.camera_y = - 1.5 + (camera_y - 50) / 100
        camera_z = cv2.getTrackbarPos('z', 'camera')
        visionSolution.camera_z = - 1 + (camera_z - 50) / 100
        pitch = cv2.getTrackbarPos('p', 'camera')
        visionSolution.pitch = pitch
        # print(visionSolution.camera_x, visionSolution.camera_y, visionSolution.camera_z, visionSolution.pitch, end='\r')
        print(camera_x, camera_y, camera_z, pitch, end='\r')
        points_2d, img_3d = visionSolution.projectMethod(img)
        show_img = img
        for p in points_2d:
            x, y = p[0]
            try:
                cv2.circle(img, p[0], 1, (0, 255, 0), -1)
                # img[y][x] = (0, 255, 0)
            except:
                pass
        # show_img = np.vstack((show_img, img_3d))
        cv2.imshow('img_3d', img_3d)
        cv2.imshow('camera', show_img)
        cv2.imshow('stand', lidar.jump_img)
        cv2.imshow('land', lidar.out_img)
        try:
            cv2.imshow('all_img', lidar.all_img)
        except:
            pass
        # if is_start:
        #     # out.write(show_img)

        key = cv2.waitKey(1)

        if key == ord('p'):
            if not is_start:
                print('start recording!')
                is_start = True
            else:
                print('stop recording!')
                is_start = False

        if key == ord('s'):
            print('save recording!')
            # out.release()

        if key == ord('b'):
            cv2.destroyAllWindows()
            break

        if tick == lidar.tick:
            continue

        tick = lidar.tick

        if is_land:
            continue

        # 检测到有落点
        if lidar.out_img.any():
            time.sleep(0.2)
            _, tick_img = cap.read()
            print(f'distance: {visionSolution.get_distance_seg(model, img_3d, tick_img)}')

            rq = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))
            print(f'land! {rq}')
            # cv2.imwrite(f'lands/land_lidar_{rq}.jpg', lidar.out_img)
            cv2.imwrite(f'lands/land_cam_{rq}.jpg', tick_img)
            is_land = True

        if is_jump:
            continue

        if not lidar.jump_img.any() and not is_stand:
            continue

        is_stand = True

        if not lidar.jump_img.any():

            # _, tick_img = cap.read()
            # rq = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))
            print(f'jump! {rq}')
            # cv2.imwrite(f'lands/land_lidar_{rq}.jpg', lidar.jump_img)
            # cv2.imwrite(f'lands/land_jump_{rq}.jpg', tick_img)
            is_jump = True
    cap.release()
    # out.release()


def main():
    # lidar = Lidar(com="COM43", file="ldata/ldata5.txt")
    lidar = Lidar(com="COM4")
    lidar_thread = threading.Thread(target=lidar.readlidar2img, daemon=True)

    # time.sleep(2)
    lidar_thread.start()

    tick = lidar.tick

    visionSolution = VisionSolution()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, visionSolution.SCREEN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, visionSolution.SCREEN_H)

    is_land = False
    is_jump = False
    is_stand = False

    while True:
        _, img = cap.read()
        points_2d, img_3d = visionSolution.projectMethod(img)
        for p in points_2d:
            x, y = p[0]
            try:
                cv2.circle(img, p[0], 1, (0, 255, 0), -1)
                # img[y][x] = (0, 255, 0)
            except:
                pass

        if len(lidar.merge_img) > 0:
            show_img = np.vstack((img, lidar.merge_img))
            show_img = cv2.resize(show_img, (800, 900))
        else:
            show_img = img
        cv2.imshow('camera', show_img)
        if cv2.waitKey(1) == ord('b'):
            cv2.destroyAllWindows()
            break
        # if not lidar_thread.is_alive():
        #     break

        if tick == lidar.tick:
            continue

        tick = lidar.tick

        if is_land:
            continue

        # 检测到有落点
        if lidar.out_img.any():
            rq = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))
            print(f'land! {rq}')
            cv2.imwrite(f'lands/land_lidar_{rq}.jpg', lidar.out_img)
            # cv2.imwrite(f'lands/land_cam_{rq}.jpg', img)
            is_land = True

        if is_jump:
            continue

        if not lidar.jump_img.any() and not is_stand:
            continue

        is_stand = True

        if not lidar.jump_img.any():
            rq = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))
            print(f'jump! {rq}')
            cv2.imwrite(f'lands/land_lidar_{rq}.jpg', lidar.jump_img)
            # cv2.imwrite(f'lands/land_cam_{rq}.jpg', img)
            is_jump = True


if __name__ == "__main__":
    main2()
    # com_lidar()
    # imgSplit()
