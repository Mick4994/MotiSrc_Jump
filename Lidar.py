import cv2
import toml
import serial
import traceback
import numpy as np
from math import sin, cos, radians


# 读取 TOML 配置文件
with open('config.toml', 'r', encoding='utf-8') as f:
    config = toml.load(f)

# 从配置中获取参数
border_m = config['border_m']
border_p = config['border_p']


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
        """
        读取2d激光雷达点云数据，转换为可视化图像
        """
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