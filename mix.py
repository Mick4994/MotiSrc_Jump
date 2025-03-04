import cv2
import time
import toml
import serial
import traceback
import threading
import numpy as np
from ultralytics import YOLO
from math import sin, cos, radians

# 读取 TOML 配置文件
with open('config.toml', 'r', encoding='utf-8') as f:
    config = toml.load(f)

# 从配置中获取参数
border_m = config['border_m']
border_p = config['border_p']
init_x = config['init_x']
init_y = config['init_y']
init_z = config['init_z']
init_p = config['init_p']
cut_bound = config['cut_bound']
CAMERA_INDEX = config['CAMERA_INDEX']

camera_K = np.array(config['camera_K']['RMX3700_K'], dtype=np.float32)

class VisionSolution:
    def __init__(self) -> None:
        """初始化虚拟相机参数和测试点云"""
        # 显示参数
        self.SCREEN_W: int = 1920
        self.SCREEN_H: int = 1080
        
        # 相机参数
        self.camera_x: float = 0.0
        self.camera_y: float = -0.55
        self.camera_z: float = -0.5
        self.pitch: int = 20
        
        # 生成测试点云
        self.calibra_line_cloud: np.ndarray = self._generate_calibra_line_cloud()
        self.map_cloud: np.ndarray = self._generate_map_cloud()

    def _generate_calibra_line_cloud(self) -> np.ndarray:
        """生成校准用的校准线三维点云阵列
        Returns:
            np.ndarray: Nx3形状的校准线点云数组
        """
        # 生成x和z的数组
        x_values = np.arange(-2, 2, 0.5)
        z_values = np.arange(-0.2, 0.2, 0.002)

        # 重复x的每个元素以匹配z的元素数量，并平铺z以匹配x的元素数量
        x_repeated = np.repeat(x_values, len(z_values))
        z_tiled = np.tile(z_values, len(x_values))

        # 创建全零的y列
        y_zeros = np.zeros_like(x_repeated)

        # 合并列形成最终的二维数组
        res = np.column_stack((x_repeated, y_zeros, z_tiled))
        return res

    def _generate_map_cloud(self) -> np.ndarray:
        """生成地图标定点云
        Returns:
            np.ndarray: Nx3形状的地图点云数组
        """
        x_values = np.arange(-1.5, 1.5, 0.01)
        z_values = np.array([-1, 1])  # z轴边界
        x_repeated = np.repeat(x_values, len(z_values))
        z_tiled = np.tile(z_values, len(x_values))
        res = np.column_stack((x_repeated, np.zeros_like(x_repeated), z_tiled))
        return res

    def human_set_camera_pitch(self, pitch: int):
        """对操作人暴露的设置相机俯仰角
        Args:
            pitch (int): 俯仰角，单位为度
        """
        self.pitch = pitch

    def human_set_camera_x(self, x: int):
        """对操作人暴露的设置相机x轴位置
        Args:
            x (float): x轴位置，单位为米
            """
        self.camera_x = (x - 50) / 100

    def human_set_camera_y(self, y: int):
        """对操作人暴露的设置相机y轴位置
        Args:
            y (float): y轴位置，单位为米
        """
        self.camera_y = - 1.5 + (y - 50) / 100

    def human_set_camera_z(self, z: int):
        """对操作人暴露的设置相机z轴位置
        Args:
            z (float): z轴位置
        """
        self.camera_z = - 1 + (z - 50) / 100

    def buildPreDistanceMap(self, camera_img):
        """
        将三维点云通过相机内外参的立体视觉计算映射到相机屏幕上，建立预映射图

        输入：
            camera_img: 相机图像
        输出：
            calibra_line_2d: 映射到屏幕的校准线点云
            pre_distance_map: 预映射图
        """
        calibra_line_cloud = np.array(self.calibra_line_cloud, dtype=np.float32)
        map_cloud = np.array(self.map_cloud, dtype=np.float32)

        camera_position = np.array([self.camera_x, self.camera_y, self.camera_z], dtype=np.float32)
        camera_euler_angles = np.array([radians(self.pitch), radians(0), radians(0)], dtype=np.float32)
        img = np.zeros((self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8)
        pre_distance_map = camera_img.copy()
        
        # 计算被旋转后的平移向量
        R, _ = cv2.Rodrigues(camera_euler_angles)

        # R 叉乘 camera_position  输出的R为旋转矩阵，这里求出R是为了后面进行旋转变换，
        # 输入的camera_euler_angles为旋转向量
        tvec = R @ camera_position

        calibra_line_2d, _ = cv2.projectPoints(
            calibra_line_cloud, camera_euler_angles, tvec, camera_K, None
        )

        map_2d, _ = cv2.projectPoints(
            map_cloud, camera_euler_angles, tvec, camera_K, None
        )

        calibra_line_2d = np.array(calibra_line_2d, dtype=np.int32)

        map_2d = np.array(map_2d, dtype=np.int32)

        # 遍历映射到屏幕的点云map_2d，每隔4个点取一个点，是一个包围的单位距离线条区域，同时取对应三维点云map_cloud的距离，写为色值，这种方法成为建立预映射图
        for i in range(len(map_2d))[::4]:
            temp_points = [map_2d[i + j][0] for j in range(4)]

            # 转换为从起跳线（点云坐标系下的-1.5m）开始的距离，单位从米转为厘米
            distance = int(300 - (map_cloud[i][0] * 100 + 150))

            # 将距离写到图像中每个像素的颜色的色值中
            if distance > 255:
                color = [255, distance - 255, 0]
            else:
                color = [distance, 0, 0]

            # 将该包围的单位距离线条区域填充为包含该距离的色值
            pre_distance_map = cv2.fillConvexPoly(pre_distance_map, np.array(temp_points), color)

        return calibra_line_2d, pre_distance_map

    def get_distance_seg(self, model: YOLO, pre_distance_map: np.ndarray, src_img: np.ndarray):
        """
        从segment图像求得脚后跟图像坐标点，求得在预映射图中的距离

        输入：
            model: YOLO模型
            pre_distance_map: 3D图像
            src_img: 源图像
        输出：
            distance: 距离，单位为m
        """
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

        # 遍历出脚后跟图像坐标点，从下往上遍历，找到第一个有像素的点，然后向上遍历，找到第一个有像素的点，
        # 然后将该点的颜色值取出，转换为距离，返回
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

                                color = pre_distance_map[y][x]
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


def com_lidar(use_lidar: bool = True, debug: bool = False):
    """
    校准窗口滑块调整虚拟相机位置，使虚拟相机校准线 对齐 真实相机画面中现实跳远刻度线
    """
    if use_lidar:
        lidar = Lidar(com="COM43")
        lidar_thread = threading.Thread(target=lidar.readlidar2img, daemon=True)

        lidar_thread.start()
        print('lidar started')

    visionSolution = VisionSolution()

    if debug:
        src = cv2.imread('lands/land_cam_2024-01-05 17_36_33.jpg')
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, visionSolution.SCREEN_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, visionSolution.SCREEN_H)

    cv2.namedWindow('camera', cv2.WINDOW_NORMAL)

    if use_lidar:
        cv2.namedWindow('stand', cv2.WINDOW_NORMAL)
        cv2.namedWindow('land', cv2.WINDOW_NORMAL)
        cv2.namedWindow('all_img', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('x', 'camera', 53, 100, lambda x: None)
    cv2.createTrackbar('y', 'camera', 49, 100, lambda x: None)
    cv2.createTrackbar('z', 'camera', 13, 100, lambda x: None)
    cv2.createTrackbar('p', 'camera', 43, 90, lambda x: None)

    while True:
        if debug:
            img = src.copy()
        else:
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
        
        # print(camera_x, camera_y, camera_z, pitch)
        points_2d, pre_distance_map = visionSolution.buildPreDistanceMap(img)
        show_img = img
        for p in points_2d:
            x, y = p[0]
            try:
                cv2.circle(img, p[0], 1, (0, 255, 0), -1)
                # img[y][x] = (0, 255, 0)
            except:
                pass
        # show_img = np.vstack((show_img, pre_distance_map))
        cv2.imshow('camera', show_img)
        if use_lidar:
            cv2.imshow('stand', lidar.jump_img)
            cv2.imshow('land', lidar.out_img)
            try:
                cv2.imshow('all_img', lidar.all_img)
            except:
                pass

        if cv2.waitKey(1) == ord('b'):
            cv2.destroyAllWindows()
            break

    if not debug:
        cap.release()

def main2(debug=False):
    """
    主函数，完成一次跳远的流程控制
    """
    lidar = Lidar(com="COM4")
    lidar_thread = threading.Thread(target=lidar.readlidar2img, daemon=True)

    lidar_thread.start()
    tick = lidar.tick
    print('lidar started')

    model = YOLO('utils/yolov8n-seg.pt')
    print('yolo_seg loaded')

    visionSolution = VisionSolution()
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    rq = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))

    # out = cv2.VideoWriter(f'out_{rq}.mp4', fourcc, 15, (visionSolution.SCREEN_W, visionSolution.SCREEN_H))

    if debug:
        src = cv2.imread('lands/land_cam_2024-01-05 17_36_33.jpg')
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, visionSolution.SCREEN_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, visionSolution.SCREEN_H)

    cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
    cv2.namedWindow('stand', cv2.WINDOW_NORMAL)
    cv2.namedWindow('land', cv2.WINDOW_NORMAL)
    cv2.namedWindow('all_img', cv2.WINDOW_NORMAL)
    cv2.namedWindow('pre_distance_map', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('x', 'camera', init_x, 100, lambda x: None)
    cv2.createTrackbar('y', 'camera', init_y, 100, lambda x: None)
    cv2.createTrackbar('z', 'camera', init_z, 100, lambda x: None)
    cv2.createTrackbar('p', 'camera', init_p, 90, lambda x: None)

    # 是否开始录制跳远（进入识别状态）
    is_start = False

    # 是否检测到落点
    is_land = False

    # 是否检测到起跳
    is_jump = False

    # 检测起点是否有人
    is_stand = False

    while True:
        if debug:
            img = src.copy()
        else:
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
        points_2d, pre_distance_map = visionSolution.buildPreDistanceMap(img)
        show_img = img

        # 把刻度线点云画在图像上
        for p in points_2d:
            x, y = p[0]
            try:
                cv2.circle(img, p[0], 1, (0, 255, 0), -1)
                # img[y][x] = (0, 255, 0)
            except:
                pass
        # show_img = np.vstack((show_img, pre_distance_map))
        cv2.imshow('pre_distance_map', pre_distance_map)
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

        # 这里流程复杂，需要补流程图 

        if tick == lidar.tick:
            continue

        tick = lidar.tick

        if is_land:
            continue

        # 检测到有落点，获取距离，完成一次跳远流程
        if lidar.out_img.any():
            time.sleep(0.2)
            _, tick_img = cap.read()
            print(f'distance: {visionSolution.get_distance_seg(model, pre_distance_map, tick_img)}')

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

        # 起跳区没有雷达点云检出，则认为起跳
        if not lidar.jump_img.any():

            # _, tick_img = cap.read()
            # rq = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))
            print(f'jump! {rq}')
            # cv2.imwrite(f'lands/land_lidar_{rq}.jpg', lidar.jump_img)
            # cv2.imwrite(f'lands/land_jump_{rq}.jpg', tick_img)
            is_jump = True
    cap.release()
    # out.release()


if __name__ == "__main__":
    # main2()
    # main2(debug=True)
    com_lidar(use_lidar=False, debug=True)

