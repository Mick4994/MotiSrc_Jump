import cv2
import time
from ultralytics import YOLO
import toml
import numpy as np
from math import radians

# 读取 TOML 配置文件
with open('config.toml', 'r', encoding='utf-8') as f:
    config = toml.load(f)

# 从配置中获取参数
cut_bound = config['cut_bound']

camera_K = np.array(config['camera_K']['RMX3700_K'], dtype=np.float32)

class VisionSolution:
    def __init__(self, static_picture=None) -> None:
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

        # 固定图片模式
        self.static_picture: np.ndarray = static_picture
        self.custom_args: dict = {}

    def _generate_calibra_line_cloud(self) -> np.ndarray:
        """生成校准用的校准线三维点云阵列
        Returns:
            np.ndarray: Nx3形状的校准线点云数组
        """
        # 生成x和z的数组
        x_values = np.arange(-2, 2, 0.5)
        z_values = np.arange(-0.3, 0.3, 0.002)

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

    def buildPreDistanceMap(self, camera_img=None):
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

        if isinstance(camera_img, np.ndarray):
            pre_distance_map = camera_img.copy()
        else:
            if self.static_picture:
                pre_distance_map = self.static_picture.copy()
            else:
                raise ValueError('camera_img is None')
        
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