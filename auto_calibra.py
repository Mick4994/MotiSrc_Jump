import os
import cv2
import copy
import time
import tqdm
import math
import warnings
import itertools
import numpy as np
import multiprocessing
from typing import NamedTuple
from mix import VisionSolution
warnings.filterwarnings("ignore", category=RuntimeWarning)


yellow_hsv_min = np.array([9, 43, 43], dtype=np.uint8)
yellow_hsv_max = np.array([90, 255, 255], dtype=np.uint8)


# 在文件顶部添加鼠标回调函数
# 在鼠标回调函数中添加滚轮处理
def on_mouse_move(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        pass
        
    # 添加滚轮缩放处理
    elif event == cv2.EVENT_MOUSEWHEEL:
        scale_factor = 1.1 if flags > 0 else 0.9
        param['zoom_scale'] = np.clip(param['zoom_scale'] * scale_factor, 1.0, 8.0)
        
    # 添加左键点击处理
    elif event == cv2.EVENT_LBUTTONDOWN:
        # 获取合并图像的高度信息
        img, hsv_img, img_height = param['img'], param['hsv_img'], param['img_height']

        # 判断鼠标在原始图像区域还是处理结果区域
        if y < img_height:
            bgr = img[y, x]
            hsv = hsv_img[y, x]
            print(f"原始图像区域 ({x}, {y}) - RGB:({bgr[2]},{bgr[1]},{bgr[0]}) HSV:({hsv[0]},{hsv[1]},{hsv[2]})")
        else:
            bgr = param['processed_img'][y - img_height, x]
            print(f"处理结果区域 - RGB:({bgr[2]},{bgr[1]},{bgr[0]})")
        
    # 计算缩放区域
    h, w = param['original_merge'].shape[:2]
    size = int(200 / param['zoom_scale'])
    x1 = max(0, x - size//2)
    y1 = max(0, y - size//2)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)

    # 显示缩放区域
    zoom_img = param['original_merge'][y1:y2, x1:x2]
    zoom_img = cv2.resize(zoom_img, (200, 200), interpolation=cv2.INTER_LINEAR)
    cv2.circle(zoom_img, (x - x1, y - y1), 1, (0, 0, 255), -1)
    cv2.imshow('Zoom', zoom_img)

    # 在原图画定位框
    marked_img = param['original_merge'].copy()
    cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imshow('test', marked_img)


def getTopKLine(top_k = 3, debug = False, show_line=True):

    img = cv2.imread('lands/land_cam_2024-01-05 17_36_33.jpg')

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 提高色彩饱和度
    # img[:, :, 1] = np.clip(img[:, :, 1] * 1.1, 0, 255)
    inrange_img = cv2.inRange(hsv_img, yellow_hsv_min, yellow_hsv_max)

    # # 膨胀操作
    kernel_dilate = np.ones((6, 1), np.uint8)
    inrange_img = cv2.dilate(inrange_img, kernel_dilate, iterations=5)

    # # 腐蚀操作
    # kernel_erode = np.ones((3, 3), np.uint8)
    # inrange_img = cv2.erode(inrange_img, kernel_erode, iterations=6)

    inrange_img_rgb = cv2.cvtColor(inrange_img, cv2.COLOR_GRAY2BGR)

    # 查找轮廓
    contours, _ = cv2.findContours(inrange_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_length_ratio = 10  # 定义细长区域的最小长宽比（例如，长度是宽度的2倍以上）

    # 定义筛选长宽比区间
    length_ratio_area = [4, 10]

    lines = []

    for contour in contours:
        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        (x, y), (w, h), angle = rect
        
        if min(w, h) == 0:
            continue

        # 计算长宽比
        ratio = w / h if w > h else h / w
        
        # 筛选长宽比较大的区域（细长区域）

        # if length_ratio_area[0] <= ratio <= length_ratio_area[1]:
        if ratio > min_length_ratio:

            # 绘制最小外接矩形
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            # cv2.drawContours(inrange_img_rgb, [box], 0, (0, 255, 0), 2)
            
            # 找到和box[0]，也就是第一个点，在其他点中欧氏距离下最近的点
            similar_i, similar = min(
                [i for i in enumerate(box[1:])], 
                key=lambda x: np.hypot(x[1][0] - box[0][0], x[1][1] - box[0][1])
            )
            others = set(range(1, 4))
            others.remove(similar_i + 1)
            other1, other2 = list(others)
            x1 = int((box[0][0] + similar[0]) / 2)
            y1 = int((box[0][1] + similar[1]) / 2)
            x2 = int((box[other1][0] + box[other2][0]) / 2)
            y2 = int((box[other1][1] + box[other2][1]) / 2)
            
            # # 绘制直线
            # cv2.line(inrange_img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            lines.append([[(x1, y1), (x2, y2)], np.hypot(x2 - x1, y2 - y1)])

    sorted_lines = sorted(lines, key=lambda x: x[1], reverse=True)

    # for line in sorted_lines[:top_k]:
    #     cv2.line(inrange_img_rgb, line[0][0], line[0][1], (0, 0, 255), 2)

    sorted_lines = sorted_lines[:top_k]
    sorted_lines = [sorted(each[0], key=lambda x: x[1], reverse=True) for each in sorted_lines]
    sorted_lines = sorted(sorted_lines, key=lambda x: x[0][0])
    
    if not debug:
        return sorted_lines

    cv2.namedWindow('test', cv2.WINDOW_NORMAL)

    if not show_line:
        merge = np.vstack((img, inrange_img_rgb))
        # 先创建param字典
        param = {
            'img': img,
            'hsv_img': hsv_img,
            'processed_img': inrange_img_rgb,
            'img_height': img.shape[0]
        }
        # 添加缩放参数初始化
        param.update({
            'original_merge': merge.copy(),
            'zoom_scale': 1.0
        })
        
        # 创建缩放窗口
        cv2.namedWindow('Zoom', cv2.WINDOW_NORMAL)
        cv2.imshow('Zoom', np.zeros((200,200,3), np.uint8))
        
        # 设置鼠标回调参数
        cv2.setMouseCallback('test', on_mouse_move, param)
    
    for line in sorted_lines[:top_k]:
        if show_line:
            img = cv2.line(img, line[0], line[1], (255, 0, 0), 2)

        else:
            for i in range(2):
                inrange_img_rgb = cv2.circle(inrange_img_rgb, line[i], 10, (0, 0, 255), -1)
                merge = np.vstack((img, inrange_img_rgb))
                cv2.imshow('test', merge)
                cv2.waitKey(2000)
    if show_line:
        cv2.imshow('test', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def loss_func(topk_lines, p2ds, debug = False, printer = False, board_size = (1920, 1080)):
    """
    校准距离计算的损失评估函数

    Args:
        topk_lines: 真实相机通过图像处理得到最长的k条校准线
        p2ds: 虚拟相机通过立体视觉得到的2d点校准线
        debug: 是否开启debug模式
        pitch: 虚拟相机的俯仰角
        camera_x: 虚拟相机的x轴偏移
        camera_y: 虚拟相机的y轴偏移
        camera_z: 虚拟相机的z轴偏移

    Returns:
        损失评估函数值
    """

    # 反转+裁剪
    p2ds = p2ds[::-1]
    p2ds = p2ds[600: 600+300*3]

    if not debug:
        total_distance = 0
        mid_distance = 0
        
        p2d_list = []

        with np.errstate(divide='ignore', invalid='ignore'):
            for i in range(3):
                p2d1:np.ndarray = p2ds[i*300][0]
                p2d2:np.ndarray = p2ds[i*300+299][0]
                p2d_list.append([p2d1.tolist(), p2d2.tolist()])
                # y = kx + b
                p2d_line_k = (p2d2[1] - p2d1[1]) / (p2d2[0] - p2d1[0])
                p2d_line_b = p2d1[1] - p2d_line_k * p2d1[0]

                tkp1 = topk_lines[i][0]
                tkp2 = topk_lines[i][1]
                
                div = tkp2[0] - tkp1[0]
                if tkp1[1] - tkp2[1] == 0:
                    div = 0.1
                tkp_line_k = (tkp2[1] - tkp1[1]) / div

                max_k = max(p2d_line_k, tkp_line_k)
                min_k = min(p2d_line_k, tkp_line_k)

                k_rate = max_k / min_k

                mid_tkp = ((tkp1[0] + tkp2[0]) // 2, (tkp1[1] + tkp2[1]) // 2)
                # d = |kx - y + b| / √ (k² + 1)
                # 使用numpy的安全除法避免警告
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p2line_distance = abs(p2d_line_k * mid_tkp[0] - mid_tkp[1] + p2d_line_b) / np.sqrt(p2d_line_k**2 + 1)
                    
                total_distance += p2line_distance * 2 + k_rate
                if printer:
                    print(f'p2line_distance: {p2line_distance}')
                    print(f'k_rate: {k_rate}')
                    print(f'total_distance: {total_distance}')
        return total_distance

    bg = np.zeros(
        (board_size[1], board_size[0], 3), 
        dtype=np.uint8
    )

    for i in range(3):
        
        p2d1 = p2ds[i*200][0]
        p2d2 = p2ds[i*200+199][0]

        cv2.line(bg, (int(p2d1[0]), int(p2d1[1])), (int(p2d2[0]), int(p2d2[1])), (0, 0, 255), 2)

    for line in topk_lines:
        cv2.line(bg, line[0], line[1], (0, 255, 0), 2)
    
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class CameraPose(NamedTuple):
    pitch: int
    camera_x: int
    camera_y: int
    camera_z: int
    img: np.ndarray
    topk_lines: list
    vision_solution: VisionSolution
    def __str__(self):
        return f'[pitch: {self.pitch}, camera_x: {self.camera_x}, camera_y: {self.camera_y}, camera_z: {self.camera_z}]'


def worker(camera_pose: CameraPose):
    copy_visionSolution = camera_pose.vision_solution
    copy_visionSolution.pitch = camera_pose.pitch
    copy_visionSolution.camera_x = (camera_pose.camera_x - 50) / 100
    copy_visionSolution.camera_y = - 1.5 + (camera_pose.camera_y - 50) / 100
    copy_visionSolution.camera_z = - 1 + (camera_pose.camera_z - 50) / 100

    p2ds, _ = copy_visionSolution.buildPreDistanceMap(camera_img=camera_pose.img)
    
    loss = loss_func(camera_pose.topk_lines, p2ds, debug=False)
    if math.isnan(loss):
        return None

    return (camera_pose, loss)


def calibrate():
    start_time = time.time()

    img = cv2.imread('lands/land_cam_2024-01-05 17_36_33.jpg')
    topk_lines = getTopKLine(debug=False)
    visionSolution = VisionSolution()

    print(f'加载图片topk校准线和虚拟相机: {time.time() - start_time:.2f}s')

    start_time = time.time()

    # 使用numpy生成参数范围
    pitch_range = np.arange(40, 60)
    camera_x_range = np.arange(45, 55)
    camera_y_range = np.arange(55, 75)
    camera_z_range = np.arange(45, 55)

    # 计算总参数组合数
    total_poses = (
        len(pitch_range) * 
        len(camera_x_range) * 
        len(camera_y_range) * 
        len(camera_z_range)
    )

    # 使用itertools.product生成所有组合
    camera_pose_list = (
        CameraPose(pitch, camera_x, camera_y, camera_z, img, topk_lines, copy.deepcopy(visionSolution)) 
        for pitch, camera_x, camera_y, camera_z in itertools.product(
        pitch_range, camera_x_range, camera_y_range, camera_z_range)
    )

    print(f'生成参数组合: {time.time() - start_time:.2f}s')

    loss_list = []

    start_time = time.time()

    with multiprocessing.Pool(
        processes=os.cpu_count()
    ) as pool:
        print(f'多线程计算初始化: {time.time() - start_time:.2f}s')
        results = pool.imap_unordered(worker, camera_pose_list, chunksize=10)
        progress = tqdm.tqdm(results, total=total_poses)
        
        min_loss = 9999
        min_loss_poses = []
        for result in progress:
            if result is not None and result[1] < min_loss:
                min_loss = result[1]
                min_loss_poses = result[0]
                del result

    print(f'{min_loss_poses} loss=>{min_loss}')
    min_loss_visionSolution: VisionSolution = min_loss_poses.vision_solution
    min_p2ds, _ = min_loss_visionSolution.buildPreDistanceMap(camera_img=img)

    for topk_line in topk_lines:
        cv2.line(img, topk_line[0], topk_line[1], (255, 0, 0), 2)

    for p2d in min_p2ds:
        cv2.circle(img, (int(p2d[0][0]), int(p2d[0][1])), 2, (0, 255, 0), -1)

    loss_func(topk_lines, min_p2ds, printer=True)

    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', img)
    cv2.waitKey(0)


def find_min_loss(left, right, img, topk_lines, visionSolution_set_func: callable, visionSolution: VisionSolution):    
    while left < right:
        mid = (left + right) // 2
        visionSolution_set_func(mid)
        res1 = loss_func(topk_lines, visionSolution.buildPreDistanceMap(camera_img=img)[0])
        visionSolution_set_func(mid + 1)
        res2 = loss_func(topk_lines, visionSolution.buildPreDistanceMap(camera_img=img)[0])
        if res1 > res2:
            left = mid + 1
        else:
            right = mid
    return left


def test_binary_calibra():
    img = cv2.imread('lands/land_cam_2024-01-05 17_36_33.jpg')
    topk_lines = getTopKLine(debug=False)
    visionSolution = VisionSolution()

    pitch_range = (48, 49)
    x_range = (0, 100)
    y_range = (0, 100)
    z_range = (0, 100)


    start_time = time.time()

    min_pitch = find_min_loss(
        pitch_range[0], pitch_range[1], img, topk_lines, visionSolution.human_set_camera_pitch, visionSolution
    )

    visionSolution.pitch = min_pitch
    print(min_pitch)

    min_x = find_min_loss(
        x_range[0], x_range[1], img, topk_lines, visionSolution.human_set_camera_x, visionSolution
    )

    print(min_x)
    visionSolution.camera_x = (min_x - 50) / 100

    min_y = find_min_loss(
        y_range[0], y_range[1], img, topk_lines, visionSolution.human_set_camera_y, visionSolution 
    )

    print(min_y)
    visionSolution.camera_y = -1.5 + (min_y - 50) / 100

    min_z = find_min_loss(
        z_range[0], z_range[1], img, topk_lines, visionSolution.human_set_camera_z, visionSolution
    )

    print(min_z)
    visionSolution.camera_z = -1 + (min_z - 50) / 100

    print(f'计算完成: {time.time() - start_time:.2f}s')



if __name__ == '__main__':
    # getTopKLine(debug=True)
    # calibrate()
    test_binary_calibra()
