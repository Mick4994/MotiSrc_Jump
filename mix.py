import cv2
import time
import toml
import math
import threading
import numpy as np
from ultralytics import YOLO
from CameraVision import VisionSolution
from Lidar import Lidar
from auto_calibra import getTopKLine, coordinate_descent


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
CAMERA_INDEX = config['CAMERA_INDEX']


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
    cv2.createTrackbar('x', 'camera', 50, 100, lambda x: None)
    cv2.createTrackbar('y', 'camera', 50, 100, lambda x: None)
    cv2.createTrackbar('z', 'camera', 50, 100, lambda x: None)
    cv2.createTrackbar('p', 'camera', 50, 90, lambda x: None)

    while True:
        if debug:
            img = src.copy()
        else:
            _, img = cap.read()

        img_src = img.copy()

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
        
        key = cv2.waitKey(1)

        if key == ord('c'):
            print('start auto calibrate!')
            
            poses_ranges = [
                (40, 60),
                (40, 60),
                (30, 70),
                (40, 60) 
            ]
            topk_lines = getTopKLine(img=img_src)
            min_loss_vars, loss = coordinate_descent([pitch, camera_x, camera_y, camera_z], poses_ranges, img_src, topk_lines, visionSolution)
            cv2.setTrackbarPos('p', 'camera', min_loss_vars[0])
            cv2.setTrackbarPos('x', 'camera', min_loss_vars[1])
            cv2.setTrackbarPos('y', 'camera', min_loss_vars[2])
            cv2.setTrackbarPos('z', 'camera', min_loss_vars[3])

            if math.isnan(loss):
                print('calibrate failed!')
            else:
                print(f'calibrate success! result: {min_loss_vars}, loss: {loss}')

        elif key == ord('b'):
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

        img_src = img.copy()

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

        if key == ord('c'):
            print('start auto calibrate!')
            # 自动校准

            poses_ranges = [
                (40, 60),
                (40, 60),
                (30, 70),
                (40, 60) 
            ]
            topk_lines = getTopKLine(img=img_src)
            min_loss_vars, loss = coordinate_descent([pitch, camera_x, camera_y, camera_z], poses_ranges, img_src, topk_lines, visionSolution)
            cv2.setTrackbarPos('p', 'camera', min_loss_vars[0])
            cv2.setTrackbarPos('x', 'camera', min_loss_vars[1])
            cv2.setTrackbarPos('y', 'camera', min_loss_vars[2])
            cv2.setTrackbarPos('z', 'camera', min_loss_vars[3])

            if math.isnan(loss):
                print('calibrate failed!')
            else:
                print(f'calibrate success! result: {min_loss_vars}, loss: {loss}')

        elif key == ord('s'):
            print('save recording!')
            # out.release()

        elif key == ord('b'):
            cv2.destroyAllWindows()
            break

        # 这里流程复杂，需要补流程图 

        # 如果在之前下面的流程中，没有复杂的计算，例如检测到起跳等，lidar.tick还未到达下一帧，直接跳过，避免重复雷达帧运算
        if tick == lidar.tick:
            continue
        
        # 赋值为当前帧
        tick = lidar.tick

        # 已经落地过，全流程走完，不必再检测，跳过
        if is_land:
            continue

        # 检测到有落点，获取距离，完成一次跳远流程
        if lidar.out_img.any():
            # 0.2是个经验值，可能需要根据实际运行速度/相机帧率调整
            time.sleep(0.2)
            _, tick_img = cap.read()
            print(f'distance: {visionSolution.get_distance_seg(model, pre_distance_map, tick_img)}')

            rq = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))
            print(f'land! {rq}')
            # cv2.imwrite(f'lands/land_lidar_{rq}.jpg', lidar.out_img)
            cv2.imwrite(f'lands/land_cam_{rq}.jpg', tick_img)
            is_land = True

        # 代表检测过是在起跳准备阶段，直接跳过
        if is_jump:
            continue

        # 如果起跳区域没有雷达点云检出，并且之前不是已经检测到进入起跳区，则认为未开始进入起跳准备
        if not lidar.jump_img.any() and not is_stand:
            continue

        # 起跳区有雷达点云检出，则认为进入起跳准备
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

