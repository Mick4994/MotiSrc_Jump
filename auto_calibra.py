import cv2
import numpy as np


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


def imgSplit():

    img = cv2.imread('lands/land_cam_2024-01-05 17_36_33.jpg')
    print(img.shape)

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

    top_k = 3

    for line in sorted_lines[:top_k]:
        cv2.line(inrange_img_rgb, line[0][0], line[0][1], (0, 0, 255), 2)

    merge = np.vstack((img, inrange_img_rgb))
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)

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
    
    cv2.imshow('test', merge)
    
    # 设置鼠标回调参数
    cv2.setMouseCallback('test', on_mouse_move, param)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    imgSplit()