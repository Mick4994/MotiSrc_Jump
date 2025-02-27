import cv2
import numpy as np
from math import sin, cos, radians
import time

# from cal_res import get_dis


# 读取串口录制数据
dataset = [bytes.fromhex(i[:-1]) for i in open('ldata5.txt', 'r').readlines()]

border_m = 5
border_p = 1000


def toImgPoint(real_pos):
    return int(border_p * (real_pos / border_m) + border_p / 2)


def toRealPoint(img_pos):
    return (img_pos - border_p / 2) / border_p * border_m


def get_dis(img: np.ndarray):
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    copy_img = img.copy()

    is_green = False

    count = 0

    index = toImgPoint(-2)

    edge_list = []

    for line in img[toImgPoint(-2):toImgPoint(2)]:
        if line[toImgPoint(0.25)][1] == 255 and not is_green:
            is_green = True
            count += 1
            cv2.line(copy_img, (0, index), (1000, index), (255, 255, 0), 1)
            edge_list.append(toRealPoint(index))
            print(f'count:{count}, {toRealPoint(index)}')

        if line[toImgPoint(0.25)][1] == 0 and is_green:
            is_green = False
            count += 1
            cv2.line(copy_img, (0, index), (1000, index), (255, 255, 0), 1)
            edge_list.append(toRealPoint(index))
            print(f'count:{count}, {toRealPoint(index):.3f}')

        # print(index)
        index += 1

    # cv2.imshow('copy', copy_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('cal_res5.jpg', copy_img)

    print(edge_list[2], edge_list[1])
    print(abs(edge_list[2] - edge_list[1]))


def dealIMG():
    img = np.zeros((border_p, border_p, 3), dtype=np.uint8)
    last_angle = 0

    odata = []
    vecdata = []
    x_list = [0 for i in range(1000)]
    for data in dataset:

        list_data = []

        # 判断是否为数据帧头
        if data[0] == 0xA5 and data[1] == 0x5A and data[2] == 0x00 and data[3] == 0xa0:
            data = data[4:]

            list_data.insert(0, "起始角度（度）:")

            # 高字节在前，低字节在后，原始角度为方便传输放大了100倍，这里要除回去
            start_angle = (data[0] * 256 + data[1]) / 100.0

            list_data.insert(1, start_angle)

            list_data.insert(2, "转速（圈/每分钟）：")
            list_data.insert(3, 2500000 / (data[2] * 256 + data[3]))
            list_data.insert(4, "距离（mm）*70个点：")
            j = 5

            # 2个字节的距离数据，步长为2
            for x in range(4, 141, 2):
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
                # cv2.circle(img, (x, y), 1, (255, 0, 0), 1)
                try:
                    img[y][x] = (255, 0, 0)
                except:
                    pass
                # print(img[y][x])

                cv2.rectangle(img, (toImgPoint(2), toImgPoint(0.8)), (toImgPoint(-2), toImgPoint(0.2)), (0, 0, 255), 1)

                if toImgPoint(-2) < x < toImgPoint(2) and toImgPoint(0.2) < y < toImgPoint(0.8):
                    x_list[x] += 1
                    # if x_list[x] > 2:
                    cv2.line(img, (x, 0), (x, border_p), (0, 255, 0), 1)

                # 将2个字节合并后转为十进制依次插入列表
                list_data.insert(j, distance)
                j += 1

            # 判断上一帧的起始角度与这一帧的角度差是否大于一定角度即为新的一圈
            if last_angle - list_data[1] > 100:
                # img = np.zeros((border_p, border_p, 3), dtype=np.uint8)
                odata.append(vecdata)
                vecdata = []

            # 将此帧角度赋值为上一帧，为下一次的判断做准备
            last_angle = list_data[1]

            vecdata.append(list_data[5:j])

            # cv2.imshow('point_cloud', img)
            # if cv2.waitKey(1) == ord('b'):
            #     break
            # cv2.waitKey(0)

    rq = time.strftime('%Y-%m-%d %H-%M', time.localtime(time.time()))
    # cv2.imwrite(f'test_data/{rq}.jpg', img)
    # np.save(f'test_data/{rq}.npy', img)
    cv2.destroyAllWindows()
    return img


def dealLidarOut():
    img = np.zeros((border_p, border_p, 3), dtype=np.uint8)

    last_angle = 0

    odata = []
    vecdata = []
    x_list = [0 for i in range(1000)]
    for data in dataset:

        list_data = []

        # 判断是否为数据帧头
        if data[0] == 0xA5 and data[1] == 0x5A and data[2] == 0x00 and data[3] == 0xa0:
            data = data[4:]

            list_data.insert(0, "起始角度（度）:")

            # 高字节在前，低字节在后，原始角度为方便传输放大了100倍，这里要除回去
            start_angle = (data[0] * 256 + data[1]) / 100.0

            list_data.insert(1, start_angle)

            list_data.insert(2, "转速（圈/每分钟）：")
            list_data.insert(3, 2500000 / (data[2] * 256 + data[3]))
            list_data.insert(4, "距离（mm）*70个点：")
            j = 5

            # 2个字节的距离数据，步长为2
            for x in range(4, 141, 2):
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

                try:
                    img[y][x] = (255, 0, 0)
                except:
                    pass

                cv2.rectangle(img, (toImgPoint(2), toImgPoint(0.8)), (toImgPoint(-2), toImgPoint(0.2)), (0, 0, 255), 1)

                if toImgPoint(-2) < x < toImgPoint(2) and toImgPoint(0.2) < y < toImgPoint(0.8):
                    x_list[x] += 1
                    # if x_list[x] > 2:
                    cv2.line(img, (x, 0), (x, border_p), (0, 255, 0), 1)

                # 将2个字节合并后转为十进制依次插入列表
                list_data.insert(j, distance)
                j += 1

            # 判断上一帧的起始角度与这一帧的角度差是否大于一定角度即为新的一圈
            if last_angle - list_data[1] > 100:
                # img = np.zeros((border_p, border_p, 3), dtype=np.uint8)
                odata.append(vecdata)
                vecdata = []

            # 将此帧角度赋值为上一帧，为下一次的判断做准备
            last_angle = list_data[1]

            vecdata.append(list_data[5:j])


if __name__ == "__main__":
    img = dealIMG()
    get_dis(img)
