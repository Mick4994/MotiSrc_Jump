import cv2


camera = cv2.VideoCapture(1)

camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(5, 30)

# print(camera.get(cv2.CAP_PROP_XI_FRAMERATE))

i = 0
cv2.namedWindow('img', cv2.WINDOW_NORMAL)

while 1:
    (grabbed, img) = camera.read()
    # print(img.shape, end='\r')
    cv2.imshow('img', img)
    key = cv2.waitKey(1)
    if key == ord('j'):  # 按j保存一张图片
        i += 1
        u = str(i)
        filename = str('./chess_img/img'+u+'.jpg')
        cv2.imwrite(filename, img)
        print('写入：', filename)
    if key == ord('b'):
        break


camera.release()