# 基于立体视觉预映射投影图的立定跳远距离识别

```
utils 工具包下：
getchessimg.py 棋盘格截图保存脚本
getcam.py 根据棋盘格图片计算相机内参
read_com_data 直接通过2d激光雷达获取立定跳远的距离
seg.py 测试ultralytics包是否顺利安装，yolov8n-seg.pt模型是否下载成功

重点
Lidar.py 2d激光雷达数据处理类，封装串口，将二进制串转为实时点云数据，以便后续主流程利用雷达点云判断落地
    - readlidar2img 读取2d激光雷达数据，转换为可视化图像
CameraVision.py 虚拟相机视觉类，集成预映射投影图和校准线点云构建的核心算法，封装YOLO-seg模型，
    - buildPreDistanceMap 建立预映射图
    - get_distance_seg 从segment图像求得脚后跟图像坐标点，求得在预映射图中的距离
auto_calibra.py 自动校准算法函数，包含：
    1. 实时topk校准线识别
    2. 校准评估的误差函数
    3. 第一版校准方案：暴力参数组合枚举校准算法，时间复杂度 2的n次方
    4. 第二版校准方案：坐标下降法校准算法，时间复杂度 n
mix.py 立定跳远主流程的逻辑，最上层应用：
    - com_lidar 只演示建立预映射投影图和校准算法的效果，不包含立定跳远流程的逻辑
    - main2 包含所有算法效果和立定跳远流程的逻辑
```
