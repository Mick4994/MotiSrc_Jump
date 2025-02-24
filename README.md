# 基于立体视觉预映射投影图的立定跳远距离识别

```
getchessimg.py 棋盘格截图保存脚本
getcam.py 根据棋盘格图片计算相机内参
read_com_data 直接通过2d激光雷达获取立定跳远的距离
seg.py 测试ultralytics包是否顺利安装，yolov8n-seg.pt模型是否下载成功

重点
mix.py 立定跳远距离识别主程序，混合整理所有试验代码：
    初始化函数：
    - buildPreDistanceMap 建立预映射图
    - com_lidar 读取2d激光雷达数据，转换为可视化图像
    触发函数：
    - get_distance_seg 从segment图像求得脚后跟图像坐标点，求得在预映射图中的距离
    循环函数：
    - readlidar2img 读取2d激光雷达数据，转换为可视化图像
```
