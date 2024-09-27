# 部分用到的库
from ultralytics import YOLO
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadStreams
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CFG, SETTINGS, callbacks
from ultralytics.yolo.utils.torch_utils import smart_inference_mode
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.checks import check_imshow
from ultralytics.yolo.cfg import get_cfg
from collections import deque
from ultralytics.yolo.utils.checks import check_imshow
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import supervision as sv
import numpy as np
import matplotlib
import threading
import datetime
import time
import json
import sys
import cv2
import os

# 要打开的文件，按q退出播放
source_dir = 'inference/input/42.mp4'
output_dir = 'inference/output'  # 要保存到的文件夹
show_video = True  # 运行时是否显示
save_video = True  # 是否保存运行结果视频
save_text = True  # 是否保存结果数据到文件中

# 一条线就是一个list，内容为[x1, y1, x2, y2, (B, G, R), 线的粗细]
# 一个视频可绘制任意条数检测线
# line[0]总车流量计数
# line[1]应急车道车流量计数
# line[2]密度统计
# 4：第4点位个视频中画线的标点
line4 = [[100, 200, 1100, 200, (0, 0, 255), 2],
         [700, 300, 1000, 310, (0, 255, 0), 2],
         [5, 210, 1000, 300, (255, 0, 0), 2]]

# 一些参数的定义
# x是点到左边的距离，y是点到顶上的距离
# 方框检测点的序号
#    1__________________2
#    |                  |
#    |                  |
#    |      0(中心点)   |
#    |                  |
#    |__________________|
#    4                  3


#    |-------> x轴
#    |
#    |
#    V
#    y轴


# 遍历每一个输出层的输出
for output in layerOutputs:
    # 遍历检测结果
    for d in output:
        # d : 1*85 0-3位置，4置信度 5-84类别
        scores = d[5:]  # 当前目标属于某一类别的概率
        classID = np.argmax(scores)  # 目标的类别ID
        confidence = scores[classID]  # 得到目标属于该类别的置信度

        if confidence > 0.3:
            # 将检测结果与原图片匹配，yolo的输出的是边界框的中心坐标和宽高，是对应图片的比例
            box = d[0:4] * np.array([W, H, W, H])
            centerX, centerY, width, height = box.astype('int')
            # 左上角坐标
            x = int(centerX - width / 2)
            y = int(centerY - height / 2)
            # 更新目标框、置信度、类别
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# 碰撞检测
if len(boxes) > 0:
    i = int(0)
    # 遍历跟踪框
    for box in boxes:
        # 左上角坐标和右下角坐标
        x1, y1 = int(box[0]), int(box[1])
        x2, y2 = int(box[2]), int(box[3])
        # # 对方框的颜色进行设定
        color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
        # 将方框绘制在画面上
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # 根据当前帧的检测结果，与上一帧的检测结果结合，通过虚拟线圈完成车流量统计
        if indexIDs[i] in previous:
            #  获取上一帧识别的目标框
            previous_box = previous[indexIDs[i]]
            # 获取上一帧画面追踪框的左上角坐标和宽高
            x3, y3 = int(previous_box[0]), int(previous_box[1])
            x4, y4 = int(previous_box[2]), int(previous_box[3])
            # 获取当前帧检测框的中心点
            p0 = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
            # 获取上一帧检测框的中心点
            p1 = (int(x3 + (x4 - x3) / 2), int(y3 + (y4 - y3) / 2))

            # # # 应急车道车辆数统计
            if intersect(p0, p1, line_yingji[0], line_yingji[1]):
                counter_yingji += 1  # 总计数加1

            if intersect(p0, p1, line_midu[0], line_midu[1]):
                counter_midu += 1  # 总计数加1

            # 进行碰撞检测
            if intersect(p0, p1, line[0], line[1]):
                counter += 1  # 总计数加1
                # # 判断行进的方向
                # if y3 < y1:
                #     counter_down += 1  # 逆向行驶+1
                # else:
                #     counter_up += 1  # 正向行驶+1

        i += 1

# 将车辆计数的相关结果绘制在视频上
# 根据设置的基准线将其绘制在画面上
cv2.line(frame, line[0], line[1], (0, 255, 0), 3)
cv2.line(frame, line_yingji[0], line_yingji[1], (0, 0, 255), 3)
cv2.line(frame, line_midu[0], line_midu[1], (255, 0, 0), 3)
# 绘制车辆的总计数
cv2.putText(frame, str(counter), (30, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 0, 0), 3)
list_counter.append(counter)
list_counter_yingji.append(counter_yingji)
list_counter_midu.append(counter_midu)
# 打印显示结果
print(f'总{list_counter[-1]}\t', f'密度{list_counter_midu[-1]}')

# # 程序运行入口，和部分配置参数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
                        help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default=source_dir, help='source')
    parser.add_argument('--output', type=str, default=output_dir, help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', default=class_list, type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
