# -*- coding: utf-8 -*-
# @Modified by: Ying CHEN
# @ProjectName:yolov5-pyqt5
# @File    : custom_util.py
# @Software: PyCharm
# @Brief   : 检测危险区域里面的人
import copy
import json
import os
from pathlib import Path
import numpy as np

import cv2

root = os.getcwd()
# 区域入侵的文件为ruqin.json，是写死的文件名
AREA_DANGEROUS_FILE_ROOT = os.path.join(root,'ruqin/ruqin.json')
a = []
b = []

# 段代码的主要目的是确定一个文件的路径（尽管路径连接可能有些问题）并初始化两个空列表。

def load_poly_area_data_simple(img_name=None):
    if(img_name is None):
        json_file_name = AREA_DANGEROUS_FILE_ROOT
    else:
        json_file_name = img_name  # 如果调用函数时没有提供 img_name（即它是 None）
                                   # 则使用先前定义的AREA_DANGEROUS_FILE_ROOT 作为JSON文件的路径否则使用提供的 img_name 作为文件路径。

    if not Path(json_file_name).exists():
        print(f"json file {json_file_name} not exists !! ")
        return []

# 如果不存在，则打印错误消息并返回一个空列表。

    with open(json_file_name, 'r') as f:
        json_info = json.load(f)
# 使用 json.load 方法读取文件内容到 json_info 变量中
        area_poly = []
        pts_len = len(json_info)  # 初始化一个空列表 area_poly 用于存储多边形数据，并计算 json_info 中的项数
        if pts_len % 2 is not 0:  # 多边形坐标点必定是2的倍数
            return []

        xy_index_max = pts_len // 2
        for i in range(0, xy_index_max):  # "x1": 402,"y1": 234,"x2": 497,"y2": 182,.....
            str_index = str(i + 1)
            x_index = 'x' + str_index
            y_index = 'y' + str_index
            one_poly = [json_info[x_index], json_info[y_index]]
            area_poly.append(one_poly)  # 读xy数据并添加到 area_poly 列表中

        return area_poly


def load_poly_area_data(img_name):
    """
    加载对用图片多边形点数据
    :param img_name: 图片名称
    :return: 多边形的坐标 [[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]] 二维数组
    """
    # area_file_path = os.getcwd() + "\\" + AREA_DANGEROUS_FILE_ROOT
    # json_file_name = area_file_path + img_name.split('.')[0] + ".json"
    json_file_name = AREA_DANGEROUS_FILE_ROOT

    if not Path(json_file_name).exists():
        print(f"json file {json_file_name} not exists !! ")
        return []

    with open(json_file_name, 'r') as f:
        json_info = json.load(f)

        area_poly = []
        for area_info in json_info['outputs']['object']:
            if 'polygon' not in area_info:
                return []
            # 代码遍历json_info字典中的'outputs'键下的'object'列表

            pts_len = len(area_info['polygon'])
            if pts_len % 2 is not 0:  # 多边形坐标点必定是2的倍数
                return []

            xy_index_max = pts_len // 2
            for i in range(0, xy_index_max):  # "x1": 402,"y1": 234,"x2": 497,"y2": 182,.....
                str_index = str(i + 1)
                x_index = 'x' + str_index
                y_index = 'y' + str_index
                one_poly = [area_info['polygon'][x_index], area_info['polygon'][y_index]]
                area_poly.append(one_poly)

        return area_poly
# 这段代码没用

def draw_poly_area_dangerous(img, img_name,throughJSON=True):
    """
    画多边形危险区域的框
    :param img: 图像本身
    :param img_name:用于加载绘制区域的json路径 / 一对点的x和y list组成的list
    :param done:是否需要执行以下代码
    :return:
    """
    if throughJSON:
        area_poly = np.array(load_poly_area_data_simple(img_name), np.int32)
        cv2.polylines(img, [area_poly], isClosed=True, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
    else:
        cv2.polylines(img, [img_name], isClosed=True, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)



# 在图像上绘制多边形危险区域

def is_poi_in_poly(pt, poly):
    """
    判断点是否在多边形内部的 pnpoly 算法
    :param pt: 点坐标 [x,y]
    :param poly: 点多边形坐标 [[x1,y1],[x2,y2],...]
    :return: 点是否在多边形之内
    """
    nvert = len(poly)
    print("判断人入侵时的多边形坐标 ",poly)
    vertx = []
    verty = []
    testx = pt[0]
    testy = pt[1]
    for item in poly:
        vertx.append(item[0])
        verty.append(item[1])
# 使用一个 for 循环遍历 poly，将每个顶点的 x 和 y 坐标分别添加到 vertx 和 verty 列表中。
    j = nvert - 1
    res = False
    for i in range(nvert):
        if (verty[j] - verty[i]) == 0:
            j = i
            continue
        x = (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) + vertx[i]
        if ((verty[i] > testy) != (verty[j] > testy)) and (testx < x):
            res = not res
            # 根据穿越数的奇偶性，代码返回了一个布尔值，表示点是否在多边形内部。
        j = i
    return res


def person_in_poly_area_dangerous_draw(xyxy,area_poly):
    if not area_poly:  # 为空
        return False

    # 求物体框的中点
    object_x1 = int(xyxy[0])
    object_y1 = int(xyxy[1])
    object_x2 = int(xyxy[2])
    object_y2 = int(xyxy[3])
    object_w = object_x2 - object_x1
    object_h = object_y2 - object_y1
    object_cx = object_x1 + (object_w / 2)
    object_cy = object_y1 + (object_h / 2)
# 计算了物体边界框的中心点坐标,下面的为判断该点是否在区域内
    return is_poi_in_poly([object_cx, object_cy], area_poly)

def person_in_poly_area_dangerous(xyxy,area_poly):
    """
    检测人体是否在多边形危险区域内
    :param xyxy: 人体框的坐标
    :param img_name: 检测的图片标号，用这个来对应图片的危险区域信息
    :return: True -> 在危险区域内，False -> 不在危险区域内
    """
    # area_poly = load_poly_area_data_simple(img_name)
    # print(area_poly)
    if not area_poly:  # 为空
        return False
    # 求物体框的中点
    object_x1 = int(xyxy[0])
    object_y1 = int(xyxy[1])
    object_x2 = int(xyxy[2])
    object_y2 = int(xyxy[3])
    object_w = object_x2 - object_x1
    object_h = object_y2 - object_y1
    object_cx = object_x1 + (object_w / 2)
    object_cy = object_y1 + (object_h / 2)

    return is_poi_in_poly([object_cx, object_cy], area_poly)


if __name__ == '__main__':
    pass
