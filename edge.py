# 测试从excel中读取数据，计算两点之间的距离，进行连边！
import networkx as nx
import matplotlib.pyplot as plt
import xlrd
from math import radians, cos, sin, asin, sqrt,atan,tan,acos

# longitude 经度 latitude 纬度

def calcDistance(Lng_A, Lat_A, Lng_B, Lat_B):
    """

    计算的距离（米）和我在arcgis上边的到的小数点对不上，所以直接返回整数值。
    根据两个点的经纬度求两点之间的距离(米）
    :param Lng_A:  经度1
    :param Lat_A:   维度1
    :param Lng_B:  经度2
    :param Lat_B:   维度2
    :return:  单位米
    """
    ra = 6378.140
    rb = 6356.755
    flatten = (ra - rb) / ra
    rad_lat_A = radians(Lat_A)
    rad_lng_A = radians(Lng_A)
    rad_lat_B = radians(Lat_B)
    rad_lng_B = radians(Lng_B)
    pA = atan(rb / ra * tan(rad_lat_A))
    pB = atan(rb / ra * tan(rad_lat_B))
    xx = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B))
    c1 = (sin(xx) - xx) * (sin(pA) + sin(pB)) ** 2 / cos(xx / 2) ** 2
    # 经测试当传入这两个点的经纬度一样时会返回 *
    if sin(xx/2)==0:
        return '*'
    c2 = (sin(xx) + xx) * (sin(pA) - sin(pB)) ** 2 / sin(xx / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (xx + dr)
    return int(distance*1000)

