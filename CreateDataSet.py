# 2022.9.11 Luna：根据gml图文件，生成cora.cites文件类型的数据集
# 2022.9.12 Luna: 根据gml图文件，根据函数create_content_file(G,filename）生成cora.content数据集
import networkx as nx
import matplotlib.pyplot as plt
import xlrd

from edge import *
import math
import copy, time, sys
import util as util
import numpy as np


def information_graph(G):
    # 输出网络相关 拓扑数据 如边数、节点数等
    print("output the number of nodes:", G.number_of_nodes())
    # print("output all nodes:", G.nodes()0)
    print("output the number of edges:", G.number_of_edges())
    print("output all edges:", G.edges())

def draw(G):

    pos = nx.spring_layout(G)
    # nx.draw(G, with_labels=True)
    # nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color=value, node_size=20)
    # plt.get_cmap（）函数用于画点，参考网站https://blog.csdn.net/weixin_39580795/article/details/102622004
    # nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    nx.draw(G,
            with_labels = False,
            node_size = 20,
            node_color='steelblue',
            width = 0.2)

    plt.show()
    # plt.savefig('1505.jpg')

def load_graph(path):
    if ".txt" in path:
        G = nx.read_edgelist(path, create_using=nx.Graph())
    elif ".gml" in path:
        G = nx.read_gml(path)
    elif ".csv" in path:
        G = nx.read_edgelist(path, create_using=nx.Graph())
    return G


def create_cites_file(G,filename):
    edge_information=G.edges()
    print(edge_information)
    print(type(G.edges()))
    edge_list=list(edge_information)
    print(type(edge_list))

    file = open(filename, 'w')  #'a'是续写 'w'是覆盖
    for i in range(len(edge_list)):
        s=str(edge_list[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        # j,k=edge_list[i][0],edge_list[i][1]
        s=s.replace("(", '').replace(')', '')
        s=s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("save successful!")

# def create_cotent_file(G,filename):    #生成的content尺寸为<id><attribute19><level>
#     n='\n'
#     list=['ls','flow','landcover_b1', 'landcover_b2','landcover_b3','landuse','ndvi',
#           'nightlight','slope','slopeslope','spi','tpi','twi','houseprice','aspectaspect',
#           'aspect','dem','geology','roadbuffer','level']
#     file = open(filename, 'w')  # 'a'是续写 'w'是覆盖
#     for j in range(1,1506):
#         node_attlist = G.nodes[str(j)]
#         s_id=str(j)+' '
#         file.write(s_id)
#         for i in list:
#             s=str(node_attlist[i])+' '
#             file.write(s)
#         file.write(n)
#     file.close()
#     print("save successful!")n

def create_content_file(G,filename):
    n='\n'
    list=['ls','flow','landcover_b1', 'landcover_b2','landcover_b3','landuse','ndvi',
          'nightlight','slope','slopeslope','spi','tpi','twi','houseprice','aspectaspect',
          'aspect','geology','roadbuffer','level']
    file = open(filename, 'w')  # 'a'是续写 'w'是覆盖
    for j in range(1,1496):
        node_attlist = G.nodes[str(j)]
        s_id=str(j)+' '
        file.write(s_id)
        for i in list:
            s=str(node_attlist[i])+' '
            file.write(s)
        file.write(n)
    file.close()
    print("save successful!")

if __name__ == '__main__':

    gmlname="linefive1495_17attributes_220m_3level"  #读取gmread_gmll
    G = nx.read_gml(gmlname+".gml")   #读取图文件，支持格式：txt，gml，csv（csv测试未通过）
    # information_graph(G)    #输出网络拓扑情况
    filename="F:\code\GraphNet\pyGAT-master\data\cora\cora.cites"     #cites文件保存路径
    filename_content="F:\code\GraphNet\pyGAT-master\data\cora\cora.content"      #content文件保存路径
    create_cites_file(G,filename)
    create_content_file(G,filename_content)
    # draw(G)