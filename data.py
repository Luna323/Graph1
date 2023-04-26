# 从excel中读取数据，建网
# 2022.9.9 Luna:可构建无向无权多属性网络，包括21个属性信息，可生成gml图文件
# 2022.9.11 Luna：可读取gml文件，输出图拓扑信息


import networkx as nx
import matplotlib.pyplot as plt
import xlrd
from edge import *
import math
import copy, time, sys
import util as util
import numpy as np
fid='fid'
x='x'
y='y'
ls='ls'
flow='flow'
landcover_b1='landcover_b1'
landcover_b2='landcover_b2'
landcover_b3='landcover_b3'
landuse='landuse'
ndvi='ndvi'
nightlight='nightlight'
slope='slope'
slopeslope='slopeslope'
spi='spi'
tpi='tpi'
twi='twi'
houseprice='houseprice'
aspectaspect='aspectaspect'
aspect='aspect'
dem='dem'
geology='geology'
roadbuffer='roadbuffer'
level='level'
path='F:\code\GraphNet\Graph\dataset\\1495points_17attributes_220m.xls'
mindistance=220  #连边最小范围 80m

outfile = "linefive1495_17attributes_220m_3level"    #生成的网络.gexf\.gml名字   在当前文件生成，形如（network/outfile.gexf)
filename=xlrd.open_workbook(path)
sheetname=filename.sheet_by_name('Sheet1')
list_labels=['fid','x','y','ls','flow','landcover_b1','landcover_b2','landcover_b3',
             'landuse','ndvi','nightlight','slope','slopeslope','spi','tpi','twi',
             'houseprice','aspectaspect','aspect','geology','roadbuffer']


data1=sheetname.row_values(0) #获得excel的第一行数据
# print(data1)
# print(sheetname.nrows)  #excel行数  1506,包括题头

G = nx.Graph()

def create_node_attribute_edge(G):
    for i in range(1,1496):    #遍历excel中所有行 ，range（1,10）范围是从fid列0到8，共9行
    # for i in range(1, 100):
        cell=sheetname.row_values(i)
        print(cell)
        G.add_node(i)
        #     # 给i结点加入所有的属性信息
        #     # k 属性列表的下标
        # k=1
        # for j in list_labels:
        #     G.nodes[i][j]=cell[k]
        #     k=k+1

        # 给i结点加入前三个属性信息 'x','y','ls','flow','landcover_b1','landcover_b2','landcover_b3',
        #                       'landuse','ndvi','nightlight','slope','slopeslope','spi','tpi','twi',
        #                       'houseprice','aspectaspect','aspect','dem','geology','roadbuffer'
        G.nodes[i][x] = cell[1]
        G.nodes[i][y] = cell[2]
        G.nodes[i][ls] = cell[3]
        G.nodes[i][flow] = cell[4]
        G.nodes[i][landcover_b1] = cell[5]
        G.nodes[i][landcover_b2] = cell[6]
        G.nodes[i][landcover_b3] = cell[7]
        G.nodes[i][landuse] = cell[8]
        G.nodes[i][ndvi] = cell[9]
        G.nodes[i][nightlight] = cell[10]
        G.nodes[i][slope] = cell[11]
        G.nodes[i][slopeslope] = cell[12]
        G.nodes[i][spi] = cell[13]
        G.nodes[i][tpi] = cell[14]
        G.nodes[i][twi] = cell[15]
        G.nodes[i][houseprice] = cell[16]
        G.nodes[i][aspectaspect] = cell[17]
        G.nodes[i][aspect] = cell[18]
        G.nodes[i][geology] = cell[19]
        G.nodes[i][roadbuffer] = cell[20]

        # if cell[3] > 0:  # 给图节点加入分级属性
        #     G.nodes[i][level] = 1
        # if cell[3] == 0:
        #     G.nodes[i][level] = 2
        # if (cell[3] < 0.0 and cell[3] > -2.0):
        #     G.nodes[i][level] = 3
        # if (cell[3] < -2.0 and cell[3] > -5.0):
        #     G.nodes[i][level] = 4
        # if cell[3] < -5.0:
        #     G.nodes[i][level] = 5

        if cell[3] > 0:  # 给图节点加入分级属性
           G.nodes[i][level] = 2
        if (cell[3] < 0.0 and cell[3] > -3.0):
           G.nodes[i][level] = 1
        if cell[3] < -3.0:
           G.nodes[i][level] = 0

        for h in range(1,i):
            if (abs(G.nodes[h][x]-G.nodes[i][x] >=0.002)) or (abs(G.nodes[h][y]-G.nodes[i][y]>=0.002)):#经纬度相差0.001，相当于米相差111米
                print("the distance of these two points %d and %d is too far,lets we give up them. " %(h,i)  )
            else:
                distance = calcDistance(G.nodes[h][x], G.nodes[h][y], G.nodes[i][x], G.nodes[i][y])
                # print("output the distance of node_h and nodei :",h,i,distance)
                if distance<=mindistance:
                    G.add_edge(h,i)
                    print("create edge between the point %d and %d."%(h,i))
        G.add_edge(1275,1276)






def draw(G):

    pos = nx.spring_layout(G)
    # nx.draw(G, with_labels=True)
    # nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color=value, node_size=20)
    # plt.get_cmap（）函数用于画点，参考网站https://blog.csdn.net/weixin_39580795/article/details/102622004
    # nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    nx.draw(G,
            with_labels = False,
            font_size=8,
            node_size = 20,
            node_color='steelblue',
            width = 0.2)

    plt.show()
    # plt.savefig('1505.jpg')

def information_graph(G):
    # 输出网络相关 拓扑数据 如边数、节点数等
    print("output the number of nodes:", G.number_of_nodes())
    # print("output all nodes:", G.nodes())
    print("output the number of edges:", G.number_of_edges())
    # print("output all edges:", G.edges())
    # largest_components = max(nx.connected_components(G), key=len)   #输出最大联通子图的点的集合，但是这不是一个图，只是一个list
    # print("output the number of maxGson",len(largest_components))
    # print("output the point of maxGson",largest_components)

    # print("output all node 1 attributes:", G.nodes['1'])  # 使用这个方法可以输出节点1的全部属性字典！！！注意[]里边的类型需要是字符型,因为输出的节点是'1','2'这种 str()可以
    # print("output the level of node 1:", G.nodes['47'][level])  # 输出节点1的属性的内容
    # print("output the ls of node 1:", G.nodes[2][ls])
    #
    # dictx = nx.get_node_attributes(G, x)  # !!!可以获得拥有x=‘x'属性的所有结点字典，a为 结点：x属性值的形式，如{1: 114.235779}
    # print("output a:", dictx)
    #
    # mater = calcDistance(G.nodes[1][x], G.nodes[1][y], G.nodes[2][x], G.nodes[2][y])
    #
    # mater63_65 = calcDistance(114.303185, 22.599676, 114.303993, 22.599821)
    # print("output distance of 2 points(mater):",  mater63_65)
    # print("output the degree of node 1",G.degree('1'))  #输出节点的度

def create_graph():
    t1 = time.time()  # 返回当前时间的时间戳
    create_node_attribute_edge(G)
    draw(G)

    t2 = time.time()
    print('Time: we use %f seconds to build this graph ' % (t2 -t1))

    # 保存网络模型

    # nx.write_gml(G, "空间沉降网络_盐田区/"+ outfile +".gml")     #保存成gml文件
    # nx.write_gexf(G, outfile+".gexf")   #保存成gexf文件
    nx.write_gml(G, outfile + ".gml")  # 保存成gml文件

    return (t2 -t1)



def load_graph(path):
    if ".txt" in path:
        G = nx.read_edgelist(path, create_using=nx.Graph())
    elif ".gml" in path:
        G = nx.read_gml(path)
    elif ".csv" in path:
        G = nx.read_edgelist(path, create_using=nx.Graph())
    return G


class FN(object):
    """docstring for FN"""

    def __init__(self, G):
        self._G_cloned = util.clone_graph(G)  # 原始网络
        self._G = G
        self._max_Q = float("-inf")  # 给Q赋值 负无穷
        self._partition = None

        self._Group = {}  # 每个节点为一个community
        self._Node_GroupIndex = {}  # 每个节点对应的community索引
        # self._A_mat = [[0] * len(G.nodes())] * len(G.nodes())
        self._A_mat = np.array(nx.adjacency_matrix(G).todense())
        # print(self._A_mat)
        for i, n in enumerate(G.nodes()):
            self._Group[i] = [n]
            self._Node_GroupIndex[n] = i

    def execute(self):
        while len(self._Group) > 1:
            t1 = time.time()
            det_Q = float("-inf")
            max_edge = None
            for edge in self._G.edges():
                index_i = self._Node_GroupIndex[edge[0]]
                index_j = self._Node_GroupIndex[edge[1]]

                # 已经划分为同一社区的edge不用考虑
                if index_i == index_j: continue

                # 计算两个community的det_Q
                cur_Q = cal_det_Q(self._G_cloned, self._Group[index_i], self._Group[index_j])

                # 找到合并两个community Q值增加最大的进行合并
                if cur_Q > det_Q:
                    det_Q = cur_Q
                    max_edge = edge

            if max_edge is None: break

            # 合并两个community
            index_i = self._Node_GroupIndex[max_edge[0]]
            index_j = self._Node_GroupIndex[max_edge[1]]
            self._Group[index_i].extend(self._Group[index_j])
            for node in self._Group[index_j]:
                self._Node_GroupIndex[node] = index_i
            del self._Group[index_j]
            # 社区i和j已合并，可移除合并后的社区内的edge，减少后续遍历
            self._G.remove_edge(max_edge[0], max_edge[1])

            # 寻找Q值最大的划分方式
            components = copy.deepcopy(list(self._Group.values()))
            cur_Q = util.cal_Q(components, self._G_cloned)
            # cur_Q = util.cal_Q_mat(self._A_mat, list(self._Node_GroupIndex.values()))
            if cur_Q > self._max_Q:
                self._max_Q = cur_Q
                self._partition = components
            t2 = time.time()
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())   , end=" we find " )
            print("%d communities and the maximal Q is %.3f " % (len(self._partition,),self._max_Q))
            print(self._partition)

        return self._partition, self._max_Q


def cal_det_Q(G, partition_i, partition_j):
    m = len(list(G.edges()))
    a_i = 0
    for node in partition_i:
        a_i += len(list(G.neighbors(node)))
    a_i = a_i / float(2 * m)

    a_j = 0
    for node in partition_j:
        a_j += len(list(G.neighbors(node)))
    a_j = a_j / float(2 * m)

    e_ij = 0
    for i in range(len(partition_i)):
        for j in range(len(partition_j)):
            if G.has_edge(partition_i[i], partition_j[j]):
                e_ij += 1
    e_ij = e_ij / float(2 * m)
    # print(e_ij, a_i, a_j)
    return 2 * (e_ij - a_i * a_j)

def Start_FN(G):
    # FN算法 社团检测算法
    t1 = time.time()  # 返回当前时间的时间戳
    algo = FN(G)
    partition, max_Q = algo.execute()  # 得到划分的社团结果（数量）及最大Q值
    t2 = time.time()
    print("the result of partition is :",partition) #将分好的社区输出
    print("Terminate,we find %d communities after %.3f seconds, the maximal Q is %.3f! " % (len(partition), t2 - t1, max_Q))

    # 可视化社团检测结果
    if len(G.nodes()) < 2000:
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, font_weight='bold')
        plt.show()
        # plt.savefig('community.jpg')

def get_value(G, community):
    '''
    Each node gets respective value. Nodes in one community have same value
    community: 形如 [[1,2,3],[4,5],[6,7,8]]
    '''
    num_node = nx.number_of_nodes(G)
    value = [[] for i in range(num_node)]
    print("lenvalue",len(value))
    for index, com in enumerate(community): #同时列出数据和数据下标 ；index为社团数量 ；com是每个小社团，通过遍历每个小社团将相同社团节点赋值相同
        print("index,com:",index,com)
        for q in com:
            value[int(q)-1] = index   #我们的结点，从1开始排序，但是value的序号从0开始，所以-1
    print(value)
    return value

def draw_community(G, com):


    value = get_value(G, com)
    pos = nx.spring_layout(G)
    # pos = nx.shell_layout(G)   #环状图

    nx.draw(G,
            pos,
            cmap=plt.get_cmap('jet'),
            with_labels=False,
            node_color=value,
            node_size=20,
            width=0.2)
    plt.show()
    # plt.savefig('community.jpg')


if __name__ == '__main__':
    create_graph()
    # G = nx.read_gml(outfile+".gml")   #读取图文件，支持格式：txt，gml，csv（csv测试未通过）
    # information_graph(G)    #输出网络拓扑情况
    # draw(G)
    # Start_FN(G)
    # community=[['206', '207', '208', '212', '213', '215', '217', '218', '219', '244', '222', '226', '227', '233', '245', '246', '247', '225', '240', '241', '257', '242', '258', '243', '259', '275', '276', '256', '260', '261', '284', '262', '263', '277', '285', '294', '293', '295', '296', '297', '298', '303', '304', '286', '264', '265', '305', '299', '306', '287', '266', '278', '300', '267', '288', '279', '268', '301', '289', '280', '302', '269', '281', '270', '248', '249', '271', '290', '282', '291', '250', '272', '251', '283', '273', '292', '252', '234', '235', '236', '274', '253', '237', '228', '254', '229', '230', '238', '255', '239', '231', '484', '232', '223', '224', '483', '482', '220', '221', '481', '214', '216', '479', '480', '209', '478', '211'], ['307', '308', '311', '318', '317', '330', '331', '332', '338', '333', '339', '347', '334', '348', '340', '335', '349', '358', '341', '359', '350', '360', '336', '342', '351', '337', '352', '343', '323', '324', '319', '344', '325', '320', '312', '313', '321', '314', '309', '310'], ['493', '502'], ['584', '492', '510', '520', '529', '545', '372', '373', '361', '558', '559', '315', '316', '322', '326', '327', '328', '353', '362', '363', '374', '375', '384', '382', '2', '1', '6', '5', '16', '30', '46', '49', '53', '50', '58', '60', '66', '64', '72', '73', '65', '74', '75', '76', '94', '112', '130', '131', '144', '159', '171', '200', '201', '202', '203', '145', '146', '147', '148', '132', '133', '113', '149', '134', '114', '135', '150', '115', '116', '136', '151', '160', '137', '117', '152', '161', '153', '162', '118', '163', '95', '96', '119', '154', '164', '174', '173', '175', '172', '176', '165', '177', '183', '184', '178', '185', '192', '191', '193', '186', '166', '196', '197', '187', '179', '180', '188', '167', '138', '155', '181', '97', '120', '98', '121', '204', '189', '99', '77', '122', '78', '100', '92', '93', '91', '79', '123', '139', '101', '80', '168', '199', '198', '156', '81', '102', '140', '124', '141', '103', '125', '82', '104', '83', '67', '84', '68', '105', '126', '85', '142', '157', '127', '106', '143', '86', '158', '107', '69', '87', '108', '88', '70', '71', '61', '59', '62', '109', '63', '89', '51', '52', '205', '55', '54', '57', '56', '169', '47', '48', '31', '32', '33', '18', '17', '34', '35', '36', '37', '38', '39', '19', '20', '21', '40', '22', '7', '8', '9', '23', '41', '10', '24', '42', '11', '25', '12', '43', '26', '44', '13', '14', '27', '28', '15', '29', '45', '3', '4', '194', '110', '90', '128', '111', '355', '129', '367', '368', '356', '380', '369', '357', '389', '370', '390', '371', '381', '406', '210', '407', '170', '195', '391', '408', '421', '422', '423', '424', '425', '426', '439', '440', '427', '441', '458', '428', '442', '457', '182', '190', '456', '409', '443', '429', '410', '430', '392', '411', '444', '459', '431', '460', '445', '461', '446', '462', '432', '412', '393', '413', '394', '447', '433', '395', '414', '448', '434', '463', '415', '396', '383', '397', '449', '466', '435', '398', '450', '416', '399', '436', '417', '451', '400', '437', '418', '452', '438', '401', '419', '453', '420', '402', '403', '454', '385', '386', '376', '404', '377', '387', '405', '388', '378', '364', '365', '379', '650', '366', '644', '354', '645', '651', '639', '646', '455', '635', '329', '640', '652', '346', '636', '641', '647', '653', '648', '642', '637', '638', '622', '623', '624', '610', '611', '612', '603', '602', '613', '604', '605', '593', '594', '595', '581', '582', '583', '568', '569', '570', '596', '614', '606', '345', '467', '464', '468', '465', '688', '469', '470', '698', '689', '699', '708', '474', '472', '473', '471', '719', '477', '476', '475', '732', '700', '709', '690', '678', '733', '710', '701', '530', '546', '531', '521', '522', '720', '702', '679', '711', '734', '721', '735', '712', '703', '670', '722', '691', '680', '704', '692', '671', '681', '705', '693', '682', '694', '706', '695', '713', '707', '736', '723', '737', '672', '654', '751', '750', '752', '753', '754', '738', '724', '755', '763', '714', '756', '725', '739', '715', '764', '765', '778', '766', '740', '757', '779', '726', '716', '741', '767', '758', '780', '727', '768', '742', '759', '728', '769', '743', '760', '781', '770', '782', '744', '761', '771', '783', '772', '784', '762', '773', '786', '717', '745', '729', '774', '718', '746', '730', '747', '748', '731', '749', '777', '776', '775', '785', '787', '788'], ['657', '658', '659', '660', '661', '666', '585', '485', '486', '532', '487', '488', '489', '490', '491', '497', '498', '499', '500', '501', '528', '789', '790', '793', '794', '795', '796', '799', '800', '801', '802', '803', '807', '814', '815', '816', '808', '818', '819', '829', '830', '831', '838', '839', '841', '846', '847', '848', '849', '853', '854', '855', '865', '866', '867', '868', '869', '884', '885', '897', '1090', '1091', '1092', '916', '906', '905', '896', '895', '886', '904', '883', '894', '882', '893', '903', '915', '902', '914', '881', '913', '892', '880', '901', '891', '912', '879', '864', '870', '863', '871', '924', '862', '878', '861', '840', '842', '832', '828', '827', '843', '850', '833', '817', '844', '933', '826', '825', '932', '860', '931', '944', '943', '824', '837', '823', '813', '812', '806', '811', '805', '822', '810', '804', '836', '821', '798', '809', '792', '791', '509', '519', '544', '518', '508', '527', '820', '543', '835', '517', '557', '526', '834', '542', '507', '525', '516', '541', '556', '567', '515', '524', '540', '506', '566', '845', '539', '555', '514', '523', '505', '554', '513', '538', '565', '579', '580', '578', '564', '577', '553', '576', '563', '552', '537', '512', '575', '536', '562', '551', '574', '592', '561', '591', '590', '573', '496', '511', '535', '550', '589', '601', '572', '504', '495', '503', '534', '549', '588', '560', '600', '587', '548', '571', '533', '494', '851', '852', '547', '586', '599', '598', '609', '597', '608', '618', '607', '617', '616', '615', '627', '626', '628', '625', '629', '797', '856', '857', '872', '873', '874', '875', '876', '887', '898', '621', '619', '620', '632', '631', '630', '858', '859', '877', '890', '889', '888', '900', '899', '907', '910', '909', '908', '920', '917', '919', '918', '921', '926', '911', '927', '922', '929', '928', '937', '936', '938', '939', '923', '940', '930', '941', '942', '945', '946', '947', '957', '948', '962', '956', '961', '955', '935', '960', '954', '959', '963', '970', '969', '953', '958', '925', '934', '952', '967', '968', '976', '977', '978', '975', '974', '979', '980', '966', '951', '965', '973', '949', '950', '634', '633', '669', '668', '964', '662', '677', '667', '971', '972', '676', '675', '687', '686', '685', '697', '674', '684', '665', '696', '673', '683', '664', '663', '649', '656', '643', '655'], ['981', '982', '984', '985', '987', '990', '995', '996', '997', '1005', '1006', '991', '1007', '1018', '992', '998', '1000', '994', '1112', '1113', '1126', '1002', '1013', '1001', '1012', '1011', '1024', '1023', '1038', '1032', '1037', '1010', '1031', '1036', '1022', '1021', '1035', '1009', '1008', '1034', '1030', '1042', '1043', '1041', '1048', '1047', '1046', '1045', '1049', '1053', '1044', '1057', '1029', '1058', '1054', '1020', '999', '1040', '1033', '1051', '1028', '993', '1019', '1027', '1039', '1050', '1026', '1025', '1017', '1016', '1004', '1015', '1003', '1014', '1055', '1067', '1065', '1066', '1064', '1059', '1068', '1069', '1071', '1070', '1073', '1072', '1074', '1075', '1076', '1060', '1179', '1180', '1192', '1173', '1187', '1080', '1193', '1079', '1081', '1201', '1078', '1084', '1085', '1202', '1188', '1194', '1181', '1212', '1203', '1213', '1225', '1087', '1226', '1214', '1227', '1086', '1083', '1077', '1204', '1195', '1228', '1215', '1235', '1234', '1236', '1229', '1196', '1205', '1216', '1182', '1189', '1174', '1230', '1197', '1206', '1183', '1217', '1237', '1089', '1242', '1243', '1244', '1241', '1175', '1082', '1088', '1161', '1176', '1162', '1184', '1190', '1198', '1185', '1207', '1218', '1177', '1191', '1199', '1208', '1178', '1163', '1164', '1158', '1159', '1250', '1200', '1209', '1150', '1151', '1210', '1219', '1220', '1152', '1052', '1063', '1056', '1062', '1061', '1238', '988', '989', '986', '983', '1153'], ['1093', '1094', '1097', '1095', '1096'], ['1098', '1099', '1102', '1103', '1131', '1138', '1105', '1104', '1106', '1107', '1114', '1115', '1116', '1117', '1122', '1123', '1127', '1128', '1132', '1133', '1134', '1141', '1142', '1124', '1125', '1304', '1305', '1307', '1309', '1310', '1130', '1306', '1129', '1137', '1308', '1136', '1144', '1311', '1148', '1135', '1149', '1147', '1143', '1157', '1156', '1155', '1154', '1166', '1167', '1165', '1169', '1170', '1171', '1172', '1140', '1121', '1118', '1108', '1109', '1120', '1110', '1119', '1111', '1146', '1139', '1145', '1312', '1313', '1100', '1101', '1160', '1168', '1186'], ['1221', '1222', '1281', '1211', '1223', '1282', '1224', '1231', '1328', '1232', '1245', '1239', '1246', '1247', '1248', '1249', '1253', '1275', '1264', '1329', '1330', '1331', '1336', '1337', '1314', '1316', '1317', '1319', '1320', '1321', '1326', '1489', '1490', '1493', '1492', '1491', '1494', '1327', '1325', '1343', '1358', '1342', '1365', '1366', '1495', '1375', '1384', '1324', '1323', '1322', '1318', '1333', '1332', '1315', '1341', '1340', '1351', '1339', '1338', '1350', '1349', '1357', '1364', '1348', '1363', '1356', '1347', '1362', '1355', '1361', '1346', '1354', '1345', '1353', '1335', '1344', '1334', '1352', '1372', '1360', '1371', '1374', '1373', '1370', '1383', '1382', '1381', '1369', '1380', '1393', '1392', '1391', '1379', '1390', '1368', '1400', '1378', '1389', '1401', '1402', '1399', '1388', '1403', '1413', '1412', '1414', '1411', '1398', '1377', '1387', '1410', '1397', '1427', '1429', '1428', '1415', '1430', '1404', '1394', '1409', '1426', '1425', '1386', '1396', '1376', '1408', '1438', '1439', '1440', '1431', '1441', '1416', '1405', '1432', '1417', '1442', '1461', '1460', '1462', '1443', '1463', '1433', '1464', '1444', '1459', '1480', '1424', '1434', '1465', '1445', '1406', '1458', '1385', '1367', '1395', '1407', '1423', '1457', '1479', '1437', '1280', '1271', '1256', '1263', '1359', '1270', '1289', '1274', '1279', '1262', '1255', '1278', '1273', '1254', '1269', '1288', '1261', '1456', '1272', '1277', '1268', '1287', '1260', '1294', '1455', '1252', '1259', '1267', '1251', '1240', '1293', '1454', '1286', '1300', '1299', '1476', '1478', '1477', '1475', '1303', '1484', '1486', '1485', '1487', '1258', '1266', '1302', '1298', '1257', '1265', '1285', '1297', '1301', '1292', '1233', '1435', '1446', '1418', '1284', '1296', '1466', '1447', '1467', '1436', '1419', '1448', '1468', '1449', '1481', '1469', '1482', '1450', '1470', '1483', '1488', '1471', '1451', '1420', '1452', '1472', '1473', '1421', '1453', '1422', '1474', '1295', '1283', '1291', '1290', '1276']]

    # draw_community(G, community)    #画出彩色的社团检测结果，这个G是上边加载的gml图文件，community就是得出的社团检测结果 list
    # draw(G)




