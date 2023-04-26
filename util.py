# -*- coding: utf-8 -*-
# @Author: 武辛
# @Email: geo_data_analysis@163.com
# @Note: 如有疑问，可加微信"wxid-3ccc"
# @All Rights Reserved!



import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics

# 加载网络
def load_graph(path):
	if ".txt" in path:
		# 可自行读取建网
		# G = nx.Graph()
		# with open(path) as text:
		# 	for line in text:
		# 		vertices = line.strip().split(" ")
		# 		source = int(vertices[0])
		# 		target = int(vertices[1])
		# 		G.add_edge(source, target)

		# 用函数读txt建网
		G = nx.read_edgelist(path, create_using = nx.Graph())
	elif ".gml" in path:
		G = nx.read_gml(path)
	elif ".csv" in path:
		G = nx.read_edgelist(path, create_using = nx.Graph())
	return G

# 克隆
def clone_graph(G):
	cloned_graph = nx.Graph()
	for edge in G.edges():
		cloned_graph.add_edge(edge[0], edge[1])
	return cloned_graph

# 计算Q值
def cal_Q(partition, G):
	m = len(list(G.edges()))
	a = []
	e = []

	# 计算每个社区的a值
	for community in partition:
		t = 0
		for node in community:
			t += len(list(G.neighbors(node)))
		a.append(t / float(2 * m))

	# 计算每个社区的e值
	for community in partition:
		t = 0
		# for node in community:
		# 	for neighbor_node in G.neighbors(node):
		# 		if neighbor_node in community:
		# 			t += 1
		for i in range(len(community)):
			for j in range(len(community)):
				if i != j:
					if G.has_edge(community[i], community[j]):
						t += 1
		e.append(t / float(2 * m))

	# 计算Q
	q = 0
	for ei, ai in zip(e, a):
		q += (ei - ai ** 2)

	return q