from modules.cluster import Cluster
from tools.configreader import absolute_path
import torch
import numpy as np

cluster = Cluster()

file_path = 'data/test.txt'

res = cluster.process(absolute_path(file_path))
print(res)
