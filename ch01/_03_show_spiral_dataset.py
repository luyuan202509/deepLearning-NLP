import sys 
sys.path.append('..') # 为了引入父目录的文件而进行的设定

from dataset.spiral import load_data
import matplotlib.pyplot as plt

x,t = load_data()
print(x.shape)
print(t.shape)

print(x)
print(t)



