import numpy as np
from collections import OrderedDict


class Sigmoid:
    """
    Sigmoid 层没有需要学习的参数,使用空列表来初始化实例变量 params
    """
    def __init__(self): 
        self.params = []
    def forward(self,x):
        return 1 / (1 + np.exp(-x)) 

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
    def forward(self, x):
        W,b = self.params
        out = np.dot(x, W) + b
        return out

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I,H,O = input_size, hidden_size, output_size  #输入层 2 、隐藏层 4、输出层 3 
        # 初始化权重和偏置
        W1 = np.random.randn(I,H) 
        b1 = np.random.randn(H) 
        W2 = np.random.randn(H,O) 
        b2 = np.random.randn(O) 

        # 生成层
        self.layers = [
            Affine(W1,b1),
            Sigmoid(),
            Affine(W2,b2)
        ]

        # 将所有的权重整理到列表中
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
    def predict(self,x):
        """"前向传播，依次通过各层"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

if __name__ == "__main__":
    x = np.random.rand(10,2)
    print("输入数据：",x)
    model = TwoLayerNet(2,4,3)
    s = model.predict(x)
    print("输出推理结果：",s)
   