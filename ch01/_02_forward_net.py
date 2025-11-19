import numpy as np
from collections import OrderedDict
from common.optimizer import SGD


class Sigmoid:
    """
    Sigmoid 层没有需要学习的参数,使用空列表来初始化实例变量 params
    """
    def __init__(self): 
        self.params = []
    def forward(self,x):
        return 1 / (1 + np.exp(-x)) 
    
    def backward(self,dout):
        """"
        sigmoid 层的反向传播 
        dy/dx = y * (1 - y)
        """
        return dout * (1 - self.out) * self.out


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
    def forward(self, x):
        W,b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out
    
    def backward(self,dout):
        W,b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


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



class MatMul:
    """"矩阵乘法层"""
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
    def forward(self,x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out
    
    def backward(self,dout):
        W, = self.params
        dx  = np.dot(dout, W.T)
        dW  = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx 

def main1():
    x = np.random.rand(10,2)
    print("输入数据：",x)
    model = TwoLayerNet(2,4,3)
    s = model.predict(x)
    print("输出推理结果：",s)

def main2():
    model = TwoLayerNet(2,4,3)
    x = np.random.rand(10,2)
    t = np.random.rand(10,3)
    optimizer = SGD()


if __name__ == "__main__":
    main1()
    main2()
   