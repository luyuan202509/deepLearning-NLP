import sys 
sys.path.append('..') 

import numpy as np
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
from common.optimizer import SGD
import time

# 设置超参数
max_epoch = 300
batch_size = 30
hidden_size = 10 
learning_rate = 1.0



#读入数据,生成模型和优化器
x, t = spiral.load_data()
model = TwoLayerNet(input_size=2,hidden_size=hidden_size,output_size=3)
optimizer = SGD(lr=learning_rate)

# 学习用的变量
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count =0
loss_list = []
start_time = time.time()

for epoch in range(max_epoch):
    # 打乱数据
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]

        # 计算梯度，更新参数
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        # 定期输出学习过程
        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| epoch %d |  iter %d / %d | loss %.2f' % (epoch + 1, iters + 1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0



elapsed_time = time.time() - start_time
print(f"训练时间： {elapsed_time}[s]")
plt.plot(loss_list)
plt.show()

