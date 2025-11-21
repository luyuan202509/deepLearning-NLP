import torch 

a  = torch.tensor([[1,2,3],
                  [4,5,6],
                  [7,8,9]])

print(a.T)
print(a.transpose(-1,-2))