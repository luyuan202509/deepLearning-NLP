import torch 
from torch import nn

class Embedding2(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding2, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.pos_embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        return self.embedding(x) + self.pos_embedding(x)

class EBD(nn.Module):
    def __init__(self,*args,**kwargs)->None:
        super(EBD, self).__init__(*args,**kwargs)
        self.embedding = nn.Embedding(28, 24)
        self.pos_embedding = nn.Embedding(12, 24)
        self.pos_t = torch.arange(0, 12).reshape(1,12)

    def forward(self, x:torch.Tensor):
        return self.embedding(x) + self.pos_embedding(self.pos_t)
        
def attention(Q,K,V):
    #A = Q @ K.T / (K.shape[1] ** 0.5) 这种转置只能使用到2维，三维以上需要使用transpose(-1,-2)
    A = Q @ K.transpose(-1,-2) / (K.shape[-1] ** 0.5)
    A =torch.softmax(A,dim=-1)
    Z = A @ V
    return Z

class Transformer_block(nn.Module):
    def __init__(self, *args,**kwargs)->None:
        super(Transformer_block, self).__init__(*args,**kwargs)
        self.W_q = nn.Linear(24, 24,bias=False)
        self.W_k = nn.Linear(24, 24,bias=False)
        self.W_v = nn.Linear(24, 24,bias=False)
        self.W_o = nn.Linear(24, 24,bias=False)
    
    def forward(self,x:torch.Tensor):
        Q,K,V = self.W_q(x),self.W_k(x),self.W_v(x)
        Z = attention(Q,K,V)
        Z = self.W_o(Z)
        return Z



if __name__ == "__main__":  
    print("\n==词嵌入====================================================================\n")
    aaa = torch.ones((2,12)).long()
    embedding = EBD()
    aaa = embedding(aaa)
    print(aaa)
    print("\n==注意力机制====================================================================\n")
    atten_en = Transformer_block()
    aaa = atten_en(aaa)
    print(aaa.shape)
   
