from typing import Iterator
import torch
import torch.nn as nn
from torch.nn import functional as F

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ["train","val"]:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out

## 超参数
batch_size=64
block_size=256
max_iter=3000
eval_interval=300
learning_rate=3e-4
device="cuda" if torch.cuda.is_available() else "cpu"
eval_iters=200
n_embd=128
n_head=4
n_layer=4
dropout=0.2


torch.manual_seed(1337)


## 读取文本数据
with open("more.txt","r",encoding="utf-8") as f:
    text=f.read()

print("Length of dataset:",len(text))

## 获取字符
chars=  sorted(list(set(text)))
vocab_size=len(chars)

# print("vocab chars:","".join(chars))
# print("vocab size:",vocab_size)


stoi={ j:i for i,j in enumerate(chars)}
itos={ i:j for i,j in enumerate(chars)}

encode=lambda s: [ stoi[i] for i in s]
decode=lambda l: "".join([itos[i] for i in l])

# print(encode("hi there"))
# print(decode(encode("hi there")))

data=torch.tensor(encode(text),dtype=torch.long)

# print(data.shape,data.dtype)

# print(data[:100])

## split train/val set
n=int(0.9*len(data))

train_data=data[:n]
val_data=data[n:]


x=train_data[:block_size]
y=train_data[1:block_size+1]

# for t in range(block_size):
#     context=x[:t+1]
#     target=y[t]
#     print(f"when input is {context} the target is {target}")

def get_batch(split):
    data= train_data if split=="train" else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))

    x=torch.stack([  data[i:i+block_size] for i in ix])
    y=torch.stack([  data[i+1:i+block_size+1] for i in ix])
    x,y=x.to(device),y.to(device)
    return x,y

xb,yb=get_batch("train")

# print("inputs:")
# print(xb.shape)
# print(xb)

# print("targets:")
# print(yb.shape)
# print(yb)

# print("------")
# for b in range(batch_size):
#     for t in range(block_size):
#         context=xb[b,:t+1]
#         target=yb[b,t]
#         print(f"when input is {context.tolist()} the target is {target}")

## define model
class LayerNorm(nn.Module):
    def __init__(self,dim,eps=1e-5,momentum=0.1) -> None:
        super().__init__()
        self.eps=eps
        self.gamma=torch.ones(dim,device=device)
        self.beta=torch.zeros(dim,device=device)

    def forward(self,x):
        print("layer norm input shape",x.shape)
        xmean=x.mean(1,keepdim=True)
        xvar=x.var(1,keepdim=True)
        xhat=(x-xmean)/torch.sqrt(xvar+self.eps)
        self.out=self.gamma*xhat+self.beta
        return self.out
    def parameters(self):
        return [self.gamma,self.beta]
    



class FeedForward(nn.Module):
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_embd,4*n_embd),
                               nn.ReLU(),
                               nn.Linear(4*n_embd,n_embd),
                               nn.Dropout(dropout))
    def forward(self,x):
        return self.net(x)


class head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key=nn.Linear(n_embd,head_size)
        self.query=nn.Linear(n_embd,head_size)
        self.value=nn.Linear(n_embd,head_size)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        B,T,C=x.shape

        k=self.key(x)
        q=self.query(x)

        wei=q@k.transpose(-1,-2)*C**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=F.softmax(wei,dim=-1)
        wei=self.dropout(wei)

        v=self.value(x)
        out=wei@v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self,num_head, head_size) -> None:
        super().__init__()
        self.heads=nn.ModuleList([ head(head_size) for _ in range(num_head)])
        self.proj=nn.Linear(n_embd,n_embd)


    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.proj(out)
        return out

class block(nn.Module):
    def __init__(self,n_embd,n_head) -> None:
        super().__init__()
        head_size=n_embd//n_head
        self.sa=MultiHeadAttention(n_head,head_size)
        self.ffwd=FeedForward(n_embd)
        self.ln1=LayerNorm(n_embd)
        self.ln2=LayerNorm(n_embd)
    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.ffwd(self.ln2(x))
        ## 这里可能还要再经过一个layernorm层输出 
        return x 
    
    
class BigramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        # self.blocks=nn.Sequential(
        #     block(n_embd,n_head=4),
        #     block(n_embd,n_head=4),
        #     block(n_embd,n_head=4),
        #     nn.LayerNorm(n_embd)
        # )
        self.blocks=nn.Sequential(*[ block(n_embd,n_head=n_head) for _ in range(n_layer)])
        # self.sa_head=head(n_embd)
        self.sa_head=MultiHeadAttention(4,n_embd//4)
        self.ffwd=FeedForward(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)



    def forward(self,idx,targets=None):
        B,T=idx.shape
        tok_emb=self.token_embedding_table(idx)
        pos_emb=self.position_embedding_table(torch.arange(T,device=device))
        x=tok_emb+pos_emb
        x=self.blocks(x)
        x=self.ffwd(x)
        logits=self.lm_head(x)
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)

        return logits,loss
    
    def generate(self,idx,max_new_tokens):

        for _ in range(max_new_tokens):
            idx_cond=idx[:,-block_size:]


            logits,loss=self(idx_cond)
            logits=logits[:,-1,:]

            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx





model=BigramLanguageModel().to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)

for iter  in range(max_iter):
    if iter%eval_interval==0:
        losses=estimate_loss()
        print(f"step {iter}: train_loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb,yb=get_batch("train")

    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



## generate from model
print(decode(model.generate(idx=torch.zeros((1,1),dtype=torch.long).to(device),max_new_tokens=500)[0].tolist()))
