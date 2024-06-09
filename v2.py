from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F

## 超参数
torch.manual_seed(1337)
batch_size=64
block_size=256
max_iters=5000
eval_interval=500
learning_rate=3e-4
device="cuda" if torch.cuda.is_available() else "cpu"
eval_iters=200
n_embd=384
n_head=6
n_layer=6
dropout=0.2

## 函数
def get_batch(split):
    data=train_data if split=="train" else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y=x.to(device),y.to(device)
    return x,y
@torch.no_grad()
def estimate_loss():
    out={}
    m.eval()
    for split in ["train","val"]:
        losses=torch.zeros(eval_iters)

        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits,loss=m(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    m.train()
    return out



## 读取文本数据
with open("input.txt","r",encoding="utf-8") as f:
    texts=f.read()  
chars=sorted(list(set(texts)))
vocab_size=len(chars)
stoi={char:i  for i,char in enumerate(chars)}
itos={i:char  for i,char in enumerate(chars)}
encode=lambda s: [stoi[c] for c in s]
decode=lambda l: "".join([itos[i] for i in l])

## tictoken
# import tiktoken
# enc=tiktoken.get_encoding("gpt2")
# print(enc.n_vocab)
# print(enc.encode("hi there"))
# print(enc.decode(enc.encode("hi there")))

data=torch.tensor(encode(texts),dtype=torch.long)
# print(data.shape)

## split
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

xb,yb=get_batch("train")

class LayerNorm(nn.Module):
    def __init__(self,dim,eps=1e-3) -> None:
        super().__init__()
        self.eps=eps
        self.gamma=torch.ones(dim,device=device)
        self.beta=torch.zeros(dim,device=device)
    def forward(self,x):
        # print("layer norm input shape",x.shape)
        x_mean=x.mean(1,keepdim=True)
        x_var=x.var(1,keepdim=True)
        x_hat=(x-x_mean)/torch.sqrt(x_var+self.eps)
        self.out=self.gamma*x_hat+self.beta
        return self.out 
    
    def parameters(self):
        return [self.gamma,self.beta]
class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key=nn.Linear(n_embd,head_size)
        self.query=nn.Linear(n_embd,head_size)
        self.value=nn.Linear(n_embd,head_size)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size,device=device)))
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        B,T,C=x.shape
        q=self.query(x)
        k=self.key(x)
        v=self.value(x)

        wei=q@k.transpose(-1,-2)*C**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float("-inf"))
        wei=F.softmax(wei,dim=-1)
        wei=self.dropout(wei)
        out=wei@v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head,head_size) -> None:
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj=nn.Linear(n_embd,n_embd)
    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.proj(out)
        return out



class FeedForward(nn.Module):
    def __init__(self,n_embd) -> None:
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        x=self.net(x)
        return x

class Block(nn.Module):
    def __init__(self,n_embd,n_head) -> None:
        super().__init__()
        self.sa_head=MultiHeadAttention(n_head,n_embd//n_head)
        self.ffwd=FeedForward(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)

    def forward(self,x):
        x=x+self.sa_head(self.ln1(x))
        x=x+self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        self.blocks=nn.Sequential(
            *[Block(n_embd,n_head) for _ in range(n_layer)]
        )
        self.ln_f=nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)
    
    def forward(self,idx,targets=None):
        B,T=idx.shape

        tok_embd=self.token_embedding_table(idx)
        pos_embd=self.position_embedding_table(torch.arange(T,dtype=torch.long,device=device))
        x=tok_embd+pos_embd
        x=self.blocks(x)
        x=self.ln_f(x)
    
        logits=self.lm_head(x)
        
       
        if targets is None:
            losses=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            losses=F.cross_entropy(logits,targets)
        return logits,losses
    
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-block_size:]
            # print(idx_cond.shape)

            logits,loss=self(idx_cond)
            logits=logits[:,-1,:] 
            # print(logits.shape)
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx
    


m=BigramLanguageModel().to(device)

idx=torch.zeros((1,1),dtype=torch.long,device=device)

print(decode(m.generate(idx,100)[0].tolist()))


## optimizer
optimizer=torch.optim.AdamW(m.parameters(),lr=learning_rate)


for iter in range(max_iters):

    if iter%eval_interval==0:
        losses=estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f} val loss {losses['val']:.4f}")

    xb,yb=get_batch("train")
    logits,loss=m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


print(decode(m.generate(idx,100)[0].tolist()))



