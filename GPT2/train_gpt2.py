from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import inspect
@dataclass
class GPTConfig:
    block_size:int =1024
    vocab_size:int=50257
    n_layer:int=12
    n_head:int=12
    n_embd:int=768
class CausalSelfAttention(nn.Module):
    def __init__(self,config:GPTConfig) -> None:
        super().__init__()
        assert config.n_embd%config.n_head==0

        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1

        self.n_head=config.n_head
        self.n_embd=config.n_embd
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))
    def forward(self,x):
        B,T,C=x.shape

        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd,dim=2)

        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)## (B,nh,T,hs)
        
        # attn=(q@v.transpose(-1,-2)*(1.0/math.sqrt(k.size(-1))))
        # attn=attn.masked_fill(self.bias[:,:,:T,:T]==0,float("-inf"))
        # attn=F.softmax(attn,dim=-1)
        # y=attn@v
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)

        y=y.transpose(1,2).contiguous().view(B,T,C)
        y=self.c_proj(y)

        return y




class MLP(nn.Module):
    def __init__(self,config:GPTConfig) -> None:
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu=nn.GELU(approximate="tanh")
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x
    

class Block(nn.Module):
    def __init__(self,config:GPTConfig) -> None:
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)
    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self,config:GPTConfig) -> None:
        super().__init__()
        self.config=config

        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            wpe=nn.Embedding(config.block_size,config.n_embd),
            h=nn.ModuleList( Block(config) for _ in range(config.n_layer)),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        
        # weight sharing schema
        self.transformer.wte.weight=self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            std=0.02
            if hasattr(module,"NANOGPT_SCALE_INIT"):
                std*=(2* self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module,nn.Embedding):
                torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,idx,targets=None):
        B,T=idx.shape

        assert T<=self.config.block_size,f"cannot forward sequence of length {T},block size is {self.config.block_size}"
        pos=torch.arange(0,T,dtype=torch.long ,device=idx.device)
        pos_embd=self.transformer.wpe(pos)
        tok_embd=self.transformer.wte(idx)
        x=tok_embd+pos_embd

        for block in self.transformer.h:
            x=block(x)

        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))

        return logits,loss
    def configure_optimizers(self,weight_decay,learning_rate,device):
        param_dict={pn:p for pn,p in self.named_parameters()}
        param_dict={pn:p for pn,p in param_dict.items() if p.requires_grad}
        decay_params=[p for n,p in param_dict.items() if p.dim()>=2]
        nodecay_params=[p for n,p in param_dict.items() if p.dim()<2    ]
        optim_groups=[
            {"params":decay_params,"weight_decay":weight_decay},
            { "params":nodecay_params,"weight_decay":0.0}
        ]

        ## numel返回张量中元素的个数 
        num_decay_params=sum(p.numel() for p in decay_params)
        num_nodecay_params= sum(p.numel() for p in nodecay_params)
        print(f"num decayed paramter tensors: {len(decay_params)},with {num_decay_params:,} paramters")
        print(f"num non-decayed paramter tensors: {len(nodecay_params)},with {num_nodecay_params} paramters")
        
        ## 查看adamw中是否有fused参数，检查是否有融合优化
        fused_available="fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused=fused_available and "cuda" in device
        print(f"using fused AdamW {use_fused}")

        # 设定优化器
        optimizer=torch.optim.AdamW(optim_groups,lr=learning_rate,betas=(0.9,0.95),eps=1e-8,fused=use_fused)
        return optimizer


    @classmethod
    def from_pretrained(cls,model_type):
        assert model_type in {"gpt2","gpt2-medium","gpt2-large","gpt-xl"}
        from transformers import GPT2LMHeadModel
        print("loading weight from pretrained gpt: %s"%model_type)
        config_args={
            "gpt2":dict(n_layer=12,n_head=12,n_embd=768),
            "gpt2-medium":dict(n_layer=24,n_head=16,n_embd=1024),
            "gpt2-large":dict(n_layer=36,n_head=20,n_embd=1280),
            "gpt-xl":dict(n_layer=48,n_head=25,n_embd=1600),
        }[model_type]
        config_args["vocab_size"]=50257
        config_args["block_size"]=1024
        config=GPTConfig(**config_args)
        model=GPT(config)
        sd=model.state_dict()
        sd_keys=sd.keys()
        sd_keys=[k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf=GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf=model_hf.state_dict()


        sd_keys_hf=sd.keys()
        sd_keys_hf=[ k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")] 
        sd_keys_hf=[ k for k in sd_keys_hf if not k.endswith(".attn.bias")] 
        transposed=["attn.c_attn.weight","attn.c_proj.weight","mlp.c_fc.weight","mlp.c_proj.weight"]

        assert len(sd_keys_hf)==len(sd_keys), f"mismatched keys: {len(sd_keys_hf)}!={len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1]==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape ==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
#数据加载器
import tiktoken
class DataLoaderLite:
    def __init__(self,B,T,process_rank,num_processes):
        self.B=B
        self.T=T
        self.process_rank=process_rank
        self.num_processes=num_processes


        with open("input.txt","r",encoding="utf-8") as f:
            text=f.read()
        enc=tiktoken.get_encoding("gpt2")
        tokens=enc.encode(text)
        self.tokens=torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)//(B*T)} batches")
        self.current_position=self.B*self.T*self.process_rank

    def next_batch(self):
        B,T =self.B,self.T
        buf=self.tokens[self.current_position:self.current_position+self.T*self.B+1]
        x=(buf[:-1]).view(B,T)
        y=buf[1:].view(B,T)
        self.current_position+=B*T*self.num_processes
        if self.current_position+(B*T*self.num_processes+1)>len(self.tokens):
            self.current_position=self.B*self.T*self.process_rank


        return x,y
    
from torch.distributed import init_process_group,destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
ddp=int(os.environ.get("RANK",-1))!=-1

if ddp:
    assert torch.cuda.is_available(), "for now i think we need cuda for ddp"
    init_process_group(backend="nccl")
    ddp_rank=int(os.environ["RANK"])
    ddp_local_rank=int(os.environ["LOCAL_RANK"])
    ddp_world_size=int(os.environ["WORLD_SIZE"])

    device=f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device=device)
    master_process=ddp_rank==0
else:
    ddp_rank=0
    ddp_local_rank=0
    ddp_world_size=1
    master_process=True
    device="cpu"

    if torch.cuda.is_available():
        device="cuda"
    elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
        device="mps"
    print(f"using device :{device}")


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

num_return_sequences=5
max_length=30
model=GPT(config=GPTConfig())
model.eval()
model.to(device)
# model=torch.compile(model)
if ddp:
    model=DDP(model,device_ids=[ddp_local_rank])
raw_model=model.module if ddp else model 


total_batch_size=524288
B=4
T=1024
assert total_batch_size%(B*T*ddp_world_size)==0,"make sure total batch size is divisible by B*T*ddp_world_size"
grad_accum_steps=total_batch_size//(B*T*ddp_world_size)


if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps:{grad_accum_steps}")

print("I am GPU ",ddp_rank)

train_loader=DataLoaderLite(B=B,T=T,process_rank=ddp_local_rank,num_processes=ddp_world_size)
torch.set_float32_matmul_precision("high")
# model=torch.compile(model)


optimizer=raw_model.configure_optimizers(weight_decay=0.1,learning_rate=6e-4,device=device)

max_lr=6e-4
min_lr=max_lr*0.1
warmup_steps=10
max_steps=50

def get_lr(it):
    if it<warmup_steps:
        return max_lr*(it+1)/warmup_steps
    if it>max_steps:
        min_lr
    decay_ratio=(it-warmup_steps)/(max_steps-warmup_steps)
    assert 0<=decay_ratio<=1
    coeff=0.5*(1.0+math.cos(math.pi*decay_ratio))
    return min_lr+coeff*(max_lr-min_lr)



for step in range(max_steps):
    t0=time.time()
    optimizer.zero_grad()
    loss_accum=0.0
    for micro_step in range(grad_accum_steps):
        x,y=train_loader.next_batch()
        x,y=x.to(device),y.to(device)
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            logits,loss=model(x,y)
        loss=loss/grad_accum_steps
        loss_accum+=loss.detach()
        if ddp:
            model.require_backward_grad_sync(micro_step==grad_accum_steps-1)
        loss.backward() 
    ##  在所有rank上平均loss_accum
    if ddp:
        dist.all_reduce(loss_accum,op=dist.ReduceOp.AVG)

    ## 用于裁剪（clip）模型中参数的梯度范数。
    # 它的主要作用是在反向传播过程中限制（或裁剪）模型参数的梯度大小，以防止梯度爆炸（gradient explosion）问题，从而提高训练过程的稳定性和效率。梯度爆炸可能导致学习过程不稳定，参数更新幅度过大，使得模型难以收敛。
    #该函数的工作原理是计算模型参数梯度的范数（默认为2范数，即L2范数），如果计算出的范数超过了给定的最大值（max_norm参数），则将所有参数的梯度按比例缩小，以确保整体梯度范数不超过这个最大阈值。如果梯度范数小于或等于max_norm，则梯度不会被改变。
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    lr=get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"]=lr
    optimizer.step()

    torch.cuda.synchronize() ## wait GPU to finish work
    t1=time.time()
    dt=(t1-t0)*1000

    token_processed=train_loader.B*train_loader.T*grad_accum_steps*ddp_world_size
    tokens_per_sec=token_processed/(t1-t0)

    if master_process:
        print(f"step {step}| loss: {loss.item():.6f} | lr:{lr:.4e}| norm:{norm:.4f}| dt:{dt:.2f} ms|tok/sec: {tokens_per_sec:.2f}")


if ddp:
    destroy_process_group()

# tokens=enc.encode("hello, I am a language model,")

# tokens=torch.tensor(tokens,dtype=torch.long)
# tokens=tokens.unsqueeze(0).repeat(num_return_sequences,1)

# x=tokens.to("cuda")

# print(x.shape)


# while x.size(1)<max_length:
#     with torch.no_grad():
#         logits=model(x)
#         logits=logits[:,-1,:]
#         probs=F.softmax(logits,dim=-1)
#         topk_probs,topk_indices=torch.topk(probs,50,dim=-1)

#         ix=torch.multinomial(topk_probs,1)
#         xcol=torch.gather(topk_indices,-1,ix)

#         x=torch.cat([x,xcol],dim=1)

# for i in range(num_return_sequences):
#     tokens=x[i,:max_length].tolist()

#     decoded=enc.decode(tokens)
#     print(">",decoded)


