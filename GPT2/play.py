from transformers import GPT2LMHeadModel
import matplotlib.pyplot as plt
model_hf=GPT2LMHeadModel.from_pretrained("gpt2")
sd_hf=model_hf.state_dict()

for k,v in sd_hf.items():
    print(k,v.shape)


# plt.imshow(sd_hf["transformer.wpe.weight"],cmap="gray")
# plt.show()

# plt.plot(sd_hf["transformer.wpe.weight"][:,150])
# plt.plot(sd_hf["transformer.wpe.weight"][:,200])
# plt.plot(sd_hf["transformer.wpe.weight"][:,250])
# plt.show()

# plt.imshow(sd_hf["transformer.h.1.attn.c_attn.weight"][:300,:300],cmap="gray")
# plt.show()

from transformers import pipeline,set_seed

generator=pipeline("text-generation",model="gpt2")
set_seed(42)
print(generator("hello, I am a language model",max_length=30,num_return_sequences=5))

