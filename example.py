import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

(?# tokenizer = AutoTokenizer.from_pretrained(r"dmayhem93/toolformer_v0_epoch2"))
tokenizer = AutoTokenizer.from_pretrained(r"/scratch/yerong/.cache/huggingface/transformers/models--dmayhem93--toolformer_v0_epoch2/snapshots/ccc07ac82b35778a218a806bb712326a53e5c5f9"))

model = AutoModelForCausalLM.from_pretrained(
    r"dmayhem93/toolformer_v0_epoch2",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).cuda()
generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device=0
) 


