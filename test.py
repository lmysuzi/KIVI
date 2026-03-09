# LLaMA model with KIVI
import warnings
warnings.filterwarnings("ignore")
import time
import torch
import json
from transformers import LlamaConfig, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

import quant.quant_flash_decode as kernel

enable_kivi = True
enable_new_kernel = False
#model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_path = "lmsys/longchat-7b-v1.5-32k"
config = LlamaConfig.from_pretrained(model_path)
config.k_bits = 2 # KiVi currently support 2/4 K/V bits
config.v_bits = 2
config.group_size = 32 
config.residual_length = 32 # corresponding to the number of recent fp16 tokens
config.use_flash = True # use flash-attention with KiVi for long context inference
CACHE_DIR = "/scratch/cached_model"

if enable_kivi:
    if enable_new_kernel:
        from models.new_llama_kivi import LlamaForCausalLM_KIVI
    else:
        from models.llama_kivi import LlamaForCausalLM_KIVI
    model = LlamaForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path=model_path,
        config=config,
        # cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).cuda()
else:
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        config=config,
        # cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    ).cuda()

enc = AutoTokenizer.from_pretrained(
    #"meta-llama/Llama-3.1-8B-Instruct", 
    model_path,
    use_fast=False, 
    trust_remote_code=True,)

model.eval()

file_name = "prompts.jsonl"

for line in open(file_name, "r"):
    example = json.loads(line)
    raw_prompt = example["input"]
    # longchat-7b-v1.5-32k 基于 Vicuna v1.5，使用 Vicuna 对话模板
    # [INST]...[/INST] 是 Llama-2-chat 的格式，用错会立即输出 EOS
    prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        f"USER: {raw_prompt} ASSISTANT:"
    )
    input_ids = enc(prompt, return_tensors="pt").input_ids.cuda()
    print( "-----------------------------------" )
    print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )

    # === Warmup: 触发 Triton JIT 编译（不计入 TTFT） ===
    print("\n[warmup] Triggering Triton JIT compilation...")
    with torch.no_grad():
        _ = model.generate(input_ids[:, :64], max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()
    print("[warmup] Done.")

    # === TTFT (Time To First Token) 测量：单独跑一次 max_new_tokens=1 ===
    torch.cuda.synchronize()
    t_ttft_start = time.perf_counter()
    _ = model.generate(input_ids, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t_ttft_start

    # === 完整生成 + 吞吐量测量 ===
    torch.cuda.synchronize()
    t_gen_start = time.perf_counter()
    tokens = model.generate(input_ids, max_new_tokens=1000, do_sample=False)
    torch.cuda.synchronize()
    t_gen_end = time.perf_counter()

    new_tokens = tokens.shape[1] - input_ids.shape[1]
    total_gen_time = t_gen_end - t_gen_start
    throughput = new_tokens / total_gen_time if total_gen_time > 0 else float("inf")

    output_text = enc.batch_decode(tokens[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print( f"Output Text: {output_text}" )

    print( f"TTFT:       {ttft * 1000:.2f} ms" )
    print( f"Throughput: {throughput:.2f} tokens/s  "
           f"({new_tokens} new tokens in {total_gen_time * 1000:.2f} ms)" )
    
    print(kernel.total_time)