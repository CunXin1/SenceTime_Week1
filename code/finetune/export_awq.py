"""微调后导出:① 合并 LoRA 适配器  ② AWQ 4-bit 压缩(部署用)。

流程:finetune.py 训练出 LoRA 适配器 → 本脚本合并进基座 → AWQ 量化成 4-bit。

⚠️ 第①步合并 Mac 能做;第②步 AWQ 量化依赖 autoawq(**CUDA 专属**),
   Mac 跑不了,请在 4090 上执行。改 MODEL 可换 7B。
用法: conda activate llama_factory && python code/export_awq.py
"""
import os
import subprocess

# ========= 可配置区 =========
MODEL   = "./models/Qwen2.5-1.5B-Instruct"
ADAPTER = "./saves/Qwen2.5-1.5B-Instruct-identity_penguin-lora"  # finetune.py 的输出
MERGED  = "./saves/merged"                                # 合并后的完整模型
AWQ_OUT = "./saves/awq"                                   # AWQ 量化输出
# ===========================

# ① 合并 LoRA 适配器到基座,得到完整 fp16 模型(Mac 可做)
subprocess.run([
    "llamafactory-cli", "export",
    f"model_name_or_path={MODEL}",
    f"adapter_name_or_path={ADAPTER}",
    "template=qwen",
    f"export_dir={MERGED}",
    "export_size=2",
    "trust_remote_code=true",
], check=True)
print(f"✅ LoRA 已合并 -> {MERGED}")

# ② AWQ 4-bit 量化(需 CUDA + `pip install autoawq`,请在 4090 上运行)
try:
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    model = AutoAWQForCausalLM.from_pretrained(MERGED)
    tok = AutoTokenizer.from_pretrained(MERGED, trust_remote_code=True)
    # AWQ 用少量校准数据统计激活分布,再按分布做 4-bit 量化
    model.quantize(tok, quant_config={
        "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"})
    model.save_quantized(AWQ_OUT)
    tok.save_pretrained(AWQ_OUT)
    print(f"✅ AWQ 4-bit 已保存 -> {AWQ_OUT}")
except ImportError:
    print("⚠️ 未安装 autoawq(CUDA 专属)。AWQ 压缩请在 4090 上:pip install autoawq 后重跑。")
