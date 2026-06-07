'''
专门针对 DeepSeek-R1 批量推理输出优化的解析脚本。
精准提取 "content" 中的诗歌正文，过滤 "reasoning_content"（思考过程）。
'''
import json
from datasets import SonnetsDataset

def get_prompt_from_sonnet(sonnet_text: str, num_prompt_lines: int = 3) -> str:
    lines = [line for line in sonnet_text.strip().split('\n') if line.strip()]
    return '\n'.join(lines[:num_prompt_lines])

def main():
    # 1. 加载原始训练集
    sonnet_dataset = SonnetsDataset("data/sonnets.txt")
    
    # 2. 读取从网页端下载的批量推理结果文件
    batch_output_path = "data/output.jsonl" 
    
    responses = {}
    with open(batch_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            custom_id = data["custom_id"]  # 例如 "sonnet_8"
            idx = int(custom_id.split('_')[1])
            
            try:
                content = data["response"]["body"]["choices"][0]["message"]["content"]
                responses[idx] = content.strip()
            except KeyError as e:
                print(f"⚠️ 警告: 样本 ID {idx} 数据结构异常或生成失败: {e}")

    # 3. 重新组装成标准的 DPO 训练 JSON
    paired_data = []
    for idx in range(len(sonnet_dataset)):
        _, sonnet_text = sonnet_dataset[idx]
        prompt = get_prompt_from_sonnet(sonnet_text, num_prompt_lines=3)
        
        if idx in responses:
            paired_data.append({
                "id": idx,
                "prompt": prompt,
                "chosen": sonnet_text.strip(),
                "rejected": responses[idx],  
            })
            
    # 4. 保存为标准 DPO 数据
    final_output_path = "data/sonnets_rejected_deepseek_r1.json"
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(paired_data, f, indent=2, ensure_ascii=False)
        
    print(f"🎉 R1 数据解析组装完毕！")
    print(f"👉 干净的 DPO 数据集已就绪（已成功剔除推理乱码）: {final_output_path}")
    print(f"   共成功组装 {len(paired_data)}/{len(sonnet_dataset)} 条困难负样本。")

if __name__ == "__main__":
    main()