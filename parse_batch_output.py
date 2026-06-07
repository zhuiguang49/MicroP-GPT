'''
将百炼离线推理返回的 output.jsonl 解析并还原为 DPO 训练用的标准 JSON 格式。
'''
import json
from datasets import SonnetsDataset

def get_prompt_from_sonnet(sonnet_text: str, num_prompt_lines: int = 3) -> str:
    lines = [line for line in sonnet_text.strip().split('\n') if line.strip()]
    return '\n'.join(lines[:num_prompt_lines])

def main():
    # 1. 加载原始数据集用作 Chosen 的对照
    sonnet_dataset = SonnetsDataset("data/sonnets.txt")
    
    # 2. 读取从网页端下载的离线推理结果
    batch_output_path = "data/output.jsonl" # 根据实际修改
    
    # 将结果读入字典，方便根据 custom_id 查找
    responses = {}
    with open(batch_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            custom_id = data["custom_id"]  # 形式为 "sonnet_X"
            idx = int(custom_id.split('_')[1])
            
            # 提取大模型生成的 flawed 诗歌
            content = data["response"]["body"]["choices"][0]["message"]["content"]
            responses[idx] = content.strip()

    # 3. 重新组装成标准的 DPO 配对数据集
    paired_data = []
    for idx in range(len(sonnet_dataset)):
        _, sonnet_text = sonnet_dataset[idx]
        prompt = get_prompt_from_sonnet(sonnet_text, num_prompt_lines=3)
        
        # 如果这个 ID 有成功返回的离线生成结果
        if idx in responses:
            paired_data.append({
                "id": idx,
                "prompt": prompt,
                "chosen": sonnet_text.strip(),
                "rejected": responses[idx],
            })
            
    # 4. 保存为标准 DPO 格式
    final_output_path = "data/sonnets_rejected_silicon.json"
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(paired_data, f, indent=2, ensure_ascii=False)
        
    print(f"🎉 完美收官！已成功将离线推理结果组装完毕。")
    print(f"👉 DPO 数据集已就绪: {final_output_path} (共 {len(paired_data)} 条样本)")

if __name__ == "__main__":
    main()