'''
将训练集 sonnet 转换为百炼批量推理（Batch Inference）所需的 JSONL 格式。
'''
import json
from datasets import SonnetsDataset

def get_prompt_from_sonnet(sonnet_text: str, num_prompt_lines: int = 3) -> str:
    lines = [line for line in sonnet_text.strip().split('\n') if line.strip()]
    return '\n'.join(lines[:num_prompt_lines])

def main():
    # 1. 读取原本的训练集
    sonnet_dataset = SonnetsDataset("data/sonnets.txt")
    
    system_prompt = "You are a poet who intentionally writes mediocre and flawed Shakespearean sonnets."
    
    output_file = "data/batch_input.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx in range(len(sonnet_dataset)):
            _, sonnet_text = sonnet_dataset[idx]
            prompt = get_prompt_from_sonnet(sonnet_text, num_prompt_lines=3)
            
            user_prompt = f'Here are the first 3 lines of a sonnet:\n"{prompt}"\n\nPlease complete this sonnet (total 14 lines including the prompt). However, you must make it LOW QUALITY by following these rules:\n1. Use awkward or forced rhymes.\n2. Include some modern or mundane vocabulary that doesn\'t fit the Shakespearean style.\n3. Make the logic slightly disjointed or cliché.\n4. Keep the 14-line structure but break the iambic pentameter occasionally.\n\nOutput ONLY the completed sonnet text.'
            
            # 2. 构造硅基流动要求的标准单行 JSON 对象
            batch_line = {
                "custom_id": f"sonnet_{idx}",  # 自定义ID，用于后续和真实数据对齐
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "deepseek-r1",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.9
                }
            }
            
            # 写入一行
            f.write(json.dumps(batch_line, ensure_ascii=False) + '\n')
            
    print(f"✅ 成功生成批量推理输入文件: {output_file}，共 {len(sonnet_dataset)} 条数据。")

if __name__ == "__main__":
    main()