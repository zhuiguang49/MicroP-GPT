import os
import re
import json
import argparse
import requests
import pandas as pd
import numpy as np


JUDGE_SYSTEM_PROMPT = """你是一位精通莎士比亚十四行诗（Sonnet）的文学专家和语言学裁判。
你需要客观地评估用户给出的十四行诗（前三行作为prompt，不计入评分）。请从以下两个维度进行打分（每个维度 1-100 分，1分最差，100分完美），并且打分要有区间度：

1. Fluency (语言流利度)：句法是否通顺？是否存在乱码、拼写错误或无意义的词汇堆砌？
2. Coherence & Poetry (连贯性与诗意)：上下文逻辑是否连贯？意象是否具有古典诗歌的美感与合理性？
3. Completeness（完整性）：十四行诗是否完整？

请严格按照以下 JSON 格式返回你的评价，不要输出任何额外的解释或散作：
{
    "fluency_score": (1-100的整数),
    "coherence_score": (1-100的整数),
    "completeness_score": (1-100的整数),
    "reason": "简短的中文评语（50字以内，指出具体好在哪里或为何崩溃）"
}"""

def call_llm_judge(sonnet_text, api_key):
    """调用 DeepSeek API 对单首诗进行打分并获取评语"""
    api_url = "https://api.siliconflow.cn/v1/chat/completions"
    model_name = "deepseek-ai/DeepSeek-V3"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"请对以下诗歌片段进行评审：\n\n{sonnet_text}"}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            res_json = response.json()
            content = res_json['choices'][0]['message']['content']
            scores = json.loads(content)
            return {
                "fluency": float(scores.get("fluency_score", 1)),
                "coherence": float(scores.get("coherence_score", 1)),
                "completeness": float(scores.get("completeness_score", 1)), 
                "reason": scores.get("reason", "未提供理由")
            }
        else:
            print(f"  ❌ API 请求失败，状态码: {response.status_code}")
            return None
    except Exception as e:
        print(f"  ❌ 请求或解析发生异常: {e}")
        return None


def parse_and_evaluate_file(filepath, api_key, model_name, detailed_logs):
    """读取文件，切分诗歌，调用 API 计算平均分并记录每条详细评语"""
    if not os.path.exists(filepath):
        print(f"警告：未找到文件 {filepath}")
        return None
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    content = content.replace("--Generated Sonnets--", "")
    raw_sonnets = re.split(r'\n\d+\n', '\n' + content.strip())
    
    fluency_scores = []
    coherence_scores = []
    completeness_scores = []
    
    sonnet_index = 0
    for text in raw_sonnets:
        text_str = text.strip()
        if len(text_str) > 10:
                
            print(f"  正在调用 API 评测第 {sonnet_index} 首诗...")
            res = call_llm_judge(text_str, api_key)
            
            if res:
                fluency_scores.append(res["fluency"])
                coherence_scores.append(res["coherence"])
                completeness_scores.append(res["completeness"]) 
                
                detailed_logs.append({
                    "model_strategy": model_name,
                    "sonnet_index": sonnet_index,
                    "raw_text": text_str,
                    "fluency_score": res["fluency"],
                    "coherence_score": res["coherence"],
                    "completeness_score": res["completeness"], 
                    "reason": res["reason"]
                })
            sonnet_index += 1
                
    if not fluency_scores:
        return None
        

    return {
        "LLM Fluency Score (1-100)": round(np.mean(fluency_scores), 2),
        "LLM Coherence Score (1-100)": round(np.mean(coherence_scores), 2),
        "LLM Completeness Score (1-100)": round(np.mean(completeness_scores), 2) 
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="维度二：基于大模型裁判(LLM-as-a-Judge)并包含审核日志的评测脚本")
    parser.add_argument("--api_key", type=str, required=True, help="硅基流动(SiliconFlow)的 API Key")
    args = parser.parse_args()

    PRED_DIR = "predictions"
    OUTPUT_DIR = "evaluation_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    experiments = {
        # === 组别 A: GPT-2 Small (124M) ===
        "Small_SFT Baseline": "generated_sonnets.txt",
        "Small_DPO (SFT Rejected)": "generated_sonnets_dpo_beta0.1_sftRejected.txt",
        "Small_DPO (Destroyed PS)": "generated_sonnets_dpo_beta0.1_DestroyPS.txt",
        "Small_DPO (DeepSeek-R1 Rejected)": "generated_sonnets_dpo_beta0.1_Deepseekr1_Rejected.txt",
        
        # === 组别 B: GPT-2 Medium (355M) ===
        "Medium_SFT Baseline": "generated_sonnets_gpt2_medium.txt",
        "Medium_DPO (SFT Rejected)": "generated_sonnets_dpo_beta0.1_medium_SFTRejected.txt",
        "Medium_DPO (Destroyed PS)": "generated_sonnets_dpo_beta0.1_medium_DestroyPS.txt",
        "Medium_DPO (DeepSeek-R1 Rejected)": "generated_sonnets_dpo_beta0.1_medium_DeepSeekR1.txt",

        # === 组别 C: GPT-2 Large (774M) ===
        "Large_SFT Baseline": "generated_sonnets_gpt2_large.txt" 
    }

    final_report = []
    detailed_logs = [] 
    
    print("开始进行维度二（LLM-as-a-Judge 大模型裁判）评测...")

    for model_name, filename in experiments.items():
        full_path = os.path.join(PRED_DIR, filename)
        print(f"\n正在评测模型: {model_name}")
        metrics = parse_and_evaluate_file(full_path, args.api_key, model_name, detailed_logs)
        if metrics:
            row = {"Model Strategy": model_name}
            row.update(metrics)
            final_report.append(row)

    if final_report:
        df = pd.DataFrame(final_report)
        markdown_table = df.to_markdown(index=False)
        
        print("\n" + "="*20 + " LLM 裁判均分结果 " + "="*20)
        print(markdown_table)
        print("="*54 + "\n")
        
        txt_output_path = os.path.join(OUTPUT_DIR, "dimension2_llm_report.txt")
        with open(txt_output_path, "w", encoding="utf-8") as f:
            f.write("# 维度二：LLM-as-a-Judge 语言流利度与诗意评测报告（均分）\n\n")
            f.write(markdown_table)

        log_df = pd.DataFrame(detailed_logs)

        light_csv_df = log_df[["model_strategy", "sonnet_index", "fluency_score", "coherence_score", "completeness_score"]]
        
        csv_log_path = os.path.join(OUTPUT_DIR, "llm_judge_audit_details.csv")
        light_csv_df.to_csv(csv_log_path, index=False, encoding="utf-8")
        
        json_log_path = os.path.join(OUTPUT_DIR, "llm_judge_audit_details.json")
        with open(json_log_path, "w", encoding="utf-8") as f:
            json.dump(detailed_logs, f, indent=4, ensure_ascii=False)
            
        print(f"评测完成：")
        print(f"结构化数据纯得分表: {csv_log_path}")
        print(f"完整审核日志与中文评语: {json_log_path}")
    else:
        print("错误：未成功获取到任何 API 评测数据。")