import os
import re
import string
import pronouncing
import numpy as np
import pandas as pd

# ==========================================
# 1. 基础处理函数
# ==========================================

def clean_word(word):
    """去除标点符号，转换为小写，用于音标查询"""
    return word.translate(str.maketrans('', '', string.punctuation)).lower()

def count_syllables(line):
    """估算一行诗的音节数（十四行诗标准为10个音节）"""
    words = line.split()
    count = 0
    for word in words:
        cleaned = clean_word(word)
        if not cleaned: continue
        phones = pronouncing.phones_for_word(cleaned)
        if phones:
            count += sum(1 for char in phones[0] if char.isdigit())
        else:
            count += max(1, len(cleaned) // 3)
    return count

# ==========================================
# 2. 单首诗歌评测核心逻辑
# ==========================================

def evaluate_single_sonnet(sonnet_text):
    """评测单首诗歌：返回行数、是否14行、音节方差、尾词押韵率"""
    lines = [line.strip() for line in sonnet_text.strip().split('\n') if line.strip()]
    
    line_count = len(lines)
    is_14_lines = 1 if line_count == 14 else 0
    
    if line_count == 0:
        return None

    syllables_per_line = [count_syllables(line) for line in lines]
    syllable_variance = np.mean([(s - 10)**2 for s in syllables_per_line])
    
    end_words = [clean_word(line.split()[-1]) for line in lines if line.split()]
    rhyme_hits = 0
    total_pairs = max(1, len(end_words) - 1)
    
    for i in range(len(end_words) - 1):
        word1 = end_words[i]
        word2 = end_words[i+1]
        word_plus_2 = end_words[i+2] if i+2 < len(end_words) else ""
        
        rhymes_of_word1 = pronouncing.rhymes(word1)
        if word2 in rhymes_of_word1 or word_plus_2 in rhymes_of_word1:
             rhyme_hits += 1
             
    rhyme_score = rhyme_hits / total_pairs

    return {
        "line_count": line_count,
        "is_14_lines_score": is_14_lines,
        "syllable_variance": syllable_variance,
        "rhyme_score": rhyme_score
    }

# ==========================================
# 3. 文件解析与批量评测
# ==========================================

def parse_and_evaluate_file(filepath):
    """读取文件，切分多首诗，计算该文件的平均指标"""
    if not os.path.exists(filepath):
        print(f"警告：未找到文件 {filepath}")
        return None
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    content = content.replace("--Generated Sonnets--", "")
    raw_sonnets = re.split(r'\n\d+\n', '\n' + content.strip())
    
    results = []
    for text in raw_sonnets:
        if len(text.strip()) > 10:
            res = evaluate_single_sonnet(text)
            if res:
                results.append(res)
                
    if not results:
        return None
        
    avg_line_count = np.mean([r["line_count"] for r in results])
    avg_14_compliance = np.mean([r["is_14_lines_score"] for r in results]) * 100
    avg_variance = np.mean([r["syllable_variance"] for r in results])
    avg_rhyme = np.mean([r["rhyme_score"] for r in results])
    
    return {
        "Average Lines (Target: 14)": round(avg_line_count, 1),
        "14-Line Compliance (%)": round(avg_14_compliance, 2),
        "Syllable Variance (Lower is better)": round(avg_variance, 2),
        "Rhyme Activity Score": round(avg_rhyme, 4)
    }

# ==========================================
# 4. 主执行流（集成文件保存功能）
# ==========================================

if __name__ == "__main__":
    PRED_DIR = "predictions"
    OUTPUT_DIR = "evaluation_results" # 结果保存的目标文件夹
    
    # 创建保存结果的文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 包含了 GPT-2 Small 和 GPT-2 Medium 的全量实验组
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
        "Medium_DPO (DeepSeek-R1 Rejected)": "generated_sonnets_dpo_beta0.1_medium_DeepSeekR1.txt"
    }
    final_report = []

    print("开始进行维度一（格式与格律约束性）评测...")
    
    for model_name, filename in experiments.items():
        full_path = os.path.join(PRED_DIR, filename)
        metrics = parse_and_evaluate_file(full_path)
        if metrics:
            row = {"Model Strategy": model_name}
            row.update(metrics)
            final_report.append(row)

    if final_report:
        df = pd.DataFrame(final_report)
        markdown_table = df.to_markdown(index=False)
        
        # 1. 屏幕打印输出
        print("\n" + "="*20 + " 评测结果 " + "="*20)
        print(markdown_table)
        print("="*50 + "\n")
        
        # 2. 保存为人类易读的 Text 报告 (包含 Markdown 表格)
        txt_output_path = os.path.join(OUTPUT_DIR, "dimension1_report.txt")
        with open(txt_output_path, "w", encoding="utf-8") as f:
            f.write("# 维度一：格式与格律约束性评测报告\n\n")
            f.write(markdown_table)
            f.write("\n\n*注：Syllable Variance 指标越低越好，代表越接近抑扬格五音步。*")
        print(f"📄 已成功将可视化报告保存至: {txt_output_path}")

        # 3. 保存为标准的 CSV 文件 (方便导入 Excel、Origin 或 Python 用来画条形图/折线图)
        csv_output_path = os.path.join(OUTPUT_DIR, "dimension1_metrics.csv")
        df.to_csv(csv_output_path, index=False, encoding="utf-8")
        print(f"📊 已成功将原始数据保存至: {csv_output_path}")
        
    else:
        print("错误：未成功读取到任何有效的评测结果。")