# MicroP-GPT: From-Scratch GPT-2 with Parameter-Efficient LoRA and Preference-Aligned DPO for Poetry Generation

![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.46.3-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Abstract

This project presents a **from-scratch implementation of GPT-2** with advanced parameter-efficient fine-tuning (PEFT) and preference alignment techniques. Built upon Stanford CS224N's Final Project framework, we extend the baseline with two significant contributions:

1. **Hand-crafted LoRA (Low-Rank Adaptation)**: A complete implementation of LoRA's forward and backward propagation logic without relying on existing libraries (PEFT, bitsandbytes). Our LoRA adapter achieves **94.0% of full fine-tuning performance** while training only **0.24% of parameters**.

2. **Multi-Scale Model Analysis & DPO Negative Sampling**: We systematically evaluate GPT-2 across three scales (Small: 124M, Medium: 355M, Large: 774M) and propose three negative sampling strategies for Direct Preference Optimization:
- **Rule-Based Perturbation**: Operating directly on ground-truth sonnets by shuffling lines or applying lexical degradation
- **SFT-Generated Rejection**: Utilizing the fine-tuned model with elevated temperature to generate distribution-aligned hard negatives
- **LLM-API Generation**: Leveraging DeepSeek-R1 to craft grammatically correct but stylistically degraded sonnets

**Key Findings**: We discover that chrF scores remain flat while LLM-as-Judge scores significantly increase with model size, indicating that chrF fails to capture qualitative improvements. This work highlights fundamental limitations of automatic evaluation metrics for creative writing tasks.

---

## Repository Structure

```
MicroP-GPT/
├── models/
│   ├── gpt2.py              # GPT-2 model architecture & weight loading
│   └── base_gpt.py          # Base class for GPT models
├── modules/
│   ├── attention.py         # Masked Multi-Head Self-Attention
│   └── gpt2_layer.py        # Transformer Decoder Layer
├── lora.py                  # Hand-written LoRA implementation
├── optimizer.py             # Hand-written AdamW optimizer
├── sonnet_generation_dpo.py # DPO training pipeline
├── generate_rejected_sonnets.py  # Negative sampling strategies
├── paraphrase_detection_lora.py  # LoRA fine-tuning for paraphrase
├── classifier.py            # Sentiment classification
├── evaluation.py            # Evaluation metrics (chrF, accuracy)
├── data/                    # Datasets (SST, CFIMDB, Quora, Sonnets)
├── predictions/             # Model outputs
├── evaluation_results/      # Quantitative evaluation reports
└── logs/                    # Training logs with GPU metrics
```

**Key Implementation Files:**
- [lora.py](lora.py): Lines 17-77 — `LinearWithLoRA` class with merge/unmerge operations
- [sonnet_generation_dpo.py](sonnet_generation_dpo.py): Lines 131-163 — DPO loss computation
- [optimizer.py](optimizer.py): Lines 29-91 — AdamW with bias correction and weight decay
- [modules/attention.py](modules/attention.py): Lines 39-60 — Causal self-attention with masking

---

## Key Features & Contributions

### 1. Hand-Written AdamW Optimizer

We implement AdamW from scratch with explicit bias correction and decoupled weight decay:

```python
# AdamW update rule with bias correction
exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

bias_correction1 = 1 - beta1 ** t
bias_correction2 = 1 - beta2 ** t
step_size = alpha * math.sqrt(bias_correction2) / bias_correction1

p.data.addcdiv_(exp_avg, denom, value=-step_size)
p.data.add_(p.data, alpha=-weight_decay * alpha)  # Decoupled weight decay
```

### 2. Masked Multi-Head Self-Attention

Complete implementation including:
- **Causal masking**: Upper triangular mask to prevent attending to future tokens
- **Padding mask**: Excludes padding tokens from attention computation
- **Multi-head splitting**: Using `einops.rearrange` for tensor manipulation

```python
# Causal mask: prevent attending to future positions
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
attention_scores.masked_fill_(causal_mask, -10000.0)

# Scaled dot-product attention
attention_scores = torch.matmul(query, key.transpose(-1,-2)) / sqrt(head_dim)
```

### 3. Hand-Crafted LoRA Implementation

Our LoRA implementation follows the original paper's design with several key features:

**Forward Pass:**
```python
def forward(self, x):
    # Original frozen pathway
    result = self.original_linear(x)
    # LoRA low-rank pathway: x -> A (down-project) -> B (up-project) -> scale
    lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
    return result + self.scaling * lora_out  # scaling = alpha / r
```

**Weight Merging for Inference:**
```python
def merge_weights(self):
    # Merge LoRA weights into base weights for zero inference overhead
    self.original_linear.weight.data += self.scaling * (self.lora_B @ self.lora_A)
```

**Parameter Efficiency:**
| Configuration | Total Params | Trainable Params | Ratio |
|--------------|-------------|------------------|-------|
| Full Fine-tuning | 125,031,938 | 125,031,938 | 100% |
| LoRA (r=8) | 125,326,850 | 296,450 | **0.24%** |

### 4. DPO Loss Implementation

We implement the DPO objective following Rafailov et al. (2023):

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right) \right]$$

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()
    return loss
```

---

## Data Construction for DPO: Negative Sampling from Untuned Base Model

### Problem Statement

DPO requires paired preference data (chosen vs. rejected), but the Shakespeare sonnet corpus contains only high-quality positive samples. How can we construct meaningful negative samples?

### Our Solution: Three Negative Sampling Strategies

#### Strategy 1: Destroyed Positive Samples (Hard Negatives)

We perturb real sonnets through controlled corruption:
- **Line shuffling**: Randomize the last 4 lines while keeping the first 10 intact
- **Lexical degradation**: Replace poetic vocabulary with mundane alternatives

```python
replacements = {
    r'\blove\b': 'like',
    r'\bfair\b': 'good',
    r'\bbeauty\b': 'look',
    r'\bthou\b': 'you',  # Remove archaic elegance
    ...
}
```

#### Strategy 2: SFT Model Rejected Samples

Generate completions using the SFT-fine-tuned model with high temperature, selecting outputs that deviate from the training distribution.

#### Strategy 3: External LLM Rejected Samples (DeepSeek-R1)

Use a large-scale reasoning model (DeepSeek-R1) to generate alternative completions, providing stronger contrast signals.

### Effectiveness Analysis

#### Small Model (124M)

| Strategy | Reward Margin | chrF Score | LLM Total Score |
|----------|--------------|------------|-----------------|
| DestroyPS | 2.79 | **42.00** | 148.75 |
| SFT Rejected | 6.23 | 41.62 | 120.41 |
| DeepSeek-R1 | 16.39 | 41.12 | 91.25 |

**Finding on Small**: DestroyPS strategy yields best performance across all metrics
(chrF 42.00, LLM total score 148.75), while DeepSeek-R1 (reward margin 16.39) shows
the highest reward margin but worst LLM quality (91.25), suggesting that large reward
margins may lead to over-optimization without perceptual quality gains.


#### Medium Model (355M)

| Strategy | Reward Margin | chrF Score | LLM Total Score |
|----------|--------------|------------|-----------------|
| DestroyPS | 4.99 | 38.69 | 159.16 |
| SFT Rejected | 10.18 | **42.11** | **192.5** |
| DeepSeek-R1 | 16.39 | 34.56 | 116.25 |

**Finding on Medium**: SFT Rejected strategy dominates across all metrics (42.11 chrF, 192.5 LLM), representing a **39% improvement over baseline** in LLM-as-Judge scores. While the DeepSeek-R1 strategy has a larger reward margin (16.39), it still performs poorly compared to SFT Rejected (10.18).This suggests that an excessively large reward margin may lead to over-optimization and compromise the quality of the generated output. In addition, the optimal strategy differs from the Small model (where DestroyPS performed best). This **suggests** that negative sampling strategy effectiveness may vary with base model capacity, though we acknowledge this observation is based on limited experiments and requires further investigation.



> Due to GPU memory constraints (24GB), DPO on Large model was not feasible even with `batch_size=1`. This highlights a practical challenge of DPO alignment for larger models.

---

## Experimental Results

### Task 1: Sentiment Analysis (SST & CFIMDB)

| Configuration | Method | Dev Accuracy |
|--------------|--------|--------------|
| Last Linear Layer | Frozen GPT-2 | 44.2% |
| Full Fine-tuning | AdamW, lr=1e-5 | **52.8%** |

### Task 2: Paraphrase Detection (Quora)

| Configuration | Trainable Params | Dev Accuracy | Dev F1 | Training Time |
|--------------|------------------|--------------|--------|---------------|
| Full Fine-tuning | 125,031,938 (100%) | **89.76%** | 89.02% | ~5.9 hrs |
| LoRA (r=8) | 296,450 (0.24%) | 84.39% | 83.50% | ~3.5 hrs |

**Analysis**: LoRA achieves **94.0% of full fine-tuning performance** with **400× fewer trainable parameters** and **40% less training time**, demonstrating the efficiency of low-rank adaptation for semantic understanding tasks.

### Task 3: Sonnet Generation

#### Model Scale Comparison

| Model | Params | chrF Score | LLM Fluency | LLM Coherence | LLM Completeness |
|-------|--------|------------|-------------|---------------|------------------|
| GPT-2 Small (SFT) | 124M | 41.78 | 36.25 | 26.67 | 55.00 |
| GPT-2 Medium (SFT) | 355M | 41.18 | 47.92 | 35.83 | 55.00 |
| GPT-2 Large (SFT) | 774M | 41.23 | 53.33 | 42.08 | 60.83 |

#### DPO on Medium Model (chrF Score)

| Strategy | chrF Score | vs SFT | LLM Fluency | LLM Coherence | LLM Completeness | LLM Total Score |
|----------|------------|--------|-------------|---------------|------------------|-----------------|
| SFT Baseline | 41.18 | - | 47.92 | 35.83 | 55.00 | 138.75 |
| DPO (SFT Rejected) | **42.11** | +0.93 | **64.17** | **53.33** | **75.00** | **192.5** |
| DPO (DestroyPS) | 38.69 | -2.49 | 55.83 | 45.00 | 58.33 | 159.16 |
| DPO (DeepSeek-R1) | 34.56 | -6.62 | 50.83 | 41.25 | 24.17 | 116.25 |


#### Key Findings

1. **Model Scale Improves LLM-based Quality**: While chrF remains flat across model sizes (41.18-41.78), LLM-as-Judge scores show significant improvement (Small: 117.92 → Medium: 138.75 → Large: 156.25), indicating that chrF fails to capture qualitative improvements in generation.

2. **SFT Rejected Strategy Dominates on Medium**: Unlike Small where DestroyPS was optimal, Medium model achieves best results with SFT Rejected strategy (chrF 42.11, LLM 192.5), suggesting that negative sampling strategy effectiveness might depend on base model capacity.

3. **DPO Improves All LLM Dimensions**: Medium + DPO (SFT Rejected) achieves 34% higher fluency, 49% higher coherence, and 36% higher completeness compared to Medium SFT baseline. 

#### Training Dynamics

```
DPO (DestroyPS) Training Log:
Epoch 0: dpo_loss=0.786, reward_margin=0.05, chrF=41.62
Epoch 2: dpo_loss=0.398, reward_margin=1.37, chrF=42.00 ← Best
Epoch 5: dpo_loss=0.230, reward_margin=2.79, chrF=38.03 (Early stopped)
```

---

## Evaluation Metric Analysis: chrF vs LLM-as-Judge

### The chrF Limitation in Creative Writing

We observe a critical divergence between automatic metrics and human perceptual quality:

| Model | chrF Score | LLM Total Score | chrF vs LLM Correlation |
|-------|------------|-----------------|-------------------------|
| Small (124M) | 41.78 | 117.92 | Baseline |
| Medium (355M) | 41.18 | 138.75 | -0.6 chrF, +20.83 LLM |
| Large (774M) | 41.23 | 156.25 | -0.55 chrF, +38.33 LLM |

**Key Insight**: Despite achieving marginally lower chrF scores, Medium and Large models receive substantially higher LLM-as-Judge evaluations (17.7% and 32.5% improvement respectively). This suggests that chrF rewards over-fitting to training data while penalizing creative variation—a fundamental limitation for creative writing tasks.

### Why chrF Fails

chrF (Character-level n-gram F-score) measures character overlap between generated and reference texts:
```python
chrF = F_score(character_ngrams(generated), character_ngrams(reference))
```

**Problems for Poetry:**
1. **Rewards Copying**: chrF rewards verbatim repetition of training examples
2. **Penalizes Creativity**: Original poetic phrases receive lower scores
3. **Ignores Quality**: chrF cannot assess fluency, coherence, or literary merit
4. **Task Mismatch**: Poetry requires variation, not n-gram matching

### LLM-as-Judge Advantages

Our evaluation uses deepseek-v3 flash model to rate three dimensions, each dimension is scored from 1 to 100 points, and the average of the 12 poems is the final score:
- **Fluency**: Grammatical correctness and natural language flow
- **Coherence**: Thematic consistency and logical structure
- **Completeness**: Poetic structure and line count compliance


---

## Qualitative Case Study

### Sonnet #0 Comparison

**Prompt:**
```
Those lips that Love's own hand did make
Breathed forth the sound that said "I hate"
To me that languished for her sake;
```

**SFT Baseline Output:**
```
Against wherethat saw locks that stop Thy pains,
And in clothe her crippled limbs gone with slavery.
Span time away, bitterness supplicates my glory,
Therefore her tears sure shame faster crushed
Such roselakes to scickls then,
That you homely hoary tones found your cold foot so cold;
```

**DPO (DestroyPS) Output:**
```
But if this blush hath not taught
With friendship an immutable art,
Its love becomes fairer on both signs;
Then what beauty did amaze
As low white complexion with all thy eye,
Nor hath the melancholy discretion of love yet bent,
For so fair fortune hath reason commanded,
To increase lamentation and pleasure in thy sight.
  I pity thee for that your well-fed
```

**Observation**: DPO output maintains better thematic coherence with the prompt's romantic imagery, while the baseline introduces disjointed concepts ("crippled limbs", "slavery").

---

## Limitations and Analysis

### 1. Evaluation Metric Limitation

Our experiments reveal a fundamental limitation of chrF for creative writing evaluation:

| Model Scale | chrF Δ | LLM Score Δ | Interpretation |
|-------------|--------|-------------|----------------|
| Small → Medium | -0.6 | +20.83 | Higher quality, lower chrF |
| Small → Large | -0.55 | +38.33 | Much higher quality, same chrF |

**Conclusion**: chrF correlates poorly with human-perceptible quality in poetry generation. Relying solely on chrF can lead to incorrect conclusions about model performance.

### 2. DPO on Larger Models

Due to GPU memory constraints (24GB), we could not implement DPO on GPT-2 Large (774M) even with minimal batch_size. This represents a practical limitation of current DPO implementations:

**Memory Requirement** (estimated):
```
Small (124M):   ~12GB  (batch_size=4)
Medium (355M):  ~18GB  (batch_size=2)
Large (774M):   ~32GB  (batch_size=1) - exceeds our 24GB limit
```

**Potential Solutions**:
- Gradient checkpointing
- Model parallelism
- LoRA-DPO (parameter-efficient DPO)

### 3. Negative Sampling Strategy Dependence

Our experiments reveal that optimal negative sampling strategy depends on base model capacity:

| Model Scale | Optimal Strategy | chrF Improvement | LLM Improvement |
|-------------|------------------|------------------|-----------------|
| Small (124M) | DestroyPS | +0.22 | +26.2% |
| Medium (355M) | SFT Rejected | +0.93 | +38.7% |

**Insight**: Small models benefit from explicit structural constraints (perturbation), while larger models benefit from distribution-aligned negatives. This suggests a **capacity-aware negative sampling strategy** could improve DPO effectiveness.

### 4. Data Scarcity

With only ~2,000 sonnets for training and 12 for held-out evaluation, both SFT and DPO suffer from limited generalization capability. The small evaluation set (n=12) also limits statistical confidence in our results.

### Future Directions

1. **Metric Development**: Design evaluation metrics that better correlate with human perceptible quality for creative writing
2. **Memory-Efficient DPO**: Implement LoRA-DPO or gradient checkpointing to enable alignment of larger models
3. **Curriculum Learning**: Progressive negative sampling that adapts to base model capacity
4. **Data Augmentation**: Use synthetic data generation to increase training diversity

---

## Getting Started

### Installation

```bash
# Create conda environment
conda env create -f env.yml
conda activate cs224n_dfp
```


### Dependencies

- Python 3.8
- PyTorch 2.0+
- Transformers 4.46.3
- einops 0.8.0
- sacrebleu 2.5.1
- scikit-learn

### Training Commands

**Sentiment Analysis (SST/CFIMDB):**
```bash
python classifier.py --use_gpu --fine_tune_mode full-model --epochs 10 --lr 1e-5
```

**Paraphrase Detection with LoRA:**
```bash
python paraphrase_detection_lora.py --use_gpu --lora_r 8 --lora_alpha 16 --epochs 10
```

**Sonnet Generation (SFT):**
```bash
# Small model (124M)
python sonnet_generation.py --use_gpu --model_size gpt2 --epochs 10

# Medium model (355M)
python sonnet_generation.py --use_gpu --model_size gpt2-medium --batch_size 4

# Large model (774M)
python sonnet_generation.py --use_gpu --model_size gpt2-large --batch_size 2
```

**Sonnet Generation with DPO:**
```bash
# First generate negative samples
python generate_rejected_sonnets.py

# Then train with DPO (Medium model example)
python sonnet_generation_dpo.py \
    --use_gpu \
    --model_size gpt2-medium \
    --beta 0.1 \
    --sft_checkpoint best_10-1e-05-gpt2_medium-sonnet.pt \
    --rejected_path data/sonnets_rejected.json
```

### Evaluation

```bash
# Evaluate sonnet quality (chrF)
python evaluation.py

# Comprehensive dimension analysis
python eval_dimension1_format.py   # Structural metrics
python eval_dimension2_llm_judge.py --api_key <your_api_key>  # LLM-as-Judge(silicon flow platform default)
```

---

## References

1. **LoRA**: Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). Lora: Low-rank adaptation of large language models. Iclr, 1(2), 3.
2. **DPO**: Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36, 53728-53741.
3. **GPT-2**: Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.
4. **Adam**: Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
5. **AdamW**: Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

---

## Acknowledgments

This project builds upon Stanford CS224N's Final Project framework. Special thanks to the course staff for providing the foundational infrastructure and evaluation pipelines.

---

## Author

Guang Zhang | Zhejiang University, School of Mathematical Sciences

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
