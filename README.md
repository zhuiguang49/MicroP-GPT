# MicroP-GPT: From-Scratch GPT-2 with Parameter-Efficient LoRA and Preference-Aligned DPO for Poetry Generation

![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.46.3-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Abstract

This project presents a **from-scratch implementation of GPT-2** with advanced parameter-efficient fine-tuning (PEFT) and preference alignment techniques. Built upon Stanford CS224N's Final Project framework, we extend the baseline with two significant contributions:

1. **Hand-crafted LoRA (Low-Rank Adaptation)**: A complete implementation of LoRA's forward and backward propagation logic without relying on existing libraries (PEFT, bitsandbytes). Our LoRA adapter achieves **94.0% of full fine-tuning performance** while training only **0.24% of parameters**.

2. **Exploration of Diverse Negative Sampling Strategies for DPO:** We propose and thoroughly evaluate three distinct strategies for constructing preference pairs in data-scarce scenarios, moving beyond naive completions to build higher-quality "losing" samples for Direct Preference Optimization:
- Strategy A (Rule-Based Heuristic Perturbation): Operating directly on ground-truth ("chosen") sonnets by shuffling quatrain lines or applying pseudo-modern replacement (e.g., thou $\to$ you), forcing the model to learn strict syntactic layout and vocabulary constraints.
- Strategy B (SFT Model with Perturbed Generation): Utilizing our own SFT model with elevated temperature ($1.2 \sim 1.3$) to generate "hard negatives" that preserve poetic form but contain flaws in meter or coherence.
- Strategy C (LLM-as-a-Poet API Generation): Leveraging advanced LLM (DeepSeek-R1 via SiliconFlow Batch Inference) with tailored prompt engineering to intentionally craft grammatically correct but logically disjointed, modern-slang-filled, and rhyme-forced sonnets.

This project features handcrafted implementations of the AdamW optimizer, Masked Multi-Head Self-Attention and DPO loss, covering fundamental Transformer architectures, optimization algorithms and modern LLM alignment techniques.

We validate our LoRA implementation on Quora paraphrase detection (achieving 94% of full fine-tuning performance with 0.24% parameters) and our DPO framework on Shakespeare sonnet generation, providing systematic analysis of negative sampling strategies for preference alignment in data-scarce creative writing tasks.

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

| Strategy | Reward Margin | chrF Score | Observations |
|----------|--------------|------------|--------------|
| DestroyPS | 2.79 | **42.00** | Moderate margin, stable training |
| SFT Rejected | 6.23 | 41.62 | High margin, potential overfitting |
| DeepSeek-R1 | 16.39 | 41.12 | Very high margin, but distribution shift |

**Key Finding**: Hard negatives from perturbation yield the best chrF improvement (+0.22 over baseline), which might suggest that **fine-grained quality differences** are more effective for DPO than coarse-grained contrasts. However, due to time and resource constraints, there are still lots of room for confimration.

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

#### Quantitative Results (chrF Score)

| Method | chrF Score | 14-Line Compliance | Syllable Variance |
|--------|------------|-------------------|-------------------|
| SFT Baseline | 41.78 | 33.33% | 10.26 |
| DPO (DestroyPS) | **42.00** | **41.67%** | 8.62 |
| DPO (SFT Rejected) | 41.62 | 8.33% | 6.42 |
| DPO (DeepSeek-R1) | 41.12 | 0.00% | 9.94 |

#### LLM-as-Judge Evaluation

| Method | Fluency (1-5) | Coherence (1-5) |
|--------|---------------|-----------------|
| SFT Baseline | 3.00 | 2.00 |
| DPO (SFT Rejected) | 3.33 | 2.42 |
| DPO (DestroyPS) | **3.50** | **2.67** |
| DPO (DeepSeek-R1) | 3.25 | 2.42 |

#### Training Dynamics

```
DPO (DestroyPS) Training Log:
Epoch 0: dpo_loss=0.786, reward_margin=0.05, chrF=41.62
Epoch 2: dpo_loss=0.398, reward_margin=1.37, chrF=42.00 ← Best
Epoch 5: dpo_loss=0.230, reward_margin=2.79, chrF=38.03 (Early stopped)
```

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

### DPO's Limited Improvement on Sonnet Generation

Despite implementing DPO with carefully constructed preference pairs, the chrF improvement is marginal (+0.22 over baseline). We identify several contributing factors:

#### 1. Base Model Capacity Constraints

GPT-2 (124M parameters) was trained on WebText, which lacks poetic structure. The model struggles with:
- **Iambic pentameter**: Consistent 10-syllable lines with alternating stress
- **Quatrain structure**: ABAB rhyme schemes
- **Sonnet conventions**: Volta (turning point) at line 9

Even with preference alignment, the model cannot generate what its representation space cannot encode.

#### 2. Negative Sample Quality Trade-off

| Strategy | Issue |
|----------|-------|
| DestroyPS | Negatives too similar to positives → weak learning signal |
| SFT Rejected | Model already learned to avoid these → redundant |
| DeepSeek-R1 | Distribution too different → model ignores signal |

Finding the "Goldilocks zone" of negative quality remains an open challenge.

#### 3. Reward Hacking Risk

The increasing reward margin (0.05 → 2.79) with decreasing chrF after epoch 2 suggests the model may be optimizing the DPO objective at the expense of generation quality—a form of reward hacking.

#### 4. Data Scarcity

With only ~2,000 sonnets for training and 12 for held-out evaluation, both SFT and DPO suffer from limited generalization capability.

### Future Directions

1. **Larger base model**: GPT-2 Medium (355M) or Large (774M) may better capture poetic structure
2. **Curriculum DPO**: Progressive difficulty in negative samples
3. **Multi-objective alignment**: Combine chrF with fluency and coherence rewards
4. **Synthetic augmentation**: Use GPT-4 to generate additional training sonnets

---

## Getting Started

### Installation

```bash
source setup.sh
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
python sonnet_generation.py --use_gpu --epochs 10 --temperature 1.2 --top_p 0.9
```

**Sonnet Generation with DPO:**
```bash
# First generate negative samples
python generate_rejected_sonnets.py

# Then train with DPO
python sonnet_generation_dpo.py --use_gpu --beta 0.1 --sft_checkpoint best_10-1e-05-sonnet.pt
```

### Evaluation

```bash
# Evaluate sonnet quality (chrF)
python evaluation.py

# Comprehensive dimension analysis
python eval_dimension1_format.py   # Structural metrics
python eval_dimension2_llm_judge.py --api_key <your_api_key>  # LLM-as-Judge
```

---

## References

1. **LoRA**: Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
2. **DPO**: Rafailov, R., et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023.
3. **GPT-2**: Radford, A., et al. "Language Models are Unsupervised Multitask Learners." OpenAI 2019.
4. **AdamW**: Loshchilov, I., & Hutter, F. "Decoupled Weight Decay Regularization." ICLR 2019.

---

## Acknowledgments

This project builds upon Stanford CS224N's Final Project framework. Special thanks to the course staff for providing the foundational infrastructure and evaluation pipelines.

---

## Author

Guang Zhang | Zhejiang University, School of Mathematical Sciences

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
