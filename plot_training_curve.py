#!/usr/bin/env python3
"""
Training Curve Visualization Script

Generates publication-quality plots for sonnet generation experiments.
Supports comparing multiple models (Small/Medium/Large) on the same plot.

Usage:
  python plot_training_curve.py                    # Plot all sonnet logs
  python plot_training_curve.py --compare          # Compare Small vs Medium
  python plot_training_curve.py --log sonnet_full_finetune_baseline_gpt2_*.json
"""

import argparse
import json
import glob
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

# 设置中文字体支持
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150


# 模型颜色和标记配置
MODEL_STYLES = {
    'gpt2': {'color': '#1f77b4', 'marker': 'o', 'label': 'GPT-2 Small (124M)'},
    'gpt2_medium': {'color': '#ff7f0e', 'marker': 's', 'label': 'GPT-2 Medium (355M)'},
    'gpt2_large': {'color': '#2ca02c', 'marker': '^', 'label': 'GPT-2 Large (774M)'},
    'gpt2_xl': {'color': '#d62728', 'marker': 'D', 'label': 'GPT-2 XL (1.5B)'},
}


def load_log_file(log_path):
    """Load a single log file and extract metrics."""
    with open(log_path, 'r') as f:
        data = json.load(f)
    return data


def extract_metrics(log_data):
    """Extract training metrics from log data."""
    metrics = log_data.get('metrics', [])

    epochs = []
    train_loss = []
    dev_chrF = []

    for m in metrics:
        epochs.append(m.get('epoch', len(epochs)))
        train_loss.append(m.get('train_loss', None))
        dev_chrF.append(m.get('dev_chrF', None))

    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'dev_chrF': dev_chrF,
        'best_chrF': log_data.get('results', {}).get('best_chrF', None),
        'model_name': log_data.get('config', {}).get('model_size', 'gpt2').replace('-', '_'),
        'exp_name': log_data.get('experiment', {}).get('name', 'unknown'),
    }


def detect_model_from_log(log_path, log_data):
    """Detect model size from log path or data."""
    # 首先尝试从配置中获取
    model_size = log_data.get('config', {}).get('model_size', 'gpt2')
    return model_size.replace('-', '_')


def plot_single_model(metrics, output_path='figures/training_curve.png', title=None):
    """Plot training curve for a single model."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    model_name = metrics['model_name']
    style = MODEL_STYLES.get(model_name, MODEL_STYLES['gpt2'])

    # 左轴：Training Loss
    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', color=color1, fontsize=12)

    valid_loss = [(e, l) for e, l in zip(metrics['epochs'], metrics['train_loss']) if l is not None]
    if valid_loss:
        epochs_loss, losses = zip(*valid_loss)
        ax1.plot(epochs_loss, losses, color=color1, linestyle='-', marker=style['marker'],
                 markersize=6, linewidth=2, label='Train Loss')
        ax1.tick_params(axis='y', labelcolor=color1)

    # 右轴：Dev chrF
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Dev chrF Score', color=color2, fontsize=12)

    valid_chrF = [(e, c) for e, c in zip(metrics['epochs'], metrics['dev_chrF']) if c is not None]
    if valid_chrF:
        epochs_chrF, chrF_scores = zip(*valid_chrF)
        ax2.plot(epochs_chrF, chrF_scores, color=color2, linestyle='--', marker='o',
                 markersize=6, linewidth=2, label='Dev chrF')
        ax2.tick_params(axis='y', labelcolor=color2)

        # 标记最佳 chrF 点
        best_idx = chrF_scores.index(max(chrF_scores))
        best_epoch = epochs_chrF[best_idx]
        best_chrF = max(chrF_scores)
        ax2.scatter([best_epoch], [best_chrF], color='gold', s=200, zorder=5,
                    edgecolors='black', linewidths=2, marker='*')
        ax2.annotate(f'Best: {best_chrF:.2f}', (best_epoch, best_chrF),
                     textcoords="offset points", xytext=(10, 10),
                     fontsize=10, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='gold'))

    # 标题和图例
    if title is None:
        title = f'Training Curve - {style["label"]}'
    plt.title(title, fontsize=14, fontweight='bold')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    plt.tight_layout()

    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved training curve to: {output_path}")
    plt.close()

    return output_path


def plot_comparison(metrics_list, output_path='figures/training_curve_comparison.png', title=None):
    """Plot training curves for multiple models for comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1：Training Loss 对比
    ax1 = axes[0]
    for metrics in metrics_list:
        model_name = metrics['model_name']
        style = MODEL_STYLES.get(model_name, MODEL_STYLES['gpt2'])

        valid_loss = [(e, l) for e, l in zip(metrics['epochs'], metrics['train_loss']) if l is not None]
        if valid_loss:
            epochs_loss, losses = zip(*valid_loss)
            ax1.plot(epochs_loss, losses, color=style['color'], linestyle='-',
                     marker=style['marker'], markersize=5, linewidth=2,
                     label=style['label'])

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 子图2：Dev chrF 对比
    ax2 = axes[1]
    for metrics in metrics_list:
        model_name = metrics['model_name']
        style = MODEL_STYLES.get(model_name, MODEL_STYLES['gpt2'])

        valid_chrF = [(e, c) for e, c in zip(metrics['epochs'], metrics['dev_chrF']) if c is not None]
        if valid_chrF:
            epochs_chrF, chrF_scores = zip(*valid_chrF)
            ax2.plot(epochs_chrF, chrF_scores, color=style['color'], linestyle='-',
                     marker=style['marker'], markersize=5, linewidth=2,
                     label=style['label'])

            # 标记最佳点
            best_idx = chrF_scores.index(max(chrF_scores))
            ax2.scatter([epochs_chrF[best_idx]], [max(chrF_scores)],
                        color=style['color'], s=100, zorder=5,
                        edgecolors='black', linewidths=1.5, marker='*')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dev chrF Score', fontsize=12)
    ax2.set_title('Dev chrF Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 总标题
    if title is None:
        title = 'Model Size Comparison: Sonnet Generation'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved comparison plot to: {output_path}")
    plt.close()

    return output_path


def plot_chrF_comparison_bar(metrics_list, output_path='figures/chrf_comparison_bar.png'):
    """Plot a bar chart comparing best chrF scores across models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    model_names = []
    best_chrFs = []
    colors = []

    for metrics in metrics_list:
        model_name = metrics['model_name']
        style = MODEL_STYLES.get(model_name, MODEL_STYLES['gpt2'])

        model_names.append(style['label'].split('(')[0].strip())  # Just "GPT-2 Small" etc.
        best_chrFs.append(metrics.get('best_chrF', max([c for c in metrics['dev_chrF'] if c is not None])))
        colors.append(style['color'])

    bars = ax.bar(model_names, best_chrFs, color=colors, edgecolor='black', linewidth=1.5)

    # 在柱子上添加数值标签
    for bar, chrF in zip(bars, best_chrFs):
        height = bar.get_height()
        ax.annotate(f'{chrF:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Best chrF Score', fontsize=12)
    ax.set_title('Best chrF Score by Model Size', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(best_chrFs) * 1.15)

    # 添加网格线
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved bar chart to: {output_path}")
    plt.close()


def find_sonnet_logs(logs_dir='logs', model_filter=None):
    """Find all sonnet generation log files."""
    pattern = os.path.join(logs_dir, 'sonnet_full_finetune_*.json')
    log_files = glob.glob(pattern)

    if model_filter:
        log_files = [f for f in log_files if model_filter.replace('-', '_') in f]

    return sorted(log_files)


def main():
    parser = argparse.ArgumentParser(description='Plot training curves for sonnet generation experiments.')

    parser.add_argument('--log', type=str, nargs='+', default=None,
                        help='Specific log file(s) to plot. If not specified, plots all sonnet logs.')
    parser.add_argument('--compare', action='store_true',
                        help='Generate comparison plots for all models.')
    parser.add_argument('--output_dir', type=str, default='figures',
                        help='Output directory for plots.')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom title for the plot.')

    args = parser.parse_args()

    # 查找日志文件
    if args.log:
        log_files = args.log
    else:
        log_files = find_sonnet_logs()
        if not log_files:
            print("❌ No sonnet log files found in logs/ directory.")
            print("   Make sure you have run sonnet_generation.py first.")
            return

    print(f"📂 Found {len(log_files)} log file(s):")
    for f in log_files:
        print(f"   - {f}")

    # 加载并处理日志
    metrics_list = []
    for log_file in log_files:
        print(f"\n📊 Processing: {log_file}")
        log_data = load_log_file(log_file)
        metrics = extract_metrics(log_data)
        metrics['model_name'] = detect_model_from_log(log_file, log_data)
        metrics_list.append(metrics)
        print(f"   Model: {metrics['model_name']}")
        print(f"   Best chrF: {metrics.get('best_chrF', 'N/A')}")

    # 生成图表
    if args.compare and len(metrics_list) > 1:
        # 对比图
        output_path = os.path.join(args.output_dir, 'training_curve_comparison.png')
        plot_comparison(metrics_list, output_path, args.title)

        # 柱状图
        bar_path = os.path.join(args.output_dir, 'chrf_comparison_bar.png')
        plot_chrF_comparison_bar(metrics_list, bar_path)
    else:
        # 单模型图
        for metrics in metrics_list:
            model_name = metrics['model_name']
            output_path = os.path.join(args.output_dir, f'training_curve_{model_name}.png')
            plot_single_model(metrics, output_path, args.title)

    print("\n✅ All plots generated successfully!")


if __name__ == "__main__":
    main()
