"""
实验日志工具类
用于结构化记录训练过程中的各项指标，便于后续分析和对比
"""

import json
import os
import time
import torch
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """
    实验日志记录器
    记录训练过程中的参数、指标、时间等信息
    """

    def __init__(self, experiment_name: str, task: str, method: str, output_dir: str = "logs"):
        """
        Args:
            experiment_name: 实验名称（如 "baseline", "lora_r8", "dpo_beta0.1"）
            task: 任务类型（"paraphrase" 或 "sonnet"）
            method: 方法类型（"full_finetune", "lora", "dpo"）
            output_dir: 日志输出目录
        """
        self.experiment_name = experiment_name
        self.task = task
        self.method = method
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化日志结构
        self.log = {
            "experiment": {
                "name": experiment_name,
                "task": task,
                "method": method,
                "timestamp": datetime.now().isoformat(),
            },
            "config": {},  # 训练配置
            "model": {},   # 模型信息
            "metrics": [], # 每个 epoch 的指标
            "training": {}, # 训练过程统计
            "results": {},  # 最终结果
            "notes": "",    # 备注
        }

        # 计时器
        self.start_time = None
        self.epoch_start_time = None

    def log_config(self, config_dict: dict):
        """记录训练配置"""
        self.log["config"] = config_dict

    def log_model_info(self, model: torch.nn.Module, method_specific_info: dict = None):
        """
        记录模型信息，包括参数量统计
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        # 计算模型保存大小（以 MB 为单位）
        param_size_mb = total_params * 4 / (1024 ** 2)  # float32

        model_info = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_ratio": trainable_params / total_params,
            "frozen_ratio": frozen_params / total_params,
            "param_size_mb": param_size_mb,
        }

        # 添加特定方法的信息
        if method_specific_info:
            model_info.update(method_specific_info)

        self.log["model"] = model_info
        print(f"[ExperimentLogger] Model info:")
        print(f"  Total params: {total_params:,} ({param_size_mb:.2f} MB)")
        print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  Frozen params: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")

    def log_epoch_start(self):
        """标记 epoch 开始"""
        self.epoch_start_time = time.time()

    def log_epoch_metrics(self, epoch: int, metrics_dict: dict):
        """
        记录单个 epoch 的指标
        Args:
            epoch: epoch 编号
            metrics_dict: 指标字典，如 {"train_loss": 2.34, "dev_acc": 0.85, ...}
        """
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0

        epoch_log = {
            "epoch": epoch,
            "epoch_time_sec": epoch_time,
            **metrics_dict
        }

        self.log["metrics"].append(epoch_log)

    def log_gpu_memory(self):
        """记录 GPU 显存占用"""
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
            self.log["training"]["gpu_memory_peak_mb"] = max_memory
            self.log["training"]["gpu_memory_current_mb"] = current_memory
            return max_memory
        return 0

    def log_training_start(self):
        """标记训练开始"""
        self.start_time = time.time()

    def log_training_end(self):
        """标记训练结束，计算总时间"""
        if self.start_time:
            total_time = time.time() - self.start_time
            self.log["training"]["total_time_sec"] = total_time
            self.log["training"]["avg_epoch_time_sec"] = total_time / len(self.log["metrics"])
            print(f"[ExperimentLogger] Total training time: {total_time:.2f}s")

    def log_final_results(self, results_dict: dict):
        """记录最终结果"""
        self.log["results"] = results_dict

    def log_note(self, note: str):
        """添加备注"""
        self.log["notes"] = note

    def save(self):
        """保存日志到 JSON 文件"""
        # 添加 GPU 显存信息
        self.log_gpu_memory()

        # 生成文件名
        filename = f"{self.task}_{self.method}_{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.log, f, indent=2, ensure_ascii=False)

        print(f"[ExperimentLogger] Log saved to: {filepath}")
        return str(filepath)

    def print_summary(self):
        """打印实验摘要"""
        print("\n" + "="*60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Task: {self.task}, Method: {self.method}")
        print("="*60)

        if self.log["model"]:
            print(f"Trainable params: {self.log['model']['trainable_params']:,} "
                  f"({self.log['model']['trainable_ratio']*100:.2f}%)")

        if self.log["metrics"]:
            last_epoch = self.log["metrics"][-1]
            print(f"Final metrics (epoch {last_epoch['epoch']}):")
            for key, value in last_epoch.items():
                if key not in ["epoch", "epoch_time_sec"]:
                    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        if self.log["training"]:
            if "total_time_sec" in self.log["training"]:
                print(f"Total time: {self.log['training']['total_time_sec']:.2f}s")
            if "gpu_memory_peak_mb" in self.log["training"]:
                print(f"Peak GPU memory: {self.log['training']['gpu_memory_peak_mb']:.2f} MB")

        if self.log["results"]:
            print("Final results:")
            for key, value in self.log["results"].items():
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        print("="*60 + "\n")


def compare_experiments(log_files: list, output_file: str = "logs/experiment_comparison.json"):
    """
    对比多个实验的结果
    """
    experiments = []
    for log_file in log_files:
        with open(log_file, "r") as f:
            experiments.append(json.load(f))

    # 按任务和方法分组
    comparison = {
        "paraphrase": {"full_finetune": [], "lora": []},
        "sonnet": {"full_finetune": [], "dpo": []},
    }

    for exp in experiments:
        task = exp["experiment"]["task"]
        method = exp["experiment"]["method"]
        if task in comparison and method in comparison[task]:
            comparison[task][method].append(exp)

    # 计算对比指标
    results = {
        "timestamp": datetime.now().isoformat(),
        "experiments_count": len(experiments),
        "comparison": {}
    }

    # Paraphrase 任务对比
    if comparison["paraphrase"]["full_finetune"] and comparison["paraphrase"]["lora"]:
        baseline = comparison["paraphrase"]["full_finetune"][0]
        lora = comparison["paraphrase"]["lora"][0]

        results["comparison"]["paraphrase"] = {
            "baseline_dev_acc": baseline["results"].get("dev_acc", 0),
            "lora_dev_acc": lora["results"].get("dev_acc", 0),
            "accuracy_drop": baseline["results"].get("dev_acc", 0) - lora["results"].get("dev_acc", 0),
            "param_efficiency": {
                "baseline_trainable_params": baseline["model"]["trainable_params"],
                "lora_trainable_params": lora["model"]["trainable_params"],
                "param_reduction_ratio": lora["model"]["trainable_params"] / baseline["model"]["trainable_params"],
            },
            "time_efficiency": {
                "baseline_time": baseline["training"].get("total_time_sec", 0),
                "lora_time": lora["training"].get("total_time_sec", 0),
            },
        }

    # Sonnet 任务对比
    if comparison["sonnet"]["full_finetune"] and comparison["sonnet"]["dpo"]:
        baseline = comparison["sonnet"]["full_finetune"][0]
        dpo = comparison["sonnet"]["dpo"][0]

        results["comparison"]["sonnet"] = {
            "baseline_chrF": baseline["results"].get("dev_chrF", 0),
            "dpo_chrF": dpo["results"].get("dev_chrF", 0),
            "chrF_improvement": dpo["results"].get("dev_chrF", 0) - baseline["results"].get("dev_chrF", 0),
        }

    # 保存对比结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[ExperimentLogger] Comparison saved to: {output_file}")
    return results
