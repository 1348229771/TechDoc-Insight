import os
import sys
import json
import argparse
import torch
import matplotlib.pyplot as plt
import csv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.t5_model import T5SummarizationModel
from utils.data_loader import create_data_loaders, load_tokenizer
from utils.metrics import evaluate_model, print_evaluation_results
from configs.config import (
    MODEL_NAME, MODEL_CACHE_DIR, TRAIN_CONFIG, DATA_CONFIG,
    INFERENCE_CONFIG, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    SAMPLE_LIMITS
)


def parse_args():
    parser = argparse.ArgumentParser(description="对比原版T5与微调T5的评估与可视化")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--model_cache_dir", type=str, default=MODEL_CACHE_DIR)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--max_test_samples", type=int, default=SAMPLE_LIMITS["test"])
    parser.add_argument("--resume_from_checkpoint", type=str, default=os.path.join("./outputs", "models", "best_model"))
    parser.add_argument("--plot_output", type=str, default=os.path.join("outputs", "results", "comparison_metrics.png"))
    parser.add_argument("--report_output", type=str, default=os.path.join("outputs", "results", "comparison_report.md"))
    parser.add_argument("--metrics", nargs="+", default=["rouge1", "rouge2", "rougeL", "bleu"])
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--csv_output", type=str, default=os.path.join("outputs", "results", "comparison_results.csv"))
    return parser.parse_args()


def ensure_dirs(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_dir = os.path.join(output_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir


def main():
    args = parse_args()
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            props = torch.cuda.get_device_properties(0)
            if props.total_memory <= 8 * 1024 * 1024 * 1024 and args.batch_size > 2:
                args.batch_size = 2
                print("检测到显存较小，调整batch_size到2")
        except Exception:
            pass
    results_dir = ensure_dirs(args.output_dir)
    tokenizer = load_tokenizer(args.model_name, args.model_cache_dir)
    _, _, test_loader = create_data_loaders(
        TRAIN_DATA_PATH,
        VAL_DATA_PATH,
        TEST_DATA_PATH,
        tokenizer,
        batch_size=args.batch_size,
        max_source_length=DATA_CONFIG["max_source_length"],
        max_target_length=DATA_CONFIG["max_target_length"],
        limit_train_samples=None,
        limit_val_samples=None,
        limit_test_samples=args.max_test_samples,
        num_workers=0,
        pin_memory=False
    )
    print("评估原版T5...")
    base_model = T5SummarizationModel(
        model_name=args.model_name,
        cache_dir=args.model_cache_dir,
        device=device,
        use_fp16=args.fp16
    )
    base_results = evaluate_model(
        model=base_model.model,
        tokenizer=tokenizer,
        data_loader=test_loader,
        device=device,
        max_length=INFERENCE_CONFIG["max_length"],
        num_beams=INFERENCE_CONFIG["num_beams"]
    )
    print("原版T5评估结果:")
    print_evaluation_results(base_results)
    tuned_results = None
    ckpt = args.resume_from_checkpoint
    if os.path.exists(ckpt):
        print("评估微调T5...")
        tuned_model = T5SummarizationModel(
            model_name=args.model_name,
            cache_dir=args.model_cache_dir,
            device=device,
            use_fp16=args.fp16
        )
        tuned_model.load_model(ckpt)
        tuned_results = evaluate_model(
            model=tuned_model.model,
            tokenizer=tokenizer,
            data_loader=test_loader,
            device=device,
            max_length=INFERENCE_CONFIG["max_length"],
            num_beams=INFERENCE_CONFIG["num_beams"]
        )
        print("微调T5评估结果:")
        print_evaluation_results(tuned_results)
    comp = {"pretrained": base_results}
    if tuned_results is not None:
        comp["fine_tuned"] = tuned_results
    comp_path = os.path.join(results_dir, "comparison_results.json")
    with open(comp_path, "w") as f:
        json.dump(comp, f, indent=2)
    print(f"对比结果已保存到: {comp_path}")
    try:
        if tuned_results is not None:
            print("\n指标对比")
            print("{:<10} {:>10} {:>12} {:>10}".format("Metric", "Pretrained", "Fine-tuned", "Delta"))
            for m in args.metrics:
                b = comp["pretrained"]["metrics"].get(m, 0.0)
                t = comp["fine_tuned"]["metrics"].get(m, 0.0)
                d = t - b
                s = "+" if d >= 0 else ""
                print("{:<10} {:>10.4f} {:>12.4f} {:>10s}{:.4f}".format(m.upper(), b, t, s, d))
        else:
            print("\n仅原版模型指标")
            for m in args.metrics:
                b = comp["pretrained"]["metrics"].get(m, 0.0)
                print("{:<10} {:>10.4f}".format(m.upper(), b))
    except Exception:
        pass
    try:
        metrics = args.metrics
        pre = [comp["pretrained"]["metrics"].get(m, 0) for m in metrics]
        ft = [comp.get("fine_tuned", {"metrics": {}})["metrics"].get(m, 0) for m in metrics]
        x = list(range(len(metrics)))
        plt.figure(figsize=(8, 4))
        plt.bar([i - 0.2 for i in x], pre, width=0.4, label="Pretrained")
        plt.bar([i + 0.2 for i in x], ft, width=0.4, label="Fine-tuned")
        plt.xticks(x, [m.upper() for m in metrics])
        plt.ylabel("Score")
        plt.title("T5 Summarization Comparison")
        plt.legend()
        plt.tight_layout()
        for i, v in enumerate(pre):
            plt.text(i - 0.2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        for i, v in enumerate(ft):
            plt.text(i + 0.2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        out_dir = os.path.dirname(args.plot_output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(args.plot_output)
        print(f"可视化已保存到: {args.plot_output}")
    except Exception as e:
        print("生成可视化失败:", str(e))
    try:
        if args.report_output:
            lines = []
            lines.append("# T5 摘要对比报告")
            lines.append("")
            lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
            lines.append("## 指标")
            for m in args.metrics:
                b = comp['pretrained']['metrics'].get(m, 0)
                t = comp.get('fine_tuned', {'metrics': {}})['metrics'].get(m, 0)
                d = t - b
                s = "+" if d >= 0 else ""
                lines.append(f"- {m.upper()}: Pretrained={b:.4f} Fine-tuned={t:.4f} Δ={s}{d:.4f}")
            lines.append("")
            lines.append("## 示例")
            n = max(0, int(args.examples))
            pre_preds = comp['pretrained']['predictions'][:n]
            pre_refs = comp['pretrained']['references'][:n]
            for i in range(len(pre_preds)):
                lines.append(f"### 示例 {i+1}")
                lines.append(f"- 参考: {pre_refs[i]}")
                lines.append(f"- 原版: {pre_preds[i]}")
                if 'fine_tuned' in comp:
                    ft_preds = comp['fine_tuned']['predictions'][:n]
                    if i < len(ft_preds):
                        lines.append(f"- 微调: {ft_preds[i]}")
            out_dir = os.path.dirname(args.report_output)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)
            with open(args.report_output, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"报告已保存到: {args.report_output}")
    except Exception as e:
        print("生成报告失败:", str(e))
    try:
        if args.csv_output:
            out_dir = os.path.dirname(args.csv_output)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)
            rows = []
            for m in args.metrics:
                b = comp['pretrained']['metrics'].get(m, 0.0)
                t = comp.get('fine_tuned', {'metrics': {}})['metrics'].get(m, 0.0)
                d = t - b
                rows.append([m.upper(), f"{b:.6f}", f"{t:.6f}", f"{d:.6f}"])
            with open(args.csv_output, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Metric", "Pretrained", "Fine-tuned", "Delta"])
                w.writerows(rows)
            print(f"CSV已保存到: {args.csv_output}")
    except Exception as e:
        print("生成CSV失败:", str(e))


if __name__ == "__main__":
    main()
