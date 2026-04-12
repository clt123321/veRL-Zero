import os
import re
from datasets import load_dataset
import pandas as pd

SYSTEM_PROMPT = "请先在 <thinking> 和 </thinking> 标签内写出思考过程，然后将最终答案放在 <answer> 和 </answer> 标签内。"


def extract_solution(solution_str):
    solution = re.search(r"#### (-?[0-9\.,]+)", solution_str)
    if solution is None:
        return None
    final_solution = solution.group(1).replace(",", "")
    return final_solution


def preprocess():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    output_dir = "/root/autodl-tmp/data/gsm8k"
    os.makedirs(output_dir, exist_ok=True)

    data_source = "openai/gsm8k"

    print("正在下载 GSM8K 数据集...")
    dataset = load_dataset(data_source, "main")

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            answer_raw = example.pop("answer")

            question = question_raw + " " + 'Let\'s think step by step and output the final answer after "####".'

            solution = extract_solution(answer_raw)
            if solution is None:
                solution = ""

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = dataset["train"].map(function=make_map_fn("train"), with_indices=True)
    test_dataset = dataset["test"].map(function=make_map_fn("test"), with_indices=True)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"训练集保存至: {train_path} ({len(train_dataset)} 条)")
    print(f"测试集保存至: {test_path} ({len(test_dataset)} 条)")

    train_df = train_dataset.to_pandas()
    print(f"\n训练集列: {list(train_df.columns)}")
    print(f"\n第1条示例:")
    print(f"  data_source: {train_df.iloc[0]['data_source']}")
    print(f"  prompt: {train_df.iloc[0]['prompt']}")
    print(f"  reward_model: {train_df.iloc[0]['reward_model']}")


if __name__ == "__main__":
    preprocess()
