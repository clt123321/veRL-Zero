import re


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    if method == "strict":
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", solution_str, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)

        if thinking_match is None and answer_match is None:
            return -1.0

        if answer_match is None:
            return -1.0

        answer_content = answer_match.group(1).strip()
        answer_numbers = re.findall(r"-?[\d,]+\.?\d*", answer_content)

        if len(answer_numbers) == 0:
            return -1.0

        predicted_answer = answer_numbers[-1].replace(",", "")

        gt_clean = ground_truth.strip().replace(",", "")

        if predicted_answer == gt_clean:
            return score
        else:
            return format_score

    elif method == "flexible":
        answer_numbers = re.findall(r"-?[\d,]+\.?\d*", solution_str)
        if len(answer_numbers) == 0:
            return -1.0

        predicted_answer = answer_numbers[-1].replace(",", "")
        gt_clean = ground_truth.strip().replace(",", "")

        if predicted_answer == gt_clean:
            return score
        else:
            return format_score

    return -1.0


def reward_func(data_source, solution_str, ground_truth, extra_info=None):
    return compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0)
