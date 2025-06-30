# 版权所有 (c) Sebastian Raschka，基于Apache License 2.0许可(详见LICENSE.txt)
# 书籍《从零开始构建大语言模型》源代码
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch
#
# 基于第7章代码的最小指令微调文件

import json
import psutil
from tqdm import tqdm
import urllib.request


def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    # 创建数据负载字典
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # 以下设置为确保确定性响应
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    # 将字典转换为JSON格式字符串并编码为字节
    payload = json.dumps(data).encode("utf-8")

    # 创建请求对象，设置POST方法并添加必要头部
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # 发送请求并捕获响应
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # 读取并解码响应
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


def check_if_running(process_name):
    """检查指定进程是否正在运行"""
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


def format_input(entry):
    """格式化指令和输入用于提示"""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def main(file_path):
    """主函数：评估模型响应"""
    # 检查Ollama是否运行
    ollama_running = check_if_running("ollama")

    if not ollama_running:
        raise RuntimeError("Ollama未运行，请先启动Ollama")
    print("Ollama running:", check_if_running("ollama"))

    # 加载测试数据
    with open(file_path, "r") as file:
        test_data = json.load(file)

    # 生成模型评分
    model = "llama3"
    scores = generate_model_scores(test_data, "model_response", model)
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")


def generate_model_scores(json_data, json_key, model="llama3"):
    """为模型响应生成评分"""
    scores = []
    for entry in tqdm(json_data, desc="评分条目"):
        if entry[json_key] == "":
            scores.append(0)
        else:
            # 构建评分提示
            prompt = (
                f"Given the input `{format_input(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"score the model response `{entry[json_key]}`"
                f" on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
            score = query_model(prompt, model)
            try:
                scores.append(int(score))
            except ValueError:
                print(f"无法转换评分: {score}")
                continue

    return scores


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate model responses with ollama"
    )
    parser.add_argument(
        "--file_path",
        required=True,
        help=(
            "The path to the test dataset `.json` file with the"
            " `'output'` and `'model_response'` keys"
        )
    )
    args = parser.parse_args()

    main(file_path=args.file_path)
