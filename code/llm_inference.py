import os  # 用于处理操作系统相关的功能，如文件路径、环境变量等
import sys
import json  # 用于处理 JSON 数据格式的编码和解码
import argparse  # 用于处理命令行参数解析
from datetime import datetime  # 用于处理日期和时间相关操作
from concurrent.futures import as_completed  # 用于处理并发任务的结果迭代
from concurrent.futures import ThreadPoolExecutor  # 用于创建线程池并执行并发任务

import pandas as pd  # 用于数据处理和分析
from openai import OpenAI  # 从 openai 模块导入 OpenAI 类


def config():
    # 创建一个 argparse.ArgumentParser 对象，这个对象用于解析命令行参数。
    parser = argparse.ArgumentParser(
        description="Causal Gene Identification Parameters"
    )
    # 添加一个名为 dataset 的命令行参数，类型为字符串，默认值为 "LUAD"，帮助信息为 "Cancer dataset."，可选值为 ["LUAD", "LUSC", "BLCA", "BRCA", "KIRC", "LIHC"]。
    parser.add_argument(
        "--dataset",
        type=str,
        default="LUAD",
        help="Cancer dataset.",
        choices=["LUAD", "LUSC", "BLCA", "BRCA", "KIRC", "LIHC"],
    )
    # 实验编号字符串，默认值为 "1"，帮助信息为 "The number of the experiment."。
    parser.add_argument(
        "--experiment_id", type=str, default="1", help="The number of the experiment."
    )

    # parser.add_argument("--api_key", type=str, default=None, help="OpenAI api key.")
    # parser.add_argument("--llm", type=str, default="gpt-4o-mini-2024-07-18", help="LLM endpoint.")

    # 有效地捕获并处理 ArgumentError 异常，并提供友好的错误信息和帮助信息。
    try:
        return vars(
            parser.parse_args()
        )  # 返回的是一个字典，字典的键是参数，值是参数的值
    except argparse.ArgumentError as e:
        print(f"Error parsing arguments: {e}\n")
        parser.print_help()
        sys.exit(1)


def inference_gene(gene_and_id_and_disease, info_dict):
    # 使用 split() 函数将一个包含基因符号、Entrez Gene ID 和癌症名称的字符串按照指定的分隔符 "__" 分解成独立的变量
    gene_symbol, gene_id, cancer_name = gene_and_id_and_disease.split(
        "__"
    )  # 基因符号、Entrez Gene ID 和癌症名称

    info1 = info_dict["summary_info"]  # 摘要信息
    info2 = info_dict["description"]  # 基因全称
    info3 = info_dict["official_symbol"]  # 基因符号

    if info1 == "NaN":  # 某些基因可能没有概要信息
        gene_info = f"1. Official Full Name: {info2}.\n2. Official Symbol: {info3}."
    else:
        gene_info = f"1. Summary Information: {info1}\n2. Official Full Name: {info2}.\n3. Official Symbol: {info3}."

    system_instruction = f"You are an expert in the fields of molecular biology, functional genomics, cancer research, and precision medicine, with deep insights into causal gene identification. Moreover, you are an expert at causal inference. You comprehend that only a very small number of genes are true causal genes for {cancer_name}, and these causal genes play key roles in the initiation and progression of {cancer_name}. You are also aware that correlation does not imply causation due to confounding factors. Therefore, it is crucial to identify true causal genes for {cancer_name}, not just related genes."

    prompt_template = f"""Determining causal genes will help researchers better understand the molecular mechanisms and signaling pathways involved in the initiation and progression of {cancer_name}. This understanding is very important for developing new drugs and targeted therapies for this cancer type.

Here is additional information about the {gene_symbol} gene:
```
{gene_info}
```
Your task now is to infer whether a causal relationship exists between the {gene_symbol} gene and {cancer_name}.
Causality is defined as follows: A causal relationship between the variables t and y exists if and only if, with all other factors being equal, a change in t leads to a corresponding change in y. In this relationship, t is the cause, and y is the effect.
To solve the problem, do the following:
1 - Analyze the biological functions of the {gene_symbol} gene.
2 - Analyze the molecular mechanisms and signaling pathways involved in the initiation and progression of {cancer_name}.
3 - Evaluate the potential role of the {gene_symbol} gene in the initiation and progression of {cancer_name}.
4 - Analyze whether there is a reasonable mechanism to explain how the {gene_symbol} gene plays a causal role in the initiation and progression of {cancer_name}.
5 - Assess the clinical value of aberrant {gene_symbol} gene alterations as a diagnostic marker for {cancer_name}.
6 - Assess the clinical value of targeting {gene_symbol} in therapeutic strategies for {cancer_name}.
7 - Assess the prognostic value of {gene_symbol} in patients with {cancer_name}, referencing clinical study outcomes.
8 - Comprehensively consider the key findings from all previous steps, the strength of evidence, and the biological plausibility of the proposed mechanism, to determine whether a causal relationship exists between the {gene_symbol} gene and {cancer_name}. If the causality can be fully established, the result should be expressed as <causality>; if not, the result should be expressed as <no causality>.
Ensure that your inference processes are accurate, logical, and responsible. Provide readable expert-level explanations for the final inference; the more detailed, the better.

Output results strictly in the following format:
Result: <final inference result>

Explanations: <readable expert-level explanations for the final inference result>
""".strip()

    try:
        client = OpenAI(
            api_key="sk-**************************************************************",
            base_url="https://api.openai.com/v1",
            max_retries=3,
            timeout=60,
        )
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # 2024 年 7 月 18 日发布的 gpt-4o-mini 模型。
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt_template},
            ],
            n=1,
            temperature=0.10,  # 请确保将温度值设定得较低，确定性。
            seed=42,
            max_tokens=1024,  # 最大生成 1024 个 tokens。
            logprobs=True,
            top_logprobs=3,
            presence_penalty=0,
            frequency_penalty=0,
        )
        # 从 chat_completion 对象中提取数据，将其转换成一个字典对象。
        response_dict = chat_completion.model_dump()  # 将一个模型对象转换为字典形式

        json_str = json.dumps(response_dict, ensure_ascii=False, indent=2)
        save_file = f"./Iterations/{dataset}/{dataset}_{experiment_id}/{gene_symbol}__{gene_id}.json"
        with open(save_file, "w", encoding="utf-8") as f:
            f.write(json_str)

        return f"{gene_symbol}__{gene_id} \t Successful inference!"

    except Exception as e:  # 记录推断成功或失败的信息
        record_time = datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")

        return f"{gene_symbol}__{gene_id} \t Inference failed! \t {record_time} \n {e}"


if __name__ == "__main__":  # 这个代码文件直接作为脚本来执行
    # 调用 config() 函数获取命令行参数，并将结果存储在 args 中。
    args = config()
    print(type(args["dataset"]), type(args["experiment_id"]))

    # 打印数据集名称和实验编号：
    dataset, experiment_id = args["dataset"], args["experiment_id"]
    print(f"Dataset name: {dataset} \t Experiment id: {experiment_id}")

    # 定义一个字典 disease_dict，将 TCGA 数据集中的癌症类型缩写映射为全称。
    disease_dict = {
        "LUAD": "lung adenocarcinoma",
        "LUSC": "lung squamous cell carcinoma",
        "BLCA": "bladder urothelial carcinoma",
        "BRCA": "breast invasive carcinoma",
        "KIRC": "kidney renal clear cell carcinoma",
        "LIHC": "liver hepatocellular carcinoma",
    }

    with open("./data/gene_to_id.json", "r") as f:
        gene_to_id = json.load(f)

    with open("./data/genes_info_clean.json", "r") as f:
        genes_info = json.load(f)

    tcga_genes = [gene for gene in gene_to_id.keys()]

    # 使用 pandas 的 read_csv 函数读取指定路径下的 csv 文件，并将结果存储在 df 中。打印 df 的内容。
    df = pd.read_csv(f"./dataset/{dataset}_TPM.csv", index_col=0)
    print(df)

    # 从 df 的列名中提取出所有的基因名称，并存储在 tcga_genes 中。
    # tcga_genes = [gene for gene in df.columns[:-1]]

    experiment_dir = f"./Iterations/{dataset}/{dataset}_{experiment_id}/"
    os.makedirs(experiment_dir, exist_ok=True)
    # 获取已经保存过结果的基因数量，并打印出来。
    save_genes = {
        i.split("__")[0]
        for i in os.listdir(f"./Iterations/{dataset}/{dataset}_{experiment_id}/")
    }
    print("Saved gene nums:", len(save_genes))

    # 从 tcga_genes 中筛选出未被保存的基因，并存储在 genes 中。
    genes = [gene for gene in tcga_genes if gene not in save_genes]
    print("gene nums for GPT-4o mini:", len(genes))

    # 根据筛选后的基因和基因名称、基因 ID 和疾病名称生成查询语句，并存储在 queries 中。
    queries = [f"{gene}__{gene_to_id[gene]}__{disease_dict[dataset]}" for gene in genes]

    # 记录当前时间：
    start = datetime.now()

    # 创建一个线程池 executor，设置其最大工作线程数为 32。
    with ThreadPoolExecutor(max_workers=32) as executor:
        # 将每个推断任务提交到线程池中，并将返回的 Future 对象存储在 futures 中。
        futures = [
            executor.submit(inference_gene, query, genes_info[query.split("__")[1]])
            for query in queries
        ]

        # 等待所有任务完成，并打印出包含 "Inference failed!" 的结果。
        for future in as_completed(futures):
            if "Inference failed!" in future.result():
                print(future.result())

    # 计算整个过程所花费的时间，并将其打印出来。
    delta = (datetime.now() - start).total_seconds()
    print("用时：{:.6f}s".format(delta))

    # python llm_inference.py --dataset LUAD --experiment_id 1
