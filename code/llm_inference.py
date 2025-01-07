import os
import sys
import json
import argparse
from datetime import datetime
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from openai import OpenAI


def config():

    parser = argparse.ArgumentParser(
        description="Causal Gene Identification Parameters"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="LUAD",
        help="Cancer dataset.",
        choices=["LUAD", "LUSC", "BLCA", "BRCA", "KIRC", "LIHC"],
    )

    parser.add_argument(
        "--experiment_id", type=str, default="1", help="The number of the experiment."
    )

    try:
        return vars(
            parser.parse_args()
        )
    except argparse.ArgumentError as e:
        print(f"Error parsing arguments: {e}\n")
        parser.print_help()
        sys.exit(1)


def inference_gene(gene_and_id_and_disease, info_dict):

    gene_symbol, gene_id, cancer_name = gene_and_id_and_disease.split("__")

    info1 = info_dict["summary_info"]
    info2 = info_dict["description"]
    info3 = info_dict["official_symbol"]

    if info1 == "NaN":
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
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt_template},
            ],
            n=1,
            temperature=0.10,
            seed=42,
            max_tokens=1024,
            logprobs=True,
            top_logprobs=3,
            presence_penalty=0,
            frequency_penalty=0,
        )
        response_dict = chat_completion.model_dump()

        json_str = json.dumps(response_dict, ensure_ascii=False, indent=2)
        save_file = f"./Iterations/{dataset}/{dataset}_{experiment_id}/{gene_symbol}__{gene_id}.json"
        with open(save_file, "w", encoding="utf-8") as f:
            f.write(json_str)

        return f"{gene_symbol}__{gene_id} \t Successful inference!"

    except Exception as e:
        record_time = datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")

        return f"{gene_symbol}__{gene_id} \t Inference failed! \t {record_time} \n {e}"


if __name__ == "__main__":

    args = config()
    print(type(args["dataset"]), type(args["experiment_id"]))

    dataset, experiment_id = args["dataset"], args["experiment_id"]
    print(f"Dataset name: {dataset} \t Experiment id: {experiment_id}")

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

    df = pd.read_csv(f"./dataset/{dataset}_TPM.csv", index_col=0)
    print(df)

    experiment_dir = f"./Iterations/{dataset}/{dataset}_{experiment_id}/"
    os.makedirs(experiment_dir, exist_ok=True)

    save_genes = {
        i.split("__")[0]
        for i in os.listdir(f"./Iterations/{dataset}/{dataset}_{experiment_id}/")
    }
    print("Saved gene nums:", len(save_genes))

    genes = [gene for gene in tcga_genes if gene not in save_genes]
    print("gene nums for GPT-4o mini:", len(genes))

    queries = [f"{gene}__{gene_to_id[gene]}__{disease_dict[dataset]}" for gene in genes]

    start = datetime.now()

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(inference_gene, query, genes_info[query.split("__")[1]])
            for query in queries
        ]

        for future in as_completed(futures):
            if "Inference failed!" in future.result():
                print(future.result())

    delta = (datetime.now() - start).total_seconds()
    print("用时：{:.6f}s".format(delta))
