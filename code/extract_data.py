import json
import pandas as pd
from tqdm import tqdm
from collections import Counter


def is_zero(num, tolerance=1e-6):

    if isinstance(num, int):
        return num == 0
    elif isinstance(num, float):
        return abs(num) < tolerance
    else:
        raise ValueError("Unsupported numeric type")


def load_metadata(dataset):

    with open(f"./{dataset}/metadata.cart.2023-12-25.json", "r") as f:
        return json.load(f)


def load_gene_ids():

    with open("gene_ensg_id_dict.json", "r") as f:
        gene_id_data = json.load(f)
    with open("gene_to_id.json", "r") as f:
        tmp_gene_id = json.load(f)
    selected_genes = {i for i in tmp_gene_id.keys()}
    gene_id_data = [(k, v) for k, v in gene_id_data.items() if k in selected_genes]
    id_to_gene = {v: k for k, v in gene_id_data}
    gene_ids = [gene_and_id[1] for gene_and_id in gene_id_data]

    return gene_ids, id_to_gene


for dataset in ["LUAD", "LUSC", "BLCA", "BRCA", "KIRC", "LIHC"]:
    print("-" * 100)
    print(f"Dataset: {dataset}")

    cart = load_metadata(dataset)
    tmp1 = {item["associated_entities"][0]["entity_submitter_id"] for item in cart}
    tmp2 = {item["associated_entities"][0]["case_id"] for item in cart}
    print(f"TCGA ID Nums: {len(tmp1)} \t Case ID Nums: {len(tmp2)}")
    tmp3 = [
        "-".join(item["associated_entities"][0]["entity_submitter_id"].split("-")[:4])
        for item in cart
    ]
    print(Counter(tmp3).most_common(10))

    samples, data = [], []
    samples_set = set()
    for item in tqdm(cart):
        file_id = item["file_id"]
        file_name = item["file_name"]
        tcga_id = item["associated_entities"][0]["entity_submitter_id"]
        case_id = item["associated_entities"][0]["case_id"]

        sample_vial = tcga_id.split("-")[3]

        if sample_vial not in {"01A", "11A"}:
            continue
        sample_id = "-".join(tcga_id.split("-")[:4])
        if sample_id in samples_set:
            continue

        sample_path = f"./{dataset}/samples_info/{file_id}/{file_name}"
        df = pd.read_csv(sample_path, sep="\t", skiprows=1)
        df = df.iloc[4:, :]
        df = df[~df["gene_id"].str.contains("_PAR_Y")]
        df.reset_index(drop=True, inplace=True)

        sample_label = 1 if sample_vial == "01A" else 0

        tmp_dict = {
            ensg_id: gene_expression
            for ensg_id, gene_expression in zip(
                df["gene_id"], df["tpm_unstranded"]
            )
        }
        sample_info = {
            k.split(".")[0]: tmp_dict[k] for k in sorted(tmp_dict.keys())
        }
        sample_info["label"] = sample_label

        samples.append(tcga_id)
        samples_set.add(sample_id)
        data.append(sample_info)

    data_df = pd.DataFrame(data, index=samples)
    print(data_df.iloc[:, :-1].shape)

    gene_ids, id_to_gene = load_gene_ids()

    data_df1 = data_df[gene_ids + ["label"]].copy(deep=True)
    data_df1.columns = [id_to_gene[ensg] for ensg in gene_ids] + ["label"]
    print(data_df1.iloc[:, :-1].shape)

    genes_selected = []
    N, _ = data_df1.shape
    for col in data_df1.iloc[:, :-1].columns:
        percentage_0 = sum([1 if is_zero(value) else 0 for value in data_df1[col]]) / N

        if percentage_0 > 0.50:
            continue
        else:
            genes_selected.append(col)
    genes_selected.append("label")

    data_df1 = data_df1[genes_selected]
    y = data_df1["label"]
    print("After processing:", data_df1.iloc[:, :-1].shape)
    print(y.value_counts())

    data_df1.to_csv(f"./dataset/{dataset}_TPM.csv", encoding="utf-8")
