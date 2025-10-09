
import json
import os
import re
import pandas as pd
from collections import defaultdict


def load_json_folder_to_df(folder_path):
    """
    Reads all JSON files in a folder and combines them into a single pandas DataFrame.

    Args:
        folder_path (str): Path to the folder containing JSON files.

    Returns:
        pd.DataFrame: Combined DataFrame with one row per gene set.
    """
    data_rows = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

                for key, value in json_data.items():
                    record = {
                        "set_name": key,
                        "collection": value.get("collection"),
                        "systematicName": value.get("systematicName"),
                        "pmid": value.get("pmid"),
                        "exactSource": value.get("exactSource"),
                        "externalDetailsURL": value.get("externalDetailsURL"),
                        "msigdbURL": value.get("msigdbURL"),
                        "geneSymbols": value.get("geneSymbols", []),
                        "filteredBySimilarity": value.get("filteredBySimilarity", []),
                        "externalNamesForSimilarTerms": value.get("externalNamesForSimilarTerms", []),
                        "source_file": filename,
                    }
                    data_rows.append(record)

    df = pd.DataFrame(data_rows)
    return df


def gene_set_counts(df, geneset_col_name="set_name", genes_col_name="geneSymbols", split_symbol=","):
    """
    Create a new DataFrame showing each gene and how many gene sets it appears in.
    """
    gene_to_sets = defaultdict(set) #dictionary where each key is a gene-name and each value is a set of gene-set names it appears in.

    for _, row in df.iterrows():
        set_name = row[geneset_col_name]
        genes = row[genes_col_name]
        # make sure genes is iterable (list)
        if isinstance(genes, str):
            genes = [g.strip() for g in genes.split(split_symbol) if g.strip()]
        for g in genes: #For each gene, add the current set name to that gene’s set in gene_to_sets
            gene_to_sets[g].add(set_name)
                # defaultdict() --> if g doesn’t exist as a key yet, Python automatically creates an empty set for it

    # build new DataFrame
    result = pd.DataFrame(
        [(gene, len(sets)) for gene, sets in gene_to_sets.items()],
        columns=["gene", "geneset_count"]
    ).sort_values("geneset_count", ascending=False).reset_index(drop=True)

    return result







