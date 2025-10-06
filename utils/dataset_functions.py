
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


def filter_gene_sets_by_keywords(
    df,
    include_pattern = r"(^PROLIFERA|\bPROLIFER\b|_PROLIFER_|_CYCLING_|^CELL_CYCLE_|_CELL_CYCLE_|_CC_|_G1_|_S_PHASE_|_G2_|_M_PHASE_|\bMITOSIS\b|\bCYCLIN\b|\bCDK\b|\bCHECKPOINT\b|\bGS1\b|\bGS2\b)",
    exclude_pattern= r"(MEIOSIS|FATTY_ACID_CYCLING_MODEL)",
    col_name = "set_name"
):
    """
    Filters a DataFrame of gene sets by regex patterns applied to 'set_name'.

    Args:
        df (pd.DataFrame): DataFrame with a 'set_name' column.
        include_pattern (str): Regex pattern for inclusion.
        exclude_pattern (str): Regex pattern for exclusion.

    Returns:
        pd.DataFrame: Filtered DataFrame matching include_pattern but not exclude_pattern.
    """
    if col_name not in df.columns:
        raise ValueError(f"DataFrame must contain a {col_name} column")

    # Apply inclusion and exclusion regex filters
    mask_include = df[col_name].str.contains(include_pattern, case=False, regex=True)
    mask_exclude = df[col_name].str.contains(exclude_pattern, case=False, regex=True)

    filtered_df = df[mask_include & ~mask_exclude].copy()
    return filtered_df


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







