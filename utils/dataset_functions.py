
import json
import os
import re
import pandas as pd
from collections import defaultdict
from scipy.stats import hypergeom, binom
import numpy as np


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


def per_cluster_hypergeom_test(cluster_row, total_genes, total_positive):
    """
    Tests if cluster is enriched for positive genes
    
    Parameters:
    - cluster_row: row from dataframe with n_genes and n_genes_positive
    - total_genes: total number of genes across all clusters
    - total_positive: total number of positive genes across all clusters
    """
    # Hypergeometric parameters:
    # M = total population size
    # n = number of success states in population (positive genes)
    # N = number of draws (genes in cluster)
    
    M = total_genes
    n = total_positive
    N = cluster_row['n_genes']
    k = cluster_row['n_genes_positive']
    
    # One-tailed p-value (testing if cluster has MORE positive genes than expected)
    p_value = 1 - hypergeom.cdf(k - 1, M, n, N)
    
    # Calculate probability of observing exactly k positive genes
    prob = hypergeom.pmf(k, M, n, N)
    
    return prob, p_value

def sample_sampled_from_single_row(row, min_sample_n, prot_col, gene_col, probs_col):

    N = min(len(row[prot_col]), min_sample_n)

    if probs_col != None:
        probs = row[probs_col] #use precompued prbs
    else:
        probs=None # unirform prob

    # sample indices
    sampled_indices = np.random.choice(len(row[prot_col]), size=N, replace=False, p=probs) 

    # covert to array
    proteins_array = np.array(row[prot_col])
    genes_array = np.array(row[gene_col])

    # return sliced
    return proteins_array[sampled_indices], genes_array[sampled_indices]



