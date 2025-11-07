from torch.utils.data import Dataset, DataLoader
from safetensors.torch import safe_open, save_file
import torch
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, HDBSCAN
from scipy.cluster.hierarchy import linkage, fcluster

from . import models as my_models


def subset_data_dict(cache_data, indices):
    """
    Create a subset of cache_data using the given indices.
    Keeps tensor fields as tensors and list fields as lists.
    """
    sub_data = {
        "protein": [cache_data["protein"][i] for i in indices],
        "label": cache_data["label"][indices],  # tensor
        "set": [cache_data["set"][i] for i in indices],
        "sequence": [cache_data["sequence"][i] for i in indices],
        "truncated_sequence": [cache_data["truncated_sequence"][i] for i in indices],
        "input_ids": cache_data["input_ids"][indices],          # tensor
        "attention_mask": cache_data["attention_mask"][indices],  # tensor
        "embedding": cache_data["embedding"][indices],          # tensor
    }
    return sub_data

    
class ProteinDataset(Dataset):
    def __init__(self, names, labels, input_ids=None, attention_mask=None, embs=None):
        """
        Generic dataset for protein data.
        Handles optional inputs (input_ids, attention_mask, embs).
        """
        self.names = names
        self.labels = labels
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.embs = embs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "name": self.names[idx],
            "label": self.labels[idx],
        }

        if self.input_ids is not None:
            item["input_ids"] = self.input_ids[idx]
        if self.attention_mask is not None:
            item["attention_mask"] = self.attention_mask[idx]
        if self.embs is not None:
            item["embs"] = self.embs[idx]

        return item

    
def create_dataloader(cache_data, batch_size, shuffle=False):

    # Create dataset
    dataset = ProteinDataset(
        names=cache_data["protein"],
        labels=cache_data["label"],
        input_ids=cache_data["input_ids"],
        attention_mask=cache_data["attention_mask"],
        embs=cache_data["embedding"]
    )

    # Wrap in DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )


def truncate_sequence(sequence, max_length=1000):
    # If sequence > 1000 residues: take first 500 + last 500 residues
    if len(sequence) <= max_length:
        return sequence
    else:
        half_length = (max_length) // 2 
        truncated_sequence = sequence[:half_length] + sequence[-half_length:]

    return truncated_sequence


def create_uniprot_embs(
    esm_model,
    config, 
    cache_data, 
    num_samples, 
    emb_dim
):
    esm_model.eval()

    #first batch will be slow (compiling), rest will be fast
    #esm_model = torch.compile(esm_model, mode='reduce-overhead')

    # Create dataloader for batched processing
    dataloader = create_dataloader(
        cache_data,
        batch_size=config.get("BATCH_SIZE", 32),
        shuffle=False,
    )

    # Prealocate
        # bad for memeory
        # good for velocoty
    if config["TYPE_EMB_FOR_CLASSIFICATION"] == "contextualized_embs":
        context_length = cache_data["input_ids"].shape[1]
        all_embeddings = torch.zeros(num_samples, context_length, emb_dim, dtype=torch.float32)
    else:
        all_embeddings = torch.zeros(num_samples, emb_dim, dtype=torch.float32)
    protein_names = []


    current_idx = 0
    for batch in tqdm(dataloader, desc="Processing protein batches"):
        batch_input_ids = batch["input_ids"].to(config["DEVICE"])
        batch_attention_mask = batch["attention_mask"].to(config["DEVICE"])
        batch_proteins = batch["name"]
        batch_size_actual = len(batch_proteins)

        with torch.no_grad(): #, autocast(dtype=torch.float16)
            outputs_esm = esm_model(
                input_ids=batch_input_ids, 
                attention_mask=batch_attention_mask, 
                return_dict=True
            )

            # Get specific embs from gene contextualised embs
            batch_embeddings = my_models.get_embs_from_context_embs( 
                context_embs_esm=outputs_esm.last_hidden_state,
                attention_mask=batch_attention_mask,
                type_embs=config["TYPE_EMB_FOR_CLASSIFICATION"],
                exclude_cls=True
            )

            # Direct assignment to preallocated tensor (no list append!)
            all_embeddings[current_idx:current_idx + batch_size_actual] = batch_embeddings.cpu()
            protein_names.extend(batch_proteins)
            current_idx += batch_size_actual
    
    print(f"Embeddings shape: {all_embeddings.shape}")

    # Save embeddings with safetensors
    safetensors_path = config["PRECOMPUTED_EMBS_PATH"]
    save_file(
        {"embedding": all_embeddings},
        safetensors_path
    )

    # Save protein names as JSON
    names_path = config["PRECOMPUTED_EMBS_PATH_PROTEIN_NAMES"]
    with open(names_path, 'w', encoding='utf-8') as f:
        json.dump(protein_names, f, ensure_ascii=False)

    print(f"✓ Saved {len(protein_names)} embeddings")
    print(f"  Embeddings: {safetensors_path}")
    print(f"  Names: {names_path}")

    return {
        'embeddings_path': safetensors_path,
        'names_path': names_path,
        'num_proteins': len(protein_names)
    }


def load_embs_safetensor(
    precomputed_embs_path,
    protein_names_path,
    protein_to_select=None
    ):
    
    print(f"Loading embeddings from: {precomputed_embs_path}")

    # Load protein names (fast)
    with open(protein_names_path, 'r', encoding='utf-8') as f:
        all_proteins = json.load(f)
    
    if protein_to_select is None:
        protein_to_select = all_proteins #take all proteins

    # Open embeddings with memory mapping (instant, no RAM usage!)
    with safe_open(precomputed_embs_path, framework="pt", device="cpu") as f:

        all_embeddings_mmap = f.get_tensor("embedding")
        
        print(f"✓ Mapped {len(all_proteins)} embeddings (shape: {all_embeddings_mmap.shape})")    
            
        # Create fast lookup dictionary
        protein_to_idx = {protein: idx for idx, protein in enumerate(all_proteins)}
        
        # Find indices for subset (vectorized - much faster than loop!)
        subset_indices = []
        missing_proteins = []
        
        for p in protein_to_select:
            if p in protein_to_idx:
                subset_indices.append(protein_to_idx[p])
            else:
                missing_proteins.append(p)
        
        # Extract subset with vectorized indexing (only loads needed rows!)
        if subset_indices:
            indices_tensor = torch.tensor(subset_indices, dtype=torch.long)
            
            # This only loads the specific rows from disk (fast with mmap!)
            subset_embeddings_tensor = all_embeddings_mmap[indices_tensor].clone()
            
            # Update cache_data
            #cache_data["embedding"] = subset_embeddings_tensor
            
            print(f"✓ Loaded {len(subset_indices)} / {len(protein_to_select)} embeddings")
            print(f"  Shape: {subset_embeddings_tensor.shape}")
            print(f"  Memory: {subset_embeddings_tensor.element_size() * subset_embeddings_tensor.nelement() / 1024**2:.1f} MB")
        
        if missing_proteins:
            print(f"⚠ Missing {len(missing_proteins)} proteins")
            if len(missing_proteins) <= 10:
                print(f"  Missing: {missing_proteins}")
            else:
                print(f"  First 10 missing: {missing_proteins[:10]}")

    return subset_embeddings_tensor, protein_to_select


# ============================================================================
# CLUSTERING
# ============================================================================

def add_dbscan_clustering(adata, basis='X_umap_PCA', eps=0.5, min_samples=10, key_added='dbscan'):
    """
    Add DBSCAN clustering to adata
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object
    basis : str
        Which embedding to use for clustering
    eps : float
        DBSCAN epsilon parameter (max distance between neighbors)
    min_samples : int
        Minimum samples in neighborhood to form cluster
    key_added : str
        Key to store clustering results
    """
    # Get embedding coordinates
    coords = adata.obsm[basis]
    
    # Run DBSCAN
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = clusterer.fit_predict(coords)
    
    # Convert to categorical (scanpy convention)
    # DBSCAN labels: -1 = noise, 0, 1, 2, ... = clusters
    labels_str = labels.astype(str)
    labels_str[labels == -1] = 'Noise'
    
    adata.obs[key_added] = pd.Categorical(labels_str)
    
    print(f"DBSCAN clustering complete:")
    print(f"  - Number of clusters: {len(np.unique(labels[labels != -1]))}")
    print(f"  - Noise points: {(labels == -1).sum()} ({100*(labels == -1).mean():.1f}%)")
    
    return adata


def add_hdbscan_clustering(adata, basis='X_umap_PCA', min_cluster_size=50, 
                          min_samples=10, key_added='hdbscan'):
    """
    Add HDBSCAN clustering (better for varying density)
    
    Parameters:
    -----------
    min_cluster_size : int
        Minimum cluster size (larger = fewer clusters)
    min_samples : int
        How conservative clustering is (larger = more conservative)
    """
    coords = adata.obsm[basis]
    
    # Run HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=0.0,
        cluster_selection_method='eom',
        metric='euclidean'
    )
    labels = clusterer.fit_predict(coords)
    
    # Store probabilities (confidence of assignment)
    adata.obs[f'{key_added}_probabilities'] = clusterer.probabilities_
    
    # Convert labels
    labels_str = labels.astype(str)
    labels_str[labels == -1] = 'Noise'
    adata.obs[key_added] = pd.Categorical(labels_str)
    
    print(f"HDBSCAN clustering complete:")
    print(f"  - Number of clusters: {len(np.unique(labels[labels != -1]))}")
    print(f"  - Noise points: {(labels == -1).sum()} ({100*(labels == -1).mean():.1f}%)")
    
    return adata


# ============================================================================
# 2. HIERARCHICAL CLUSTERING
# ============================================================================

def add_hierarchical_clustering(adata, basis='X_umap_PCA', n_clusters=20, 
                                linkage_method='ward', key_added='hierarchical'):
    """
    Add hierarchical clustering to adata
    
    Parameters:
    -----------
    n_clusters : int
        Number of clusters to extract
    linkage_method : str
        'ward', 'complete', 'average', 'single'
    """
    coords = adata.obsm[basis]
    
    # Run hierarchical clustering
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method
    )
    labels = clusterer.fit_predict(coords)
    
    adata.obs[key_added] = pd.Categorical(labels.astype(str))
    
    print(f"Hierarchical clustering complete:")
    print(f"  - Number of clusters: {n_clusters}")
    print(f"  - Linkage method: {linkage_method}")
    
    return adata


def add_adaptive_hierarchical_clustering(adata, basis='X_umap_PCA', 
                                        distance_threshold=None, 
                                        key_added='hierarchical_adaptive'):
    """
    Hierarchical clustering with adaptive height cutting
    (doesn't require specifying n_clusters)
    X_PCA
    Parameters:
    -----------
    distance_threshold : float or None
        Maximum distance for merging clusters
        If None, will be auto-determined
    """
    coords = adata.obsm[basis]
    
    # Compute linkage matrix
    Z = linkage(coords, method='ward')
    
    # Auto-determine threshold if not provided
    if distance_threshold is None:
        # Use elbow method: find point of maximum acceleration
        distances = Z[:, 2]
        accelerations = np.diff(np.diff(distances))
        distance_threshold = distances[np.argmax(accelerations) + 1]
        print(f"  Auto-determined distance threshold: {distance_threshold:.3f}")
    
    # Cut tree
    labels = fcluster(Z, distance_threshold, criterion='distance')
    
    adata.obs[key_added] = pd.Categorical(labels.astype(str))
    
    # Store linkage matrix for plotting dendrogram later
    adata.uns[f'{key_added}_linkage'] = Z
    
    print(f"Adaptive hierarchical clustering complete:")
    print(f"  - Number of clusters: {len(np.unique(labels))}")
    print(f"  - Distance threshold: {distance_threshold:.3f}")
    
    return adata