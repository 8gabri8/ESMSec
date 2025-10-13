############################################################################################################################
### EMBEDDINGS & UMAP
############################################################################################################################


import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange, tqdm
import umap
import pandas as pd
import numpy as np
import torch
import matplotlib.patches as mpatches



@torch.no_grad()
def extract_embeddings(net, dl, device, cls_index=0, return_numpy=True, from_precomputed_embs=False):
    """
    Extract embeddings from a trained EsmDeepSec model.

    Args:
        net: EsmDeepSec instance (already loaded with weights).
        dl: yields (input_ids, attention_mask, label, names).
        device: torch.device
        cls_index: token index to use as CLS if pooler_output not available (default 0).
        return_numpy: if True, returned tensors are numpy arrays on CPU; otherwise torch tensors on CPU.

    Returns:
        tuple:
            results: dict mapping embedding key -> stacked embeddings (N x D)
            names_list: list of protein names
            labels_array: np.array of true labels
            preds_array: np.array of predicted labels
    """

    net.eval()
    net = net.to(device)

    # storage: will be initialized dynamically based on what the model returns
    buffers = None
    names_list = []
    labels_list = []
    preds_list = []

    for batch in tqdm(dl, desc=f"Batch", unit=" batch", leave=False):

        seq = batch.get("input_ids").to(device)
        attention_mask = batch.get("attention_mask").to(device)
        label = batch.get("label").to(device)
        emb = batch.get("embs").to(device)
        names = batch.get("name")


        # Forward pass with return_embs=True
        if from_precomputed_embs:
            logits, embs = net(precomputed_embs=emb, return_embs=True) 
        else:
            logits, embs = net(seq, attention_mask=attention_mask, return_embs=True)  # seq --> [batch_size, 2], TOKENISEd protein sequences

        
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        # Flatten nested dicts (e.g., class_head_embs might be a dict)
        flat_embs = {}
        for k, v in embs.items():
            if isinstance(v, dict):
                # If value is a dict, flatten it with prefixed keys
                for sub_k, sub_v in v.items():
                    flat_embs[f"{k}_{sub_k}"] = sub_v
            else:
                flat_embs[k] = v
        
        # Initialize buffers on first batch based on what model actually returns
        if buffers is None:
            buffers = {k: [] for k in flat_embs.keys()}
        
        # Store embeddings (move to CPU)
        for k, v in flat_embs.items():
            if v is not None:
                buffers[k].append(v.cpu())
        
        # Store names, labels, predictions
        names_list.extend(names)
        labels_list.extend(label.cpu().tolist())
        preds_list.extend(preds.cpu().tolist())

    # Concatenate all batches
    results = {}
    for k, list_of_tensors in buffers.items():
        if len(list_of_tensors) == 0:
            results[k] = None
            continue
        stacked = torch.cat(list_of_tensors, dim=0)
        results[k] = stacked.numpy() if return_numpy else stacked

    return results, names_list, np.array(labels_list), np.array(preds_list)


def compute_umap_tensors(embeddings_dict, n_neighbors=15, min_dist=0.1, random_state=None):
    """
    Compute 2D UMAP projections for each embedding type and return as tensors.

    Args:
        embeddings_dict (dict): dict of embeddings, e.g., output of extract_embeddings()
        n_neighbors (int): UMAP n_neighbors parameter
        min_dist (float): UMAP min_dist parameter
        random_state (int): random seed for reproducibility

    Returns:
        dict: keys are embedding types, values are 2D embeddings as torch.FloatTensor
    """

    umap_tensors = {}

    for key, emb in embeddings_dict.items():

        if emb is None:
            print(f"Skipping {key}: embedding is None")
            continue

        print(f"Computing UMAP for {key} with shape {emb.shape if emb is not None else None}...")
        
        # Convert numpy to tensor if needed
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        
        emb_2d = umap.UMAP(n_neighbors=n_neighbors, 
                           min_dist=min_dist, 
                           random_state=random_state, 
                           n_jobs=-1 #Prioritize speed with parallelism, sacrifice exact reproducibility
                           ).fit_transform(emb)
        umap_tensors[key] = torch.tensor(emb_2d, dtype=torch.float32)

    return umap_tensors


def plot_umap_embeddings(umap_embs, names, labels, preds, embedding_keys=None,
                         class_palette=None, corr_palette=None):
    """
    Plot UMAP embeddings for multiple embedding types with True/Pred/Correct info.

    Args:
        umap_embs (dict): dict of 2D embeddings (keys like 'esm_mean', 'esm_cls') as np.array or torch.Tensor
        names (list): sequence/protein names
        labels (array-like): true class labels
        preds (array-like): predicted class labels
        embedding_keys (list, optional): subset of embedding types to plot
        class_palette (dict, optional): color mapping for classes
        corr_palette (dict, optional): color mapping for correctness
    """
    if embedding_keys is None:
        embedding_keys = list(umap_embs.keys())

    if class_palette is None:
        class_palette = {'1': 'tab:blue', '0': 'tab:orange'}
    if corr_palette is None:
        corr_palette = {'correct': 'grey', 'wrong': 'red'}

    labels = np.array(labels).astype(str)
    preds = np.array(preds).astype(str)

    # Build DataFrames for each embedding type
    dfs = {}
    for key in embedding_keys:
        if key not in umap_embs:
            continue
        emb = umap_embs[key]
        if emb is None:
            continue
        if isinstance(emb, torch.Tensor):
            emb_np = emb.cpu().numpy() if emb.is_cuda else emb.numpy()
        else:
            emb_np = np.array(emb)
        
        df = pd.DataFrame({
            'Name': names,
            'UMAP1': emb_np[:, 0],
            'UMAP2': emb_np[:, 1],
            'TrueClass': labels,
            'PredClass': preds
        })
        df['CorrectStr'] = np.where(labels == preds, 'correct', 'wrong')
        dfs[key] = df

    n_embs = len(dfs)
    if n_embs == 0:
        raise ValueError("No valid embeddings to plot.")

    # Create subplots: 3 rows x n_embs columns
    fig, axes = plt.subplots(3, n_embs, figsize=(6*n_embs, 18), squeeze=False)

    for col_idx, (key, df) in enumerate(dfs.items()):
        # Row 0: True labels
        sns.scatterplot(x='UMAP1', y='UMAP2', hue='TrueClass', palette=class_palette,
                        data=df, alpha=0.8, s=50, ax=axes[0, col_idx], legend=False)
        axes[0, col_idx].set_title(f"{key.replace('_', ' ').title()} - True Labels")
        axes[0, col_idx].grid(True, alpha=0.3)

        # Row 1: Predicted labels
        sns.scatterplot(x='UMAP1', y='UMAP2', hue='PredClass', palette=class_palette,
                        data=df, alpha=0.8, s=50, ax=axes[1, col_idx], legend=False)
        axes[1, col_idx].set_title(f"{key.replace('_', ' ').title()} - Predicted Labels")
        axes[1, col_idx].grid(True, alpha=0.3)

        # Row 2: Correct vs Wrong
        df_correct = df[df['CorrectStr'] == 'correct']
        df_wrong = df[df['CorrectStr'] == 'wrong']

        axes[2, col_idx].scatter(df_correct['UMAP1'], df_correct['UMAP2'],
                                 c=corr_palette['correct'], alpha=0.6, s=40, label='correct', edgecolor='none')
        axes[2, col_idx].scatter(df_wrong['UMAP1'], df_wrong['UMAP2'],
                                 c=corr_palette['wrong'], alpha=0.9, s=90, label='wrong', edgecolor='k', linewidth=0.4)
        axes[2, col_idx].set_title(f"{key.replace('_', ' ').title()} - Correct vs Wrong")
        axes[2, col_idx].set_xlabel("UMAP 1")
        axes[2, col_idx].set_ylabel("UMAP 2")
        axes[2, col_idx].grid(True, alpha=0.3)

    # Legends
    class_patches = [mpatches.Patch(color=v, label=k) for k, v in class_palette.items()]
    corr_patches = [mpatches.Patch(color=v, label=k) for k, v in corr_palette.items()]

    fig.legend(handles=class_patches, title='Class', loc='upper right', bbox_to_anchor=(0.98, 0.95))
    fig.legend(handles=corr_patches, title='Prediction correctness', loc='upper right', bbox_to_anchor=(0.98, 0.65))

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.show()

    return dfs