
from torch.optim import AdamW
from torch import nn
from torch.optim import lr_scheduler
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange, tqdm
import umap
import pandas as pd
import numpy as np
import math
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef
from scipy.ndimage import gaussian_filter1d
import matplotlib.patches as mpatches
import warnings
from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm


import dataset as my_dataset



def monitor_gpu_memory():
    """
    Check if a GPU is available and print its current memory usage.

    - If a GPU is available:
        1. Creates a small tensor and moves it to the GPU 
           (ensures CUDA is initialized).
        2. Prints the allocated memory (used by tensors).
        3. Prints the reserved/cached memory (allocated by CUDA for reuse).
    - If no GPU is available:
        Prints a message indicating no GPU.
    """
    if torch.cuda.is_available():
        # Create a small tensor on the GPU to initialize CUDA context
        _ = torch.randn(1, device="cuda")

        # Memory actively used by tensors
        allocated_memory = torch.cuda.memory_allocated()
        # Memory reserved by the caching allocator (may be reused later)
        cached_memory = torch.cuda.memory_reserved()

        print(f"Allocated memory: {allocated_memory / 1024 ** 3:.2f} GB")
        print(f"Cached memory:    {cached_memory / 1024 ** 3:.2f} GB")
    else:
        print("No GPU available.")


############################################################################################################################
### ESM  MODEL
############################################################################################################################


def evaluate_model(net, dl, device, loss_fn=None, split_name="Eval", verbose=True):
    """
    Evaluate the model on a given dataset (train/val/test).
    
    Args:
        net: torch.nn.Module, the model
        dl: torch DataLoader, dataset to evaluate on
        device: torch.device
        loss_fn: loss function (e.g. nn.CrossEntropyLoss)
        split_name: string, e.g. "Train", "Validation", "Test"
        
    Returns:
        metrics dict with:
            - loss (float)
            - accuracy (float)
            - balanced_accuracy (float)
            - f1 (float)
            - mcc (float)
            - probs (tensor, [N, num_classes])
            - probs_class1 (tensor, [N])
            - labels (tensor, [N])
            - pred_labels (tensor, [N])
    """
    net.eval()
    total_loss = torch.zeros(1, dtype=torch.float32, device=device)
    total_correct = torch.zeros(1, dtype=torch.float32, device=device)

    all_probs = [] # list containing the probs of each batch, to be concatenated at the end
    all_probs_class1 = []
    all_labels = []
    all_preds = []

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dl, desc=f"Evaluation", unit=f" {split_name} batch", leave=False):

            seq, attention_mask, label, _ = batch

            seq, attention_mask, label = seq.to(device), attention_mask.to(device), label.to(device)

            output = net(seq, attention_mask=attention_mask)
            probs = torch.softmax(output, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            loss = loss_fn(output, label) # already compute SINGLE BATCH AVERAGE

            total_loss += loss
            total_correct += (preds == label).sum()

            all_probs.append(probs.cpu())
            all_probs_class1.append(probs.cpu()[:, 1])
            all_labels.append(label.cpu())
            all_preds.append(preds.cpu())

    # average loss per batch
    avg_loss = (total_loss / len(dl)).item()
    # accuracy over all samples
    avg_acc = (total_correct / len(dl.dataset)).item()

    # concat all batch results
    all_probs = torch.cat(all_probs, dim=0).cpu()
    all_probs_class1 = torch.cat(all_probs_class1, dim=0).cpu()
    all_labels = torch.cat(all_labels, dim=0).cpu()
    all_preds = torch.cat(all_preds, dim=0).cpu()

    # Convert to numpy for sklearn metrics
    labels_np = all_labels.numpy()
    preds_np = all_preds.numpy()

    ### Calculate additional metrics
    # F1 Score (binary or weighted for multiclass)
    f1 = f1_score(labels_np, preds_np, average='binary' if len(torch.unique(all_labels)) == 2 else 'weighted', zero_division=0)
    # Balanced Accuracy
    balanced_acc = balanced_accuracy_score(labels_np, preds_np)
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(labels_np, preds_np)

    if verbose:
        print(f"\t{split_name} set: Loss: {avg_loss:.4f}, Acc: {avg_acc*100:.2f}%, "
          f"Balanced Acc: {balanced_acc*100:.2f}%, F1: {f1:.4f}, MCC: {mcc:.4f}")

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "balanced_accuracy": balanced_acc,
        "f1": f1,
        "mcc": mcc,
        "probs": all_probs,
        "probs_class1": all_probs_class1,
        "labels": all_labels,
        "pred_labels": all_preds
    }


def train(net, train_dl, valid_dl, test_dl, config):

    # Train
    train_loss_history = []
    train_acc_history = []
    train_balanced_acc_history = []
    train_f1_history = []
    train_mcc_history = []
    train_last_eval = {}

    # Validation
    valid_loss_history = []
    valid_acc_history = []
    valid_balanced_acc_history = []
    valid_f1_history = []
    valid_mcc_history = []
    valid_last_eval = {}

    # Test
    test_loss_history = []
    test_acc_history = []
    test_balanced_acc_history = []
    test_f1_history = []
    test_mcc_history = []
    test_last_eval = {}


    # read info
    device = config["DEVICE"]
    lr = config["LR"]
    step_size = config["LR_DECAY_STEPS_EPOCHS"]
    gamma = config["LR_DECAY_GAMMA"]
    num_epochs = config["NUM_EPOCHS"]
    eval_epoch_freq = config["EVAL_EPOCH_FREQ"]

    # Initialise obj
    optimizer = AdamW(net.class_head.parameters(), lr=lr) # ONLY train the classification head
    exp_lr = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) # step_size=N → the LR changes after step() has been called N times
    loss_fn = nn.CrossEntropyLoss() # binary classfication, needs logits

    # Set model to training mode
    net.train()


    for epoch_idx in trange(1, num_epochs + 1, desc="Training", unit="epoch"):

        net.train() # back to train mode

        for batch in tqdm(train_dl, desc=f"Epoch {epoch_idx}", unit=" train batch", leave=False):
            
            # unpack
            seq, attention_mask, label, _ = batch

            # set model to trainign mode
            net.train()

            #move to device
            seq, attention_mask, label = seq.to(device), attention_mask.to(device), label.to(device)
            
            # Forward apss
            output = net(seq, attention_mask=attention_mask) # seq --> [batch_size, 2], TOKENISEd protein sequences
            # compute loss
            loss = loss_fn(output, label) # crossEntripyLoss() expects RAW logits, so it is correct passing direclty "outpu"
            # clear previous gradients
            optimizer.zero_grad(set_to_none=True)  
            # compute gradients (backpropgation)
            loss.backward()
            # paramers update
            optimizer.step()


        # EVALUATION EVERY TOT EPOCHS
        # NB if num_iters%eval_size != 0 the last iters will not be evaluated
        if epoch_idx % eval_epoch_freq == 0:

            print(f"--- Evaluation at iteration {epoch_idx} ---")

            # Return metrics of current evaluation
            train_metrics = evaluate_model(net, train_dl, device, loss_fn, split_name="Train")
            valid_metrics = evaluate_model(net, valid_dl, device, loss_fn, split_name="Validation")
            test_metrics  = evaluate_model(net, test_dl, device, loss_fn, split_name="Test")

            # Save histories
            train_loss_history.append(train_metrics["loss"])
            train_acc_history.append(train_metrics["accuracy"])
            train_balanced_acc_history.append(train_metrics["balanced_accuracy"])
            train_f1_history.append(train_metrics["f1"])
            train_mcc_history.append(train_metrics["mcc"])

            valid_loss_history.append(valid_metrics["loss"])
            valid_acc_history.append(valid_metrics["accuracy"])
            valid_balanced_acc_history.append(valid_metrics["balanced_accuracy"])
            valid_f1_history.append(valid_metrics["f1"])
            valid_mcc_history.append(valid_metrics["mcc"])

            test_loss_history.append(test_metrics["loss"])
            test_acc_history.append(test_metrics["accuracy"])
            test_balanced_acc_history.append(test_metrics["balanced_accuracy"])
            test_f1_history.append(test_metrics["f1"])
            test_mcc_history.append(test_metrics["mcc"])

            # Save last evaluation results
            train_last_eval = train_metrics
            valid_last_eval = valid_metrics
            test_last_eval = test_metrics

            ### GPU monitoring
            monitor_gpu_memory()

        # increment learning rate decay counter PER EPOCH
        #exp_lr.step() # increments the internal counter

    # claean up
    torch.cuda.empty_cache() 

    return (
        # Train
        train_loss_history,
        train_acc_history,
        train_last_eval,
        
        # Validation
        valid_loss_history,
        valid_acc_history,
        valid_last_eval,
        
        # Test
        test_loss_history,
        test_acc_history,
        test_last_eval
    ) 


def summarize_training(
    train_loss_history, train_acc_history, train_last_eval,
    valid_loss_history, valid_acc_history, valid_last_eval,
    test_loss_history, test_acc_history, test_last_eval
):
    """
    Prints final loss & accuracy for each set and plots loss/accuracy histories.
    """

    # --- Print final metrics ---
    print("\n=== Final Evaluation Metrics ===")
    print(f"Train   -> Loss: {train_last_eval['loss']:.4f}, Accuracy: {train_last_eval['accuracy']*100:.2f}%")
    print(f"Valid   -> Loss: {valid_last_eval['loss']:.4f}, Accuracy: {valid_last_eval['accuracy']*100:.2f}%")
    print(f"Test    -> Loss: {test_last_eval['loss']:.4f}, Accuracy: {test_last_eval['accuracy']*100:.2f}%")

    # --- Prepare data for plotting ---
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14,5))

    # Loss plot
    axes[0].plot(train_loss_history, label="Train Loss", marker='o')
    axes[0].plot(valid_loss_history, label="Valid Loss", marker='o')
    axes[0].plot(test_loss_history, label="Test Loss", marker='o')
    axes[0].set_title("Loss History")
    axes[0].set_xlabel("Evaluation Step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy plot
    axes[1].plot(train_acc_history, label="Train Acc", marker='o')
    axes[1].plot(valid_acc_history, label="Valid Acc", marker='o')
    axes[1].plot(test_acc_history, label="Test Acc", marker='o')
    axes[1].set_title("Accuracy History")
    axes[1].set_xlabel("Evaluation Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


############################################################################################################################
### EMBEDDINGS & UMAP
############################################################################################################################


@torch.no_grad()
def extract_embeddings(net, dl, device, which=None, cls_index=0, return_numpy=True):

    """
    Extract embeddings from a trained EsmDeepSec model.

    Args:
        net: EsmDeepSec instance (already loaded with weights).
        dl: yields (input_ids, attention_mask, label) or (input_ids, attention_mask).
        device: torch.device
        which: list of strings specifying embeddings to return. See doc above for allowed keys.
        cls_index: token index to use as CLS if pooler_output not available (default 0).
        return_numpy: if True, returned tensors are numpy arrays on CPU; otherwise torch tensors on CPU.

    Returns:
        dict:
            'embeddings': dict mapping requested key -> stacked embeddings (N x D)
            'names': list of protein names
            'labels': np.array of true labels
            'preds': np.array of predicted labels (or None if not available)
    """

    if which is None:
        which = [
            "esm_mean", "esm_max", "esm_cls",
            "feature_mean", "feature_max", "feature_cls", "feature_concat"
        ]

    net.eval()
    net = net.to(device)

    # storage: lists where we append CPU tensors
    buffers = {k: [] for k in which}
    names_list = []
    labels_list = []
    preds_list = []

    for batch in tqdm(dl, desc=f"Batch", unit=" train batch", leave=False):
    
        # unpack
        seq, attention_mask, label, names = batch

        #move to device
        seq, attention_mask, label = seq.to(device), attention_mask.to(device), label.to(device)      

        # Forward apss
        logits = net(seq, attention_mask=attention_mask) 
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        # add info to lists
        names_list.extend(names)
        labels_list.extend(label.detach().cpu().numpy())
        preds_list.extend(preds.detach().cpu().numpy())

        # final contectualised embs
        esm_last_hidden_state = net.esm_last_hidden_state  # [B, L, H]

        # --- ESM-level embeddings ---
        if any(k.startswith("esm_") for k in which):
            if "esm_mean" in which:
                buffers["esm_mean"].append(esm_last_hidden_state.mean(dim=1).detach().cpu())
            if "esm_max" in which:
                esm_max, _ = esm_last_hidden_state.max(dim=1)
                buffers["esm_max"].append(esm_max.detach().cpu())
            if "esm_cls" in which:
                buffers["esm_cls"].append(esm_last_hidden_state[:, cls_index, :].detach().cpu())
            if "esm_tokens" in which:
                buffers["esm_tokens"].append(esm_last_hidden_state.detach().cpu())  # [B, L, H]

        # --- class_head embeddings ---
        if any(k.startswith("feature_") for k in which):
            if "feature_mean" in which:
                buffers["feature_mean"].append(net.class_head.avg_pool.detach().cpu())
            if "feature_max" in which:
                buffers["feature_max"].append(net.class_head.max_pool.detach().cpu())
            if "feature_cls" in which:
                buffers["feature_cls"].append(net.class_head.cls_repr.detach().cpu())
            if "feature_concat" in which:
                buffers["feature_concat"].append(net.class_head.concat_repr.detach().cpu())

    results = {}
    for k, list_of_tensors in buffers.items():
        if len(list_of_tensors) == 0:
            results[k] = None
            continue
        stacked = torch.cat(list_of_tensors, dim=0)
        results[k] = stacked.numpy() if return_numpy else stacked

    return results, names_list, np.array(labels_list), np.array(preds_list)


def compute_umap_tensors(embeddings_dict, n_neighbors=15, min_dist=0.1, random_state=42):
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
        print(f"Computing UMAP for {key} with shape {emb.shape if emb is not None else None}...")
        if emb is None or emb.shape[1] < 2:
            continue  # skip embeddings with fewer than 2 dimensions

        emb_2d = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state).fit_transform(emb)
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
        emb = umap_embs[key]
        if emb is None:
            continue
        if hasattr(emb, "numpy"):
            emb_np = emb.numpy()
        else:
            emb_np = emb
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
        axes[0, col_idx].set_title(f"{key.capitalize()} - True Labels")
        axes[0, col_idx].grid(True, alpha=0.3)

        # Row 1: Predicted labels
        sns.scatterplot(x='UMAP1', y='UMAP2', hue='PredClass', palette=class_palette,
                        data=df, alpha=0.8, s=50, ax=axes[1, col_idx], legend=False)
        axes[1, col_idx].set_title(f"{key.capitalize()} - Predicted Labels")
        axes[1, col_idx].grid(True, alpha=0.3)

        # Row 2: Correct vs Wrong
        df_correct = df[df['CorrectStr'] == 'correct']
        df_wrong = df[df['CorrectStr'] == 'wrong']

        axes[2, col_idx].scatter(df_correct['UMAP1'], df_correct['UMAP2'],
                                 c=corr_palette['correct'], alpha=0.6, s=40, label='correct', edgecolor='none')
        axes[2, col_idx].scatter(df_wrong['UMAP1'], df_wrong['UMAP2'],
                                 c=corr_palette['wrong'], alpha=0.9, s=90, label='wrong', edgecolor='k', linewidth=0.4)
        axes[2, col_idx].set_title(f"{key.capitalize()} - Correct vs Wrong")
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

    return df

############################################################################################################################
### A-SCANNING
############################################################################################################################


def alanine_scanning(model, tokenizer, single_protein_info, window_size: int = 3, device="cuda", SUBSTITUTE_AA="A", normalise_true_substitution=False):
    """
    Perform scanning by replacing a window of residues with alanine.
    Window_size should be odd (so there is a center residue).
    Returns baseline probability and delta probabilities mapped to positions.

    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        single_protein_info: Series containing single protein data
        window_size: Size of mutation window (must be odd)
        device: Device to run inference on
        SUBSTITUTE_AA: Amino acid to substitute (default: "A" for alanine)
    
    Returns:
        dict: Contains baseline_prob, delta_probs, and mutated_probs
    """
    assert window_size % 2 == 1, "window_size must be odd"


    ### CLAUCLATE BASELINE PROB ###

    # Create a temporary dataframe with the single protein
    temp_df = pd.DataFrame([single_protein_info])

    # Create dataloader for single protein
    single_protein_dl = my_dataset.create_dataloader(
        temp_df, 
        set_name=single_protein_info['set'], 
        batch_size=1, 
        shuffle=False
    )

    # calculate baseline peob
    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore", category=UserWarning)
        baseline_dict = evaluate_model(model, single_protein_dl, device, split_name="Single protein", verbose=False)
    baseline_p = baseline_dict["probs_class1"][0].item() 


    ### GENERATE ALL MUTATED PROBS ###
    
    # Sequence to mutate (use TRUNCATED sequence)
    seq_list = list(single_protein_info['trunc_sequence'])
    trunc_len = len(seq_list)

    mutated_sequences = []
    true_substitution_count = [] # how many true substitutions (not substituting A->A)

    # Iterate through each position to generate the mutated sequence for that position
    for i in tqdm(range(trunc_len), desc="Generating mutations"):

        # Define window boundaries
        half_w = window_size // 2
        start = max(0, i - half_w)
        end = min(trunc_len, i + half_w + 1)

        true_sub_count = 0
        
        # Create mutated sequence
        mutated_seq_list = seq_list.copy()
        for j in range(start, end):
            mutated_seq_list[j] = SUBSTITUTE_AA
            if seq_list[j] != SUBSTITUTE_AA: # cpunt true substitutions
                true_sub_count += 1
        mutated_sequences.append(''.join(mutated_seq_list)) # Conver to str
        true_substitution_count.append(true_sub_count)


    ### PREPROCESS ALL MUTATED PROTS ###

    # Idea: creata a dataloder to re-use the evaluate_model() fucntion

    # Create a list of processed data dictionaries for all mutations
    all_mutated_data = []
    
    # Preprocess all sequences (this still uses a loop but it's lightweight)
    for mutated_seq in tqdm(mutated_sequences, desc="Preprocessing mutations"):

        mutated_data = my_dataset.preprocess_sequence(
            sequence=mutated_seq,
            label=single_protein_info['label'], # Use the original label for all
            protein_name=single_protein_info['protein'],
            tokenizer=tokenizer,
            protein_max_length=single_protein_info['inputs_ids_length'] # Use the original max length
        )
        mutated_data['set'] = single_protein_info['set']
        all_mutated_data.append(mutated_data)

    # Create one DataFrame from all processed data
    mutated_df = pd.DataFrame(all_mutated_data)

    # Create one DataLoader for ALL mutated sequences
    # Batch size is set to the total number of mutations (1 batch total)
    mutated_dl = my_dataset.create_dataloader(
        mutated_df,
        set_name=single_protein_info['set'], #  all samples have this set name
        batch_size=int(trunc_len // 100),         
        shuffle=False
    )

    ### CALCULATE PROBS ALL MUTATED PROTS ###

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # Assuming 'evaluate_model' is accessible
        print("Evaluating model on mutated sequences...")
        mutated_dict = evaluate_model(model, mutated_dl, device, split_name="Batch Mutation Scan", verbose=False)
    
    # The result is a tensor of probabilities [trunc_len]
    mutated_probs = mutated_dict["probs_class1"].cpu().numpy()


    ### CALCULATE DELTAS ###
    delta_p = mutated_probs - baseline_p

    if normalise_true_substitution:
        true_substitution_count_np = np.array(true_substitution_count)
        delta_p = np.where(true_substitution_count_np > 0, delta_p / true_substitution_count_np, 0.0)

    return {
        'baseline_prob': baseline_p,
        'delta_probs': delta_p,
        'mutated_probs': mutated_probs,
        'sequence': single_protein_info['trunc_sequence'],
        'protein_name': single_protein_info['protein']
    }

def plot_alanine_scan(delta_p, sequence, sigma=3, threshold=True, figsize=(20, 6), 
                      highlight_residues=True, top_n=10, show_sequence=True,
                      style='darkgrid', palette='RdBu_r', protein_name="N/A"):
    """
    Plot the Δp values across the protein sequence with optional smoothing,
    threshold lines, and residue highlighting using Seaborn styling.

    Args:
        delta_p: array of Δp values (signed deltas from baseline).
        sequence: protein sequence string.
        sigma: Gaussian smoothing width (residues) for smoothing curve.
        threshold: whether to plot threshold lines (mean ± 2*std).
        figsize: tuple for figure size.
        highlight_residues: whether to annotate top important residues.
        top_n: number of top residues to highlight (both positive and negative).
        show_sequence: whether to show amino acid letters on x-axis (only for short sequences).
        style: seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks').
        palette: color palette for gradient coloring.
        protein_name: Name/ID of the protein (string, added for robustness).
    """
    # Set seaborn style
    sns.set_style(style)
    sns.set_context("notebook", font_scale=1.1)
    
    positions = np.arange(1, len(sequence) + 1)
    delta_p = np.array(delta_p)
    
    # Smoothed signal
    smooth_delta = gaussian_filter1d(delta_p, sigma=sigma)
    
    # Statistics
    mu = np.mean(delta_p)
    std = np.std(delta_p)
    cutoff = 2 * std
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create color map based on delta_p values
    max_abs_delta = np.max(np.abs(delta_p))     # Calculate the maximum absolute deviation from zero for a centered color map
    norm = plt.Normalize(vmin=-max_abs_delta, vmax=max_abs_delta)
    colors = plt.cm.RdBu_r(norm(delta_p))
    
    # Plot bars with gradient coloring
    bars = ax.bar(positions, delta_p, color=colors, alpha=0.6, 
                  edgecolor='black', linewidth=0.5, label="Δp per residue")
    
    # Plot smoothed curve
    ax.plot(positions, smooth_delta, color="black", linewidth=3, 
            label=f"Smoothed (σ={sigma})", zorder=5, alpha=0.8)
    
    # Add threshold lines
    if threshold:
        ax.axhline(y=mu + cutoff, color="#2ecc71", linestyle="--", linewidth=2, 
                   label=f"+2σ = {mu + cutoff:.3f}", alpha=0.8)
        ax.axhline(y=mu - cutoff, color="#e74c3c", linestyle="--", linewidth=2, 
                   label=f"-2σ = {mu - cutoff:.3f}", alpha=0.8)
        ax.axhline(y=mu, color="#95a5a6", linestyle=":", linewidth=1.5, 
                   label=f"Mean = {mu:.3f}", alpha=0.7)
    
    # Highlight important residues
    if highlight_residues and top_n > 0:
        # Most negative deltas (most important for positive class)
        neg_indices = np.argsort(delta_p)[:top_n]
        for idx in neg_indices:
            if delta_p[idx] < (mu - cutoff):
                ax.annotate(f'{sequence[idx]}{idx+1}', 
                           xy=(positions[idx], delta_p[idx]),
                           xytext=(0, -20), textcoords='offset points',
                           ha='center', fontsize=9, color='white', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', 
                                    facecolor='#e74c3c', 
                                    edgecolor='darkred',
                                    alpha=0.9, linewidth=2),
                           arrowprops=dict(arrowstyle='->', 
                                         color='darkred', 
                                         lw=1.5,
                                         connectionstyle='arc3,rad=0'))
        
        # Most positive deltas (stabilizing residues)
        pos_indices = np.argsort(delta_p)[-top_n:]
        for idx in pos_indices:
            if delta_p[idx] > (mu + cutoff):
                ax.annotate(f'{sequence[idx]}{idx+1}', 
                           xy=(positions[idx], delta_p[idx]),
                           xytext=(0, 20), textcoords='offset points',
                           ha='center', fontsize=9, color='white', weight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', 
                                    facecolor='#3498db', 
                                    edgecolor='darkblue',
                                    alpha=0.9, linewidth=2),
                           arrowprops=dict(arrowstyle='->', 
                                         color='darkblue', 
                                         lw=1.5,
                                         connectionstyle='arc3,rad=0'))
    
    # Labels and styling
    ax.set_xlabel("Residue Position", fontsize=13, fontweight='bold')
    ax.set_ylabel("Δp (Change in Probability)", fontsize=13, fontweight='bold')
    ax.set_title("Alanine Scanning Importance Map\n" + 
                 "Negative Δp = Critical for function", 
                 fontsize=15, fontweight='bold', pad=20)
    
    # Show sequence on x-axis if short enough
    if show_sequence and len(sequence) <= 50:
        ax.set_xticks(positions)
        ax.set_xticklabels(list(sequence), fontsize=9, family='monospace')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    else:
        ax.set_xlim(0, len(sequence) + 1)
    
    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95, 
             edgecolor='black', fancybox=True, shadow=True)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, aspect=30)
    cbar.set_label('Δp Value', fontsize=11, fontweight='bold')
    
    # Add statistics box with seaborn styling
    stats_text = (f'Statistics:\n'
                 f'Mean: {mu:.4f}\n'
                 f'Std: {std:.4f}\n'
                 f'Min: {np.min(delta_p):.4f}\n'
                 f'Max: {np.max(delta_p):.4f}\n'
                 f'Range: {np.max(delta_p) - np.min(delta_p):.4f}')
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, 
                edgecolor='black', linewidth=1.5)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', bbox=props,
            family='monospace')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed summary
    print(f"\n{'='*70}")
    print(f"{'ALANINE SCANNING SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"\n{'Sequence Information:':<30}")
    print(f"  {'Length:':<25} {len(sequence)}")
    print(f"  {'Protein:':<25} {protein_name}")
    print(f"\n{'Statistical Summary:':<30}")
    print(f"  {'Mean Δp:':<25} {mu:.4f}")
    print(f"  {'Std Δp:':<25} {std:.4f}")
    print(f"  {'Min Δp:':<25} {np.min(delta_p):.4f}")
    print(f"  {'Max Δp:':<25} {np.max(delta_p):.4f}")
    print(f"  {'Threshold (+2σ):':<25} {mu + cutoff:.4f}")
    print(f"  {'Threshold (-2σ):':<25} {mu - cutoff:.4f}")
    
    print(f"\n{'─'*70}")
    print(f"{'TOP CRITICAL RESIDUES (Largest Negative Δp)':^70}")
    print(f"{'─'*70}")
    print(f"{'Rank':<8}{'Position':<12}{'Residue':<12}{'Δp':<15}{'Status':<20}")
    print(f"{'─'*70}")
    
    critical_indices = np.argsort(delta_p)[:top_n]
    for rank, idx in enumerate(critical_indices, 1):
        status = "⚠️  Beyond threshold" if delta_p[idx] < (mu - cutoff) else "Within range"
        print(f"{rank:<8}{idx+1:<12}{sequence[idx]:<12}{delta_p[idx]:<15.4f}{status:<20}")
    
    print(f"\n{'─'*70}")
    print(f"{'TOP STABILIZING RESIDUES (Largest Positive Δp)':^70}")
    print(f"{'─'*70}")
    print(f"{'Rank':<8}{'Position':<12}{'Residue':<12}{'Δp':<15}{'Status':<20}")
    print(f"{'─'*70}")
    
    stabilizing_indices = np.argsort(delta_p)[-top_n:][::-1]
    for rank, idx in enumerate(stabilizing_indices, 1):
        status = "✓ Beyond threshold" if delta_p[idx] > (mu + cutoff) else "Within range"
        print(f"{rank:<8}{idx+1:<12}{sequence[idx]:<12}{delta_p[idx]:<15.4f}{status:<20}")
    
    print(f"{'='*70}\n")





