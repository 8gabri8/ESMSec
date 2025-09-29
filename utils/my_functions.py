
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



def truncate_sequence(sequence, max_length=1000):
    # If sequence > 1000 residues: take first 500 + last 500 residues
    if len(sequence) <= max_length:
        return sequence
    else:
        half_length = (max_length) // 2 
        truncated_sequence = sequence[:half_length] + sequence[-half_length:]

    return truncated_sequence



def evaluate_model(net, dl, device, loss_fn, split_name="Eval"):
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

    with torch.no_grad():
        for seq, attention_mask, label, _ in dl:
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

    # Calculate additional metrics
    # F1 Score (binary or weighted for multiclass)
    f1 = f1_score(labels_np, preds_np, average='binary' if len(torch.unique(all_labels)) == 2 else 'weighted', zero_division=0)
    
    # Balanced Accuracy
    balanced_acc = balanced_accuracy_score(labels_np, preds_np)
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(labels_np, preds_np)

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
    optimizer = AdamW(net.feature_fn.parameters(), lr=lr) # ONLY train the classification head
    exp_lr = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) # step_size=N → the LR changes after step() has been called N times
    loss_fn = nn.CrossEntropyLoss() # binary classfication, needs logits

    # Set model to training mode
    net.train()

    # Initialize progress bar for total iterations
    pbar = trange(1, num_epochs + 1, desc="Training", unit="iter") #tqdm(range(...))
    epoch_idx = 1


    for epoch_idx in pbar:

        net.train() # back to train mode

        for [seq, attention_mask, label, _] in train_dl: # take one BATCH

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

    for seq, attention_mask, label, names in dl:

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

        # --- feature_fn embeddings ---
        if any(k.startswith("feature_") for k in which):
            if "feature_mean" in which:
                buffers["feature_mean"].append(net.feature_fn.avg_pool.detach().cpu())
            if "feature_max" in which:
                buffers["feature_max"].append(net.feature_fn.max_pool.detach().cpu())
            if "feature_cls" in which:
                buffers["feature_cls"].append(net.feature_fn.cls_repr.detach().cpu())
            if "feature_concat" in which:
                buffers["feature_concat"].append(net.feature_fn.concat_repr.detach().cpu())

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
















import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

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






