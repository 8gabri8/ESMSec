
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
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef, classification_report, confusion_matrix
from scipy.ndimage import gaussian_filter1d
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


def evaluate_model(net, dl, device, loss_fn=None, split_name="Eval", verbose=True, from_precomputed_embs=False):
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

            seq, attention_mask, label, _, emb= batch

            seq, attention_mask, label, emb = seq.to(device), attention_mask.to(device), label.to(device), emb.to(device)

            if from_precomputed_embs:
                output = net(precomputed_embs=emb)
            else:
                output = net(seq, attention_mask=attention_mask) # seq --> [batch_size, 2], TOKENISEd protein sequences

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


def train(net, train_dl, valid_dl, test_dl, loss_fn, config, from_precomputed_embs=False):

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
    #loss_fn = nn.CrossEntropyLoss() # binary classfication, needs logits

    # Set model to training mode
    net.train()


    for epoch_idx in trange(1, num_epochs + 1, desc="Training", unit="epoch"):

        net.train() # back to train mode

        pbar = tqdm(train_dl, desc=f"Epoch {epoch_idx}", unit="train batch", leave=False)

        for batch in pbar:
            
            # unpack
            seq, attention_mask, label, _, emb = batch

            # set model to trainign mode
            net.train()

            #move to device
            seq, attention_mask, label, emb = seq.to(device), attention_mask.to(device), label.to(device), emb.to(device)
            
            # Forward apss
            if from_precomputed_embs:
                output = net(precomputed_embs=emb)
            else:
                output = net(seq, attention_mask=attention_mask) # seq --> [batch_size, 2], TOKENISEd protein sequences

            # compute loss
            loss = loss_fn(output, label) # crossEntripyLoss() expects RAW logits, so it is correct passing direclty "outpu"
            # clear previous gradients
            optimizer.zero_grad(set_to_none=True)  
            # compute gradients (backpropgation)
            loss.backward()
            # paramers update
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")



        # EVALUATION EVERY TOT EPOCHS
        # NB if num_iters%eval_size != 0 the last iters will not be evaluated
        if epoch_idx % eval_epoch_freq == 0:

            print(f"--- Evaluation at iteration {epoch_idx} ---")

            # Return metrics of current evaluation
            train_metrics = evaluate_model(net, train_dl, device, loss_fn, split_name="Train", from_precomputed_embs=from_precomputed_embs)
            valid_metrics = evaluate_model(net, valid_dl, device, loss_fn, split_name="Validation", from_precomputed_embs=from_precomputed_embs)
            test_metrics  = evaluate_model(net, test_dl, device, loss_fn, split_name="Test", from_precomputed_embs=from_precomputed_embs)

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
        exp_lr.step() # increments the internal counter
        print(f"Epoch {epoch_idx}, New LR: {exp_lr.get_last_lr()[0]}")

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
    Comprehensive summary of training with multiple metrics, visualizations, and confusion matrices.
    
    Args:
        *_loss_history: List of loss values at each evaluation step
        *_acc_history: List of accuracy values at each evaluation step
        *_last_eval: Dictionary containing final evaluation metrics including:
            - loss, accuracy, balanced_accuracy, f1, mcc
            - probs, labels, pred_labels (tensors)
    """
    
    # ============================================================================
    # SECTION 1: Comprehensive Metrics Table
    # ============================================================================
    print("\n" + "="*80)
    print(" "*25 + "FINAL EVALUATION METRICS")
    print("="*80)
    
    metrics_table = [
        ["Metric", "Train", "Validation", "Test"],
        ["-"*15, "-"*15, "-"*15, "-"*15],
        ["Loss", 
         f"{train_last_eval['loss']:.4f}",
         f"{valid_last_eval['loss']:.4f}",
         f"{test_last_eval['loss']:.4f}"],
        ["Accuracy",
         f"{train_last_eval['accuracy']*100:.2f}%",
         f"{valid_last_eval['accuracy']*100:.2f}%",
         f"{test_last_eval['accuracy']*100:.2f}%"],
        ["Balanced Acc",
         f"{train_last_eval['balanced_accuracy']*100:.2f}%",
         f"{valid_last_eval['balanced_accuracy']*100:.2f}%",
         f"{test_last_eval['balanced_accuracy']*100:.2f}%"],
        ["F1 Score",
         f"{train_last_eval['f1']:.4f}",
         f"{valid_last_eval['f1']:.4f}",
         f"{test_last_eval['f1']:.4f}"],
        ["MCC",
         f"{train_last_eval['mcc']:.4f}",
         f"{valid_last_eval['mcc']:.4f}",
         f"{test_last_eval['mcc']:.4f}"],
    ]
    
    for row in metrics_table:
        print(f"{row[0]:<15} {row[1]:>15} {row[2]:>15} {row[3]:>15}")
    
    print("="*80)
    
    # ============================================================================
    # SECTION 2: Performance Analysis
    # ============================================================================
    print("\n" + "="*80)
    print(" "*28 + "PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Overfitting check
    train_val_gap = train_last_eval['accuracy'] - valid_last_eval['accuracy']
    if abs(train_val_gap) > 0.10:
        status = "⚠️  SIGNIFICANT OVERFITTING DETECTED" if train_val_gap > 0 else "⚠️  UNUSUAL PATTERN"
        print(f"\n{status}")
        print(f"   Train-Valid gap: {train_val_gap*100:+.2f}%")
    else:
        print(f"\n✓ Good generalization (Train-Valid gap: {train_val_gap*100:+.2f}%)")
    
    # Test performance
    test_val_gap = test_last_eval['accuracy'] - valid_last_eval['accuracy']
    print(f"✓ Test-Valid gap: {test_val_gap*100:+.2f}%")
    
    # Class balance check
    train_labels = train_last_eval['labels'].numpy()
    class_dist = np.bincount(train_labels)
    imbalance_ratio = class_dist.max() / class_dist.min()
    print(f"\n✓ Class distribution (Test): {dict(enumerate(class_dist))}")
    if imbalance_ratio > 1.5:
        print(f"⚠️  Class imbalance detected (ratio: {imbalance_ratio:.2f}:1)")
        print(f"   → Balanced accuracy is more reliable than accuracy")
    
    print("="*80)
    
    # ============================================================================
    # SECTION 3: Visualizations
    # ============================================================================
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Loss and Accuracy curves
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(train_loss_history, label="Train", marker='o', linewidth=2, markersize=6)
    ax1.plot(valid_loss_history, label="Valid", marker='s', linewidth=2, markersize=6)
    ax1.plot(test_loss_history, label="Test", marker='^', linewidth=2, markersize=6)
    ax1.set_title("Loss History", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Evaluation Step", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(train_acc_history, label="Train", marker='o', linewidth=2, markersize=6)
    ax2.plot(valid_acc_history, label="Valid", marker='s', linewidth=2, markersize=6)
    ax2.plot(test_acc_history, label="Test", marker='^', linewidth=2, markersize=6)
    ax2.set_title("Accuracy History", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Evaluation Step", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # Row 2: Confusion Matrices
    datasets = [
        (train_last_eval, "Train", gs[1, 0]),
        (valid_last_eval, "Valid", gs[1, 1]),
        (test_last_eval, "Test", gs[1, 2])
    ]
    
    for eval_dict, name, grid_pos in datasets:
        ax = fig.add_subplot(grid_pos)
        cm = confusion_matrix(
            eval_dict['labels'].numpy(),
            eval_dict['pred_labels'].numpy()
        )
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    cbar=False, square=True, annot_kws={"size": 12})
        ax.set_title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
    
    # Row 3: Metrics comparison bar chart
    ax7 = fig.add_subplot(gs[2, :])
    metrics_names = ['Accuracy', 'Balanced Accuracy', 'F1 Score', 'MCC']
    train_metrics = [
        train_last_eval['accuracy'],
        train_last_eval['balanced_accuracy'],
        train_last_eval['f1'],
        (train_last_eval['mcc'] + 1) / 2  # Normalize MCC to [0,1] for visualization
    ]
    valid_metrics = [
        valid_last_eval['accuracy'],
        valid_last_eval['balanced_accuracy'],
        valid_last_eval['f1'],
        (valid_last_eval['mcc'] + 1) / 2
    ]
    test_metrics = [
        test_last_eval['accuracy'],
        test_last_eval['balanced_accuracy'],
        test_last_eval['f1'],
        (test_last_eval['mcc'] + 1) / 2
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    ax7.bar(x - width, train_metrics, width, label='Train', alpha=0.8)
    ax7.bar(x, valid_metrics, width, label='Valid', alpha=0.8)
    ax7.bar(x + width, test_metrics, width, label='Test', alpha=0.8)
    
    ax7.set_xlabel('Metrics', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax7.set_title('Metrics Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(metrics_names)
    ax7.legend(fontsize=10)
    ax7.grid(axis='y', alpha=0.3)
    ax7.set_ylim([0, 1.1])
    
    # Add note about MCC normalization
    ax7.text(0.98, 0.02, 'Note: MCC normalized to [0,1] for visualization',
             transform=ax7.transAxes, fontsize=8, style='italic',
             horizontalalignment='right', verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Training Summary Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.show()
    
    # ============================================================================
    # SECTION 4: Detailed Classification Report (Test Set)
    # ============================================================================
    print("\n" + "="*80)
    print(" "*25 + "TEST SET CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(
        test_last_eval['labels'].numpy(),
        test_last_eval['pred_labels'].numpy(),
        target_names=['Class 0', 'Class 1'],
        digits=4
    ))
    print("="*80 + "\n")

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


def multi_aa_scanning(model, tokenizer, 
                      single_protein_info, 
                      window_size: int = 3, 
                      substitute_aas=["A", "R", "E", "F"],
                      normalise_true_substitution=False,
                      device="cuda"):
    """
    Perform scanning by replacing a window of residues with multiple amino acids.
    For each position, substitutes with each amino acid in substitute_aas and averages the results.
    Window_size should be odd (so there is a center residue).
    Returns baseline probability and delta probabilities mapped to positions.

    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        single_protein_info: Series containing single protein data
        window_size: Size of mutation window (must be odd)
        device: Device to run inference on
        substitute_aas: List of amino acids to substitute (default: ["A", "R", "E", "F"])
        normalise_true_substitution: Whether to normalize by number of true substitutions
    
    Returns:
        dict: Contains baseline_prob, delta_probs, mutated_probs, and per_aa_results
    """
    assert window_size % 2 == 1, "window_size must be odd"


    ### CALCULATE BASELINE PROB ###

    # Create a temporary dataframe with the single protein
    temp_df = pd.DataFrame([single_protein_info])

    # Create dataloader for single protein
    single_protein_dl = my_dataset.create_dataloader(
        temp_df, 
        set_name=single_protein_info['set'], 
        batch_size=1, 
        shuffle=False
    )

    # Calculate baseline prob
    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore", category=UserWarning)
        baseline_dict = evaluate_model(model, single_protein_dl, device, split_name="Single protein", verbose=False)
    baseline_p = baseline_dict["probs_class1"][0].item() 


    ### GENERATE ALL MUTATED SEQUENCES FOR ALL AMINO ACIDS ###
    
    # Sequence to mutate (use TRUNCATED sequence)
    seq_list = list(single_protein_info['trunc_sequence'])
    trunc_len = len(seq_list)

    mutated_sequences = []
    mutation_info = []  # Store info about each mutation (position, AA used)

    # Iterate through each position and each substitute amino acid
    for i in tqdm(range(trunc_len), desc="Generating mutations"):

        # Define window boundaries
        half_w = window_size // 2
        start = max(0, i - half_w)
        end = min(trunc_len, i + half_w + 1)

        # For each position, create mutations with each substitute amino acid
        for sub_aa in substitute_aas:
            true_sub_count = 0
            
            # Create mutated sequence
            mutated_seq_list = seq_list.copy()
            for j in range(start, end):
                if seq_list[j] != sub_aa:  # Count true substitutions
                    true_sub_count += 1
                mutated_seq_list[j] = sub_aa
            
            mutated_sequences.append(''.join(mutated_seq_list))
            mutation_info.append({
                'position': i,
                'substitute_aa': sub_aa,
                'true_sub_count': true_sub_count
            })


    ### PREPROCESS ALL MUTATED PROTEINS ###

    # Create a list of processed data dictionaries for all mutations
    all_mutated_data = []
    
    # Preprocess all sequences
    for mutated_seq in tqdm(mutated_sequences, desc="Preprocessing mutations"):

        mutated_data = my_dataset.preprocess_sequence(
            sequence=mutated_seq,
            label=single_protein_info['label'],
            protein_name=single_protein_info['protein'],
            tokenizer=tokenizer,
            protein_max_length=single_protein_info['inputs_ids_length']
        )
        mutated_data['set'] = single_protein_info['set']
        all_mutated_data.append(mutated_data)

    # Create one DataFrame from all processed data
    mutated_df = pd.DataFrame(all_mutated_data)

    # Create one DataLoader for ALL mutated sequences
    # Calculate appropriate batch size
    total_mutations = len(mutated_sequences)
    batch_size = max(1, int(total_mutations // 100))
    
    mutated_dl = my_dataset.create_dataloader(
        mutated_df,
        set_name=single_protein_info['set'],
        batch_size=batch_size,         
        shuffle=False
    )

    ### CALCULATE PROBS FOR ALL MUTATED PROTEINS ###

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        print("Evaluating model on mutated sequences...")
        mutated_dict = evaluate_model(model, mutated_dl, device, split_name="Multi-AA Scan", verbose=False)
    
    # The result is a tensor of probabilities
    mutated_probs = mutated_dict["probs_class1"].cpu().numpy()


    ### CALCULATE DELTAS AND AVERAGE ACROSS AMINO ACIDS ###
    
    delta_p_all = mutated_probs - baseline_p
    
    # Organize results by position and amino acid
    n_positions = trunc_len
    n_aas = len(substitute_aas)
    
    # Reshape to [positions, amino_acids]
    delta_p_reshaped = delta_p_all.reshape(n_positions, n_aas)
    mutated_probs_reshaped = mutated_probs.reshape(n_positions, n_aas)
    
    # Store per-AA results for detailed analysis
    per_aa_results = {}
    for aa_idx, aa in enumerate(substitute_aas):
        per_aa_results[aa] = {
            'delta_probs': delta_p_reshaped[:, aa_idx],
            'mutated_probs': mutated_probs_reshaped[:, aa_idx]
        }
    
    # Calculate mean across amino acids for each position
    delta_p_mean = np.mean(delta_p_reshaped, axis=1)
    mutated_probs_mean = np.mean(mutated_probs_reshaped, axis=1)
    
    # Optional normalization by true substitution count
    if normalise_true_substitution:
        # Calculate mean true substitution count per position
        true_sub_counts = np.array([info['true_sub_count'] for info in mutation_info])
        true_sub_counts_reshaped = true_sub_counts.reshape(n_positions, n_aas)
        mean_true_sub_counts = np.mean(true_sub_counts_reshaped, axis=1)
        
        delta_p_mean = np.where(mean_true_sub_counts > 0, 
                                delta_p_mean / mean_true_sub_counts, 
                                0.0)

    return {
        'baseline_prob': baseline_p,
        'delta_probs': delta_p_mean,  # Mean across all substitute AAs
        'mutated_probs': mutated_probs_mean,  # Mean across all substitute AAs
        'per_aa_results': per_aa_results,  # Individual results per amino acid
        'sequence': single_protein_info['trunc_sequence'],
        'protein_name': single_protein_info['protein'],
        'substitute_aas': substitute_aas
    }


def plot_multi_aa_scan(scan_results, sigma=3, threshold=True, figsize=(20, 6), 
                       highlight_residues=True, top_n=10, show_sequence=True,
                       style='darkgrid', palette='RdBu_r', show_per_aa=False):
    """
    Plot the Δp values across the protein sequence from multi-AA scanning results
    with optional smoothing, threshold lines, and residue highlighting.

    Args:
        scan_results: Dictionary output from multi_aa_scanning() function containing:
                     - 'delta_probs': Mean delta probabilities
                     - 'sequence': Protein sequence
                     - 'protein_name': Protein identifier
                     - 'per_aa_results': Individual results per amino acid
                     - 'substitute_aas': List of amino acids used
        sigma: Gaussian smoothing width (residues) for smoothing curve.
        threshold: whether to plot threshold lines (mean ± 2*std).
        figsize: tuple for figure size.
        highlight_residues: whether to annotate top important residues.
        top_n: number of top residues to highlight (both positive and negative).
        show_sequence: whether to show amino acid letters on x-axis (only for short sequences).
        style: seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks').
        palette: color palette for gradient coloring.
        show_per_aa: whether to show individual amino acid traces.
    """
    # Extract data from scan_results
    delta_p = scan_results['delta_probs']
    sequence = scan_results['sequence']
    protein_name = scan_results['protein_name']
    per_aa_results = scan_results.get('per_aa_results', {})
    substitute_aas = scan_results.get('substitute_aas', [])
    baseline_prob = scan_results['baseline_prob']
    
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
    max_abs_delta = np.max(np.abs(delta_p))
    norm = plt.Normalize(vmin=-max_abs_delta, vmax=max_abs_delta)
    colors = plt.cm.RdBu_r(norm(delta_p))
    
    # Plot bars with gradient coloring
    bars = ax.bar(positions, delta_p, color=colors, alpha=0.6, 
                  edgecolor='black', linewidth=0.5, 
                  label=f"Mean Δp (across {', '.join(substitute_aas)})")
    
    # Plot individual amino acid traces if requested
    if show_per_aa and per_aa_results:
        aa_colors = {'A': '#ff7f0e', 'R': '#2ca02c', 'E': '#d62728', 'F': '#9467bd'}
        for aa in substitute_aas:
            if aa in per_aa_results:
                aa_delta = per_aa_results[aa]['delta_probs']
                ax.plot(positions, aa_delta, 
                       color=aa_colors.get(aa, 'gray'), 
                       linewidth=1.5, alpha=0.4, 
                       label=f"Δp ({aa})", linestyle='--')
    
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
    ax.set_title(f"Multi-AA Scanning Importance Map - {protein_name}\n" + 
                 f"Negative Δp = Critical for function | Baseline prob: {baseline_prob:.4f}", 
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
    
    # Add statistics box
    stats_text = (f'Statistics:\n'
                 f'Mean: {mu:.4f}\n'
                 f'Std: {std:.4f}\n'
                 f'Min: {np.min(delta_p):.4f}\n'
                 f'Max: {np.max(delta_p):.4f}\n'
                 f'Range: {np.max(delta_p) - np.min(delta_p):.4f}\n'
                 f'Baseline: {baseline_prob:.4f}')
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, 
                edgecolor='black', linewidth=1.5)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', bbox=props,
            family='monospace')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed summary
    print(f"\n{'='*70}")
    print(f"{'MULTI-AA SCANNING SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"\n{'Sequence Information:':<30}")
    print(f"  {'Length:':<25} {len(sequence)}")
    print(f"  {'Protein:':<25} {protein_name}")
    print(f"  {'Substitute AAs:':<25} {', '.join(substitute_aas)}")
    print(f"  {'Baseline Probability:':<25} {baseline_prob:.4f}")
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
    
    # Show per-AA breakdown for top critical residues if available
    if per_aa_results:
        print(f"\n{'Per-AA breakdown for top critical residues:':<30}")
        for rank, idx in enumerate(critical_indices[:5], 1):  # Top 5
            print(f"\n  Position {idx+1} ({sequence[idx]}):")
            for aa in substitute_aas:
                if aa in per_aa_results:
                    aa_delta = per_aa_results[aa]['delta_probs'][idx]
                    print(f"    {aa}: {aa_delta:>8.4f}")
    
    print(f"\n{'─'*70}")
    print(f"{'TOP STABILIZING RESIDUES (Largest Positive Δp)':^70}")
    print(f"{'─'*70}")
    print(f"{'Rank':<8}{'Position':<12}{'Residue':<12}{'Δp':<15}{'Status':<20}")
    print(f"{'─'*70}")
    
    stabilizing_indices = np.argsort(delta_p)[-top_n:][::-1]
    for rank, idx in enumerate(stabilizing_indices, 1):
        status = "✓ Beyond threshold" if delta_p[idx] > (mu + cutoff) else "Within range"
        print(f"{rank:<8}{idx+1:<12}{sequence[idx]:<12}{delta_p[idx]:<15.4f}{status:<20}")
    
    # Show per-AA breakdown for top stabilizing residues if available
    if per_aa_results:
        print(f"\n{'Per-AA breakdown for top stabilizing residues:':<30}")
        for rank, idx in enumerate(stabilizing_indices[:5], 1):  # Top 5
            print(f"\n  Position {idx+1} ({sequence[idx]}):")
            for aa in substitute_aas:
                if aa in per_aa_results:
                    aa_delta = per_aa_results[aa]['delta_probs'][idx]
                    print(f"    {aa}: {aa_delta:>8.4f}")
    
    print(f"{'='*70}\n")

