
from torch.optim import AdamW
from torch import nn
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange, tqdm
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef, classification_report, confusion_matrix, roc_auc_score
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from tqdm.notebook import tqdm, trange
import random
import yaml
import os
import requests

def set_all_seeds(seed_value=42):
    """
    Sets seeds for reproducibility across random, numpy, and PyTorch (if available).

    Args:
        seed_value (int): The integer seed to use. Default is 42.
    """
    # 1. Python's built-in 'random' module
    random.seed(seed_value)

    # 2. NumPy (often used for data manipulation and initialization)
    np.random.seed(seed_value)

    # 3. PyTorch (if using deep learning)
    if torch is not None:
        torch.manual_seed(seed_value)
        # For GPU-based operations:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value) # For multi-GPU setups
            
            # Additional configuration for deterministic GPU operations
            # NOTE: These settings can sometimes slow down execution.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    # 4. Environment Variables (optional, for some specific libraries/kernels)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    print(f"Seeds set successfully to {seed_value} for random, numpy, and PyTorch (if used).")

def load_config(file_path):
    """Loads a YAML configuration file into a Python dictionary."""
    try:
        with open(file_path, 'r') as file:
            # Use safe_load to safely parse the YAML file
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {file_path}")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None

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

            seq = batch.get("input_ids").to(device)
            attention_mask = batch.get("attention_mask").to(device)
            label = batch.get("label").to(device)
            emb = batch.get("embs").to(device)

            if from_precomputed_embs:
                output = net(precomputed_embs=emb, attention_mask=attention_mask)
            else:
                output = net(seq, attention_mask=attention_mask)

            probs = torch.softmax(output, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            loss = loss_fn(output, label) # already compute SINGLE BATCH AVERAGE

            total_loss += loss
            total_correct += (preds == label).sum()

            all_probs.append(probs.cpu())
            all_labels.append(label.cpu())
            all_preds.append(preds.cpu())

            # Attention mutiplcass
            if probs.shape[1] == 2:
                # Binary classification: keep class 1 probabilities
                all_probs_class1.append(probs.cpu()[:, 1])
            else:
                # Multiclass: you need to specify which class to track
                # Track a specific class (e.g., class 1)
                all_probs_class1.append(probs.cpu()[:, 1])

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
    # F1 Score
    #f1 = f1_score(labels_np, preds_np, average='binary' if len(torch.unique(all_labels)) == 2 else 'weighted', zero_division=0)
    f1 = f1_score(labels_np, preds_np, average='weighted', zero_division=0)  # takes the average of these per-class F1-scores, weighted by the number of true instances in each class
    # Balanced Accuracy
    balanced_acc = balanced_accuracy_score(labels_np, preds_np)
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(labels_np, preds_np)

    ### Calculate AUC
    # AUC requires probability scores, not just predictions
    # For binary: use class 1 probabilities
    # For multiclass: use one-vs-rest (OVR) or one-vs-one (OVO)

    try:
        if net.num_classes == 2:
            # Binary classification: use probabilities of positive class
            auc = roc_auc_score(labels_np, all_probs[:, 1].numpy())
        else:
            # Multiclass: use one-vs-rest with all class probabilities
            # Check if we have at least 2 classes in the current split
            unique_labels = np.unique(labels_np)
            if len(unique_labels) >= 2:
                auc = roc_auc_score(
                    labels_np, 
                    all_probs.numpy(), 
                    multi_class='ovr',  # One-vs-Rest
                    average='weighted'   # Weight by support
                )
            else:
                auc = float('nan')  # Not enough classes to compute AUC
    except ValueError as e:
        # Handle edge cases (e.g., only one class present)
        auc = float('nan')
        print(f"Warning: Could not compute AUC - {e}")

    if verbose:
        print(f"\t{split_name} set: Loss: {avg_loss:.4f}, Acc: {avg_acc*100:.2f}%, "
            f"Balanced Acc: {balanced_acc*100:.2f}%, F1: {f1:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "balanced_accuracy": balanced_acc,
        "f1": f1,
        "mcc": mcc,
        "auc": auc,  
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
    train_auc_history = []
    train_last_eval = {}

    # Validation
    valid_loss_history = []
    valid_acc_history = []
    valid_balanced_acc_history = []
    valid_f1_history = []
    valid_mcc_history = []
    valid_auc_history = []
    valid_last_eval = {}

    # Test
    test_loss_history = []
    test_acc_history = []
    test_balanced_acc_history = []
    test_f1_history = []
    test_mcc_history = []
    test_auc_history = []
    test_last_eval = {}

    # TODO:
        # implement early stopping
        # LR_schedulaer on VALIDATION set


    # read info
    device = config["DEVICE"]
    lr = config["LR"]
    step_size = config["LR_DECAY_STEPS_EPOCHS"]
    gamma = config["LR_DECAY_GAMMA"]
    num_epochs = config["NUM_EPOCHS"]
    eval_epoch_freq = config["EVAL_EPOCH_FREQ"]
    l2_reg = config["L2_REG"]
    min_lr=1e-7

    # --- Verify ESM is Frozen ---
    if net.esm_model is not None:
        assert all(not p.requires_grad for p in net.esm_model.parameters()), \
            "ESM model parameters should be frozen!"
        
    # optimizer
    optimizer = AdamW(net.class_head.parameters(), lr=lr, weight_decay=l2_reg) # ONLY train the classification head

    # learning rate schefuler
    #LR_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) # step_size=N → the LR changes after step() has been called N times

    LR_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs, # Decay over the entire training duration
        eta_min=1e-7,          # Don't go below this LR
    )

    # LR_scheduler = ReduceLROnPlateau(
    #     optimizer, 
    #     mode='min',           # Change to 'min' since we are tracking loss
    #     factor=gamma,         # Multiply LR by 0.5 when plateau detected
    #     patience=10,          # Wait 10 evaluations before reducing LR
    #     verbose=True,         # Print when LR changes
    #     min_lr=min_lr,          # Don't go below this LR
    #     threshold=5e-3,       # minimum change in the monitored quantity to qualify as an "improvement." If the loss improvement is smaller than this, it's considered a plateau.
    #     threshold_mode='abs'  # Changed from 'rel' to 'abs' for fixed minimum improvement
    # )

    # LR_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, 
    #     T_0=5,          # Number of epochs for the first restart cycle
    #     T_mult=2,         # Multiply T_0 by 2 after each restart (makes cycles grow)
    #     eta_min=min_lr    # The minimum LR to anneal down to
    # )

    # Set model to training mode
    net.train()

    pbar_epochs = trange(1, num_epochs + 1, desc="Training", unit="epoch", position=0, leave=True)

    for epoch_idx in pbar_epochs:

        net.train() # back to train mode

        pbar = tqdm(train_dl, desc=f"Epoch {epoch_idx}", unit="train batch", position=1, leave=False)

        # train natches
        epoch_loss = torch.zeros(1, dtype=torch.float32, device=device)

        for batch in pbar:

            # set model to trainign mode
            net.train()

            # Add LR as postfix
            # Update variable name here as well:
            current_lr = LR_scheduler.optimizer.param_groups[0]['lr'] # Use .optimizer.param_groups[0]['lr'] for clean access
            pbar_epochs.set_postfix(LR=f"{current_lr:.6f}")

            #move to device
            seq = batch.get("input_ids").to(device)
            attention_mask = batch.get("attention_mask").to(device)
            label = batch.get("label").to(device)
            emb = batch.get("embs").to(device)
                        
            # Forward apss
            if from_precomputed_embs:
                output = net(precomputed_embs=emb, attention_mask=attention_mask)
            else:
                output = net(seq, attention_mask=attention_mask)

            # compute loss
            loss = loss_fn(output, label) # crossEntripyLoss() expects RAW logits, so it is correct passing direclty "outpu"
            # clear previous gradients
            optimizer.zero_grad(set_to_none=True)  
            # compute gradients (backpropgation)
            loss.backward()
            # paramers update
            optimizer.step()

            # Track loss
            epoch_loss += loss.item()

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
            train_auc_history.append(train_metrics["auc"])

            valid_loss_history.append(valid_metrics["loss"])
            valid_acc_history.append(valid_metrics["accuracy"])
            valid_balanced_acc_history.append(valid_metrics["balanced_accuracy"])
            valid_f1_history.append(valid_metrics["f1"])
            valid_mcc_history.append(valid_metrics["mcc"])
            valid_auc_history.append(valid_metrics["auc"])

            test_loss_history.append(test_metrics["loss"])
            test_acc_history.append(test_metrics["accuracy"])
            test_balanced_acc_history.append(test_metrics["balanced_accuracy"])
            test_f1_history.append(test_metrics["f1"])
            test_mcc_history.append(test_metrics["mcc"])
            test_auc_history.append(test_metrics["auc"])

            # Save last evaluation results
            train_last_eval = train_metrics
            valid_last_eval = valid_metrics
            test_last_eval = test_metrics

            ### GPU monitoring
            #monitor_gpu_memory()

        # average loss per batch
        avg_epoch_loss = (epoch_loss / len(train_dl)).item()

        # increment learning rate decay counter PER EPOCH
        LR_scheduler.step() # increments the internal counter
        #LR_scheduler.step(avg_epoch_loss)  #plteau
        #print(f"Epoch {epoch_idx}, New LR: {exp_lr.get_last_lr()[0]}")

    # claean up
    torch.cuda.empty_cache() 

    return (
        # Train
        train_loss_history,
        train_acc_history,
        train_balanced_acc_history,
        train_f1_history,
        test_mcc_history, 
        train_auc_history,
        train_last_eval,
        
        # Validation
        valid_loss_history,
        valid_acc_history,
        valid_balanced_acc_history,
        valid_f1_history,
        valid_mcc_history, 
        valid_auc_history,
        valid_last_eval,
        
        # Test
        test_loss_history,
        test_acc_history,
        test_balanced_acc_history, 
        test_f1_history,
        test_mcc_history,
        test_auc_history,
        test_last_eval
    ) 



def summarize_results(
    training_output,
    num_classes=2,
    plot_train=True,
    plot_val=False,
    plot_test=True
):
    """
    Comprehensive summary of training with multiple metrics, visualizations, and confusion matrices.
    
    Args:
        training_output: Tuple containing all training histories and evaluation results
            (train_loss_history, train_acc_history, train_balanced_acc_history, train_f1_history,
             train_mcc_history, train_auc_history, train_last_eval,
             valid_loss_history, valid_acc_history, valid_balanced_acc_history, valid_f1_history,
             valid_mcc_history, valid_auc_history, valid_last_eval,
             test_loss_history, test_acc_history, test_balanced_acc_history, test_f1_history,
             test_mcc_history, test_auc_history, test_last_eval)
        num_classes: Number of classes in classification task
        plot_train: Whether to plot training data
        plot_val: Whether to plot validation data
        plot_test: Whether to plot test data
    """
    
    # Unpack the training output tuple
    (train_loss_history, train_acc_history, train_balanced_acc_history, train_f1_history,
     train_mcc_history, train_auc_history, train_last_eval,
     valid_loss_history, valid_acc_history, valid_balanced_acc_history, valid_f1_history,
     valid_mcc_history, valid_auc_history, valid_last_eval,
     test_loss_history, test_acc_history, test_balanced_acc_history, test_f1_history,
     test_mcc_history, test_auc_history, test_last_eval) = training_output
    
    # ============================================================================
    # SECTION 1: Comprehensive Metrics Table
    # ============================================================================
    print("\n" + "="*80)
    print(" "*25 + "FINAL EVALUATION METRICS")
    print("="*80)
    
    # Build header based on what's being plotted
    headers = ["Metric"]
    if plot_train:
        headers.append("Train")
    if plot_val:
        headers.append("Validation")
    if plot_test:
        headers.append("Test")
    
    metrics_table = [headers]
    metrics_table.append(["-"*15] * len(headers))
    
    # Helper function to add metric rows
    def add_metric_row(metric_name, train_val, valid_val, test_val, format_str="{:.4f}"):
        row = [metric_name]
        if plot_train:
            row.append(format_str.format(train_val))
        if plot_val:
            row.append(format_str.format(valid_val))
        if plot_test:
            row.append(format_str.format(test_val))
        metrics_table.append(row)
    
    # Add all metrics
    add_metric_row("Loss",
                   train_last_eval['loss'],
                   valid_last_eval['loss'],
                   test_last_eval['loss'])
    
    add_metric_row("Accuracy",
                   train_last_eval['accuracy']*100,
                   valid_last_eval['accuracy']*100,
                   test_last_eval['accuracy']*100,
                   "{:.2f}%")
    
    add_metric_row("Balanced Acc",
                   train_last_eval['balanced_accuracy']*100,
                   valid_last_eval['balanced_accuracy']*100,
                   test_last_eval['balanced_accuracy']*100,
                   "{:.2f}%")
    
    add_metric_row("F1 Score",
                   train_last_eval['f1'],
                   valid_last_eval['f1'],
                   test_last_eval['f1'])
    
    add_metric_row("MCC",
                   train_last_eval['mcc'],
                   valid_last_eval['mcc'],
                   test_last_eval['mcc'])
    
    add_metric_row("AUC-ROC",
                   train_last_eval.get('auc', float('nan')),
                   valid_last_eval.get('auc', float('nan')),
                   test_last_eval.get('auc', float('nan')))
    
    # Print the table
    for row in metrics_table:
        formatted_row = [f"{row[0]:<15}"] + [f"{val:>15}" for val in row[1:]]
        print("".join(formatted_row))
    
    print("="*80)
    
    # ============================================================================
    # SECTION 2: Visualizations
    # ============================================================================
    sns.set_style("whitegrid")
    
    # Count how many datasets to plot
    num_datasets = sum([plot_train, plot_val, plot_test])
    
    if num_datasets == 0:
        print("\nNo datasets selected for plotting.")
        return
    
    # Adjust figure layout - now with more rows for additional metrics
    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(5, 3, hspace=0.4, wspace=0.3)
    
    # Row 1: Loss and Accuracy curves
    ax1 = fig.add_subplot(gs[0, :2])
    if plot_train:
        ax1.plot(train_loss_history, label="Train", marker='o', linewidth=2, markersize=6)
    if plot_val:
        ax1.plot(valid_loss_history, label="Valid", marker='s', linewidth=2, markersize=6)
    if plot_test:
        ax1.plot(test_loss_history, label="Test", marker='^', linewidth=2, markersize=6)
    ax1.set_title("Loss History", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Evaluation Step", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 2])
    if plot_train:
        ax2.plot(train_acc_history, label="Train", marker='o', linewidth=2, markersize=6)
    if plot_val:
        ax2.plot(valid_acc_history, label="Valid", marker='s', linewidth=2, markersize=6)
    if plot_test:
        ax2.plot(test_acc_history, label="Test", marker='^', linewidth=2, markersize=6)
    ax2.set_title("Accuracy History", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Evaluation Step", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # Row 2: Balanced Accuracy, F1 Score, and MCC curves
    ax3 = fig.add_subplot(gs[1, 0])
    if plot_train:
        ax3.plot(train_balanced_acc_history, label="Train", marker='o', linewidth=2, markersize=6)
    if plot_val:
        ax3.plot(valid_balanced_acc_history, label="Valid", marker='s', linewidth=2, markersize=6)
    if plot_test:
        ax3.plot(test_balanced_acc_history, label="Test", marker='^', linewidth=2, markersize=6)
    ax3.set_title("Balanced Accuracy History", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Evaluation Step", fontsize=11)
    ax3.set_ylabel("Balanced Accuracy", fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    if plot_train:
        ax4.plot(train_f1_history, label="Train", marker='o', linewidth=2, markersize=6)
    if plot_val:
        ax4.plot(valid_f1_history, label="Valid", marker='s', linewidth=2, markersize=6)
    if plot_test:
        ax4.plot(test_f1_history, label="Test", marker='^', linewidth=2, markersize=6)
    ax4.set_title("F1 Score History", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Evaluation Step", fontsize=11)
    ax4.set_ylabel("F1 Score", fontsize=11)
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 2])
    if plot_train:
        ax5.plot(train_mcc_history, label="Train", marker='o', linewidth=2, markersize=6)
    if plot_val:
        ax5.plot(valid_mcc_history, label="Valid", marker='s', linewidth=2, markersize=6)
    if plot_test:
        ax5.plot(test_mcc_history, label="Test", marker='^', linewidth=2, markersize=6)
    ax5.set_title("MCC History", fontsize=14, fontweight='bold')
    ax5.set_xlabel("Evaluation Step", fontsize=11)
    ax5.set_ylabel("Matthews Correlation Coefficient", fontsize=11)
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)  # Reference line at 0
    
    # Row 3: AUC-ROC curve
    ax6 = fig.add_subplot(gs[2, :])
    if plot_train:
        ax6.plot(train_auc_history, label="Train", marker='o', linewidth=2, markersize=6)
    if plot_val:
        ax6.plot(valid_auc_history, label="Valid", marker='s', linewidth=2, markersize=6)
    if plot_test:
        ax6.plot(test_auc_history, label="Test", marker='^', linewidth=2, markersize=6)
    ax6.set_title("AUC-ROC History", fontsize=14, fontweight='bold')
    ax6.set_xlabel("Evaluation Step", fontsize=11)
    ax6.set_ylabel("AUC-ROC Score", fontsize=11)
    ax6.legend(fontsize=10)
    ax6.grid(alpha=0.3)
    ax6.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)  # Random classifier baseline
    ax6.set_ylim([0, 1.05])
    
    # Row 4: Confusion Matrices (only for selected datasets)
    datasets_to_plot = []
    grid_positions = [gs[3, 0], gs[3, 1], gs[3, 2]]
    position_idx = 0
    
    if plot_train:
        datasets_to_plot.append((train_last_eval, "Train", grid_positions[position_idx]))
        position_idx += 1
    if plot_val:
        datasets_to_plot.append((valid_last_eval, "Valid", grid_positions[position_idx]))
        position_idx += 1
    if plot_test:
        datasets_to_plot.append((test_last_eval, "Test", grid_positions[position_idx]))
        position_idx += 1
    
    for eval_dict, name, grid_pos in datasets_to_plot:
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
    
    # Row 5: Metrics comparison bar chart
    ax7 = fig.add_subplot(gs[4, :])
    metrics_names = ['Accuracy', 'Balanced Accuracy', 'F1 Score', 'MCC', 'AUC-ROC']
    
    # Prepare data based on what's being plotted
    datasets_metrics = []
    labels = []
    
    if plot_train:
        train_metrics = [
            train_last_eval['accuracy'],
            train_last_eval['balanced_accuracy'],
            train_last_eval['f1'],
            (train_last_eval['mcc'] + 1) / 2,  # Normalize MCC to [0,1] for visualization
            train_last_eval.get('auc', 0.5)
        ]
        datasets_metrics.append(train_metrics)
        labels.append('Train')
    
    if plot_val:
        valid_metrics = [
            valid_last_eval['accuracy'],
            valid_last_eval['balanced_accuracy'],
            valid_last_eval['f1'],
            (valid_last_eval['mcc'] + 1) / 2,
            valid_last_eval.get('auc', 0.5)
        ]
        datasets_metrics.append(valid_metrics)
        labels.append('Valid')
    
    if plot_test:
        test_metrics = [
            test_last_eval['accuracy'],
            test_last_eval['balanced_accuracy'],
            test_last_eval['f1'],
            (test_last_eval['mcc'] + 1) / 2,
            test_last_eval.get('auc', 0.5)
        ]
        datasets_metrics.append(test_metrics)
        labels.append('Test')
    
    # Plot bars
    x = np.arange(len(metrics_names))
    width = 0.8 / num_datasets  # Adjust width based on number of datasets
    
    for i, (metrics, label) in enumerate(zip(datasets_metrics, labels)):
        offset = (i - num_datasets/2 + 0.5) * width
        ax7.bar(x + offset, metrics, width, label=label, alpha=0.8)
    
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
    # SECTION 3: Detailed Classification Reports
    # ============================================================================
    if plot_test:
        print("\n" + "="*80)
        print(" "*25 + "TEST SET CLASSIFICATION REPORT")
        print("="*80)
        
        # Get unique classes actually present in test set
        unique_test_classes = np.unique(np.concatenate([
            test_last_eval['labels'].numpy(),
            test_last_eval['pred_labels'].numpy()
        ]))
        
        print(classification_report(
            test_last_eval['labels'].numpy(),
            test_last_eval['pred_labels'].numpy(),
            labels=unique_test_classes,  # Only use classes that are present
            target_names=[f'Class {i}' for i in unique_test_classes],
            digits=4,
            zero_division=0
        ))
        print("="*80 + "\n")

    if plot_val:
        print("\n" + "="*80)
        print(" "*25 + "VALIDATION SET CLASSIFICATION REPORT")
        print("="*80)
        
        # Get unique classes actually present in validation set
        unique_valid_classes = np.unique(np.concatenate([
            valid_last_eval['labels'].numpy(),
            valid_last_eval['pred_labels'].numpy()
        ]))
        
        print(classification_report(
            valid_last_eval['labels'].numpy(),
            valid_last_eval['pred_labels'].numpy(),
            labels=unique_valid_classes,
            target_names=[f'Class {i}' for i in unique_valid_classes],
            digits=4,
            zero_division=0
        ))
        print("="*80 + "\n")

    if plot_train:
        print("\n" + "="*80)
        print(" "*25 + "TRAIN SET CLASSIFICATION REPORT")
        print("="*80)
        
        # Get unique classes actually present in training set
        unique_train_classes = np.unique(np.concatenate([
            train_last_eval['labels'].numpy(),
            train_last_eval['pred_labels'].numpy()
        ]))
        
        print(classification_report(
            train_last_eval['labels'].numpy(),
            train_last_eval['pred_labels'].numpy(),
            labels=unique_train_classes,
            target_names=[f'Class {i}' for i in unique_train_classes],
            digits=4,
            zero_division=0
        ))
        print("="*80 + "\n")


def get_uniprot_info(uniprot_id):
    """Fetch gene name and function from UniProt"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.txt"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            lines = response.text.split('\n')
            gene_name = ""
            function = ""
            
            # Extract gene name
            for line in lines:
                if line.startswith('GN   Name='):
                    gene_name = line.split('Name=')[1].split(';')[0].strip()
                    break
            
            # Extract function (first CC line with "FUNCTION:")
            in_function = False
            for line in lines:
                if 'FUNCTION:' in line:
                    in_function = True
                    function = line.split('FUNCTION:')[1].strip()
                elif in_function:
                    if line.startswith('CC   '):
                        function += " " + line[5:].strip()
                    else:
                        break
            
            # Truncate function to ~150 chars
            if len(function) > 150:
                function = function[:147] + "..."
            
            return gene_name, function
        else:
            return "N/A", "Failed to fetch"
    except Exception as e:
        return "N/A", f"Error: {str(e)}"