
from torch.optim import AdamW
from torch import nn
from torch.optim import lr_scheduler
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange, tqdm


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



def evaluate_model(net, dataloader, device, loss_fn, split_name="Eval"):
    """
    Evaluate the model on a given dataset (train/val/test).
    
    Args:
        net: torch.nn.Module, the model
        dataloader: torch DataLoader, dataset to evaluate on
        device: torch.device
        loss_fn: loss function (e.g. nn.CrossEntropyLoss)
        split_name: string, e.g. "Train", "Validation", "Test"
        
    Returns:
        metrics dict with:
            - loss (float)
            - accuracy (float)
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
        for seq, attention_mask, label in dataloader:
            seq, attention_mask, label = seq.to(device), attention_mask.to(device), label.to(device)

            output = net(seq, attention_mask=attention_mask)
            probs = torch.softmax(output, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            loss = loss_fn(output, label) # alredy compute SINGLE BATCH AVERAGE

            total_loss += loss
            total_correct += (preds == label).sum()

            all_probs.append(probs.cpu())
            all_probs_class1.append(probs.cpu()[:, 1])
            all_labels.append(label.cpu())
            all_preds.append(preds.cpu())

    # average loss per batch
    avg_loss = (total_loss / len(dataloader)).item()
    # accuracy over all samples
    avg_acc = (total_correct / len(dataloader.dataset)).item()

    # concat all batch results
    all_probs = torch.cat(all_probs, dim=0).cpu()
    all_probs_class1 = torch.cat(all_probs_class1, dim=0).cpu()
    all_labels = torch.cat(all_labels, dim=0).cpu()
    all_preds = torch.cat(all_preds, dim=0).cpu()

    print(f"\t{split_name} set: Average loss: {avg_loss:.4f}, Accuracy: {avg_acc*100:.2f}%")

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "probs": all_probs,
        "probs_class1": all_probs_class1,
        "labels": all_labels,
        "pred_labels": all_preds
    }



def train(net, train_dl, valid_dl, test_dl, config):

    # Train
    train_loss_history = []
    train_acc_history = []
    train_last_eval = {}

    # Validation
    valid_loss_history = []
    valid_acc_history = []
    valid_last_eval = {}

    # Test
    test_loss_history = []
    test_acc_history = []
    test_last_eval = {}


    # read info
    device = config["DEVICE"]
    lr = config["LR"]
    step_size = config["LR_DECAY_STEPS"]
    gamma = config["LR_DECAY_GAMMA"]
    num_iters = config["NUM_ITERS"]
    eval_size = config["EVAL_SIZE"]

    # Initialise obj
    optimizer = AdamW(net.feature_fn.parameters(), lr=lr) # ONLY train the classification head
    exp_lr = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) # step_size=N → the LR changes after step() has been called N times
    loss_fn = nn.CrossEntropyLoss() # binary classfication, needs logits

    # Set model to training mode
    net.train()

    # Initialize progress bar for total iterations
    pbar = trange(1, num_iters + 1, desc="Training", unit="iter")

    # Iteration index, ie One forward and backward pass through one batch
    iter_idx = 1


    for iter_idx in pbar:

        #print("Iteration number: ", iter_idx)
        net.train() # back to train mode

        for [seq, attention_mask, label] in train_dl: # take one BATCH

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
            # increment learning rate decay counter  
            #exp_lr.step() # increments the internal counter

            # EVALUATION EVERY TOT ITERS
            # NB if num_iters%eval_size != 0 the last iters will not be evaluated
            if iter_idx % eval_size == 0:

                print(f"--- Evaluation at iteration {iter_idx} ---")

                # Return metrics of current evaluation
                train_metrics = evaluate_model(net, train_dl, device, loss_fn, split_name="Train")
                valid_metrics = evaluate_model(net, valid_dl, device, loss_fn, split_name="Validation")
                test_metrics  = evaluate_model(net, test_dl, device, loss_fn, split_name="Test")

                # Save histories
                train_loss_history.append(train_metrics["loss"])
                train_acc_history.append(train_metrics["accuracy"])
                valid_loss_history.append(valid_metrics["loss"])
                valid_acc_history.append(valid_metrics["accuracy"])
                test_loss_history.append(test_metrics["loss"])
                test_acc_history.append(test_metrics["accuracy"])

                # Save last evaluation results
                train_last_eval = train_metrics
                valid_last_eval = valid_metrics
                test_last_eval = test_metrics

                ### GPU monitoring
                monitor_gpu_memory()

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
