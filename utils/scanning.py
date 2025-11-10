import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch
from scipy.ndimage import gaussian_filter1d


from . import dataset as my_dataset
from . import my_functions as mf




def get_data_from_dict(cache_data, protein):

    idx = cache_data["protein"].index(protein)
    selected_protein_data = {}
    for key, value in cache_data.items():
        if key in ["protein", "set", "sequence", "truncated_sequence"]:
            # These are lists - wrap single item in a list
            selected_protein_data[key] = [value[idx]]
        elif key == "label":
            # Convert to 1D tensor with batch dimension
            selected_protein_data[key] = torch.tensor([value[idx].item()], dtype=torch.long)
        elif key in ["input_ids", "attention_mask", "embedding"]:
            # These are 2D tensors - use slicing to keep batch dimension
            if value is not None:
                selected_protein_data[key] = value[idx:idx+1]
            else:
                selected_protein_data[key] = None
        else:
            # Fallback for any other keys
            selected_protein_data[key] = [value[idx]] if isinstance(value, list) else value[idx:idx+1]
    return selected_protein_data


def get_prob_single_protein(model, single_protein_info, device="cuda"):
    
    # Extract data from input dict
    original_sequence = single_protein_info['truncated_sequence'][0]
    seq_len = len(original_sequence)
    label = single_protein_info['label'][0]
    split_name = single_protein_info['set'][0]
    protein_name = single_protein_info['protein'][0]
        
    baseline_dl = my_dataset.create_dataloader(
        single_protein_info, 
        batch_size=1,
        shuffle=False
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        baseline_dict = mf.evaluate_model(
            model, baseline_dl, device, 
            split_name="Single protein", 
            verbose=False,
            from_precomputed_embs=False
        )
    
    return baseline_dict["probs_class1"][0].item()


def create_all_mutations(truncated_seq, window_size = 3, substitute_aas =  ["A", "R", "E", "F"]):

    assert window_size % 2 == 1, "window_size must be odd"

        
    mutated_sequences = []
    seq_array = list(truncated_seq) # already truncated
    seq_len = len(truncated_seq)
    half_window = window_size // 2
    
    for pos in tqdm(range(seq_len), desc="Generating mutations"):
        # Calculate window boundaries
        window_start = max(0, pos - half_window)
        window_end = min(seq_len, pos + half_window + 1)
        
        for sub_aa in substitute_aas:
            # Create mutated sequence
            mut_seq = seq_array.copy()
            true_sub_count = 0
            
            for idx in range(window_start, window_end):
                if seq_array[idx] != sub_aa:
                    true_sub_count += 1
                mut_seq[idx] = sub_aa
            
            mutated_sequences.append(''.join(mut_seq))

    return mutated_sequences


def multi_aa_scanning_final(model,
                            baseline_prob,
                            cache_mutations,
                            substitute_aas,
                            window_size,
                            normalise_true_substitution=False,
                            device="cuda",
                            batch_size=64):
    """
    Final part of multi-AA scanning: evaluates pre-generated mutated sequences
    and computes delta probabilities per position and amino acid.

    Args:
        model: Trained model for inference.
        baseline_prob: Baseline probability for wild-type protein.
        cache_mutations: Dict containing all mutated protein data (already tokenized).
        substitute_aas: List of amino acids used for mutation.
        window_size: Size of substitution window (odd integer).
        normalise_true_substitution: Whether to normalize by number of true substitutions.
        device: 'cuda' or 'cpu'.

    Returns:
        dict with:
            - 'baseline_prob': float
            - 'delta_probs': np.array (mean delta per position)
            - 'mutated_probs': np.array (mean mutated prob per position)
            - 'per_aa_results': dict mapping AA -> {'delta_probs', 'mutated_probs'}
            - 'sequence': str (original sequence)
            - 'protein_name': str
            - 'substitute_aas': list
    """
    import numpy as np
    import torch
    from tqdm import tqdm
    import warnings

    model.eval()

    mutated_sequences = cache_mutations["truncated_sequence"]
    seq_len = len(mutated_sequences[0])
    n_positions = seq_len
    n_substitute = len(substitute_aas)
    n_mutations = len(mutated_sequences)

    # --- Evaluate all mutated sequences ---
    mutated_dl = my_dataset.create_dataloader(
        cache_mutations,
        batch_size=batch_size,  
        shuffle=False
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mutated_eval = mf.evaluate_model(
            model,
            mutated_dl,
            device,
            split_name="mutations",
            verbose=False,
            from_precomputed_embs=False
        )

    mutated_probs = np.array(mutated_eval["probs_class1"])  # shape (n_mutations,)
    delta_probs = mutated_probs - baseline_prob

    # --- Reshape results [positions, amino_acids] ---
    delta_reshaped = delta_probs.reshape(n_positions, n_substitute)
    mutated_reshaped = mutated_probs.reshape(n_positions, n_substitute)

    # --- Compute per-AA results ---
    per_aa_results = {
        aa: {
            "delta_probs": delta_reshaped[:, i],
            "mutated_probs": mutated_reshaped[:, i],
        }
        for i, aa in enumerate(substitute_aas)
    }

    # --- Average across amino acids ---
    delta_mean = np.mean(delta_reshaped, axis=1)
    mutated_mean = np.mean(mutated_reshaped, axis=1)

    # --- Optional normalization ---
    if normalise_true_substitution:
        # approximate true substitution count (since window overlaps are fixed)
        half_window = window_size // 2
        true_sub_count = np.ones(seq_len) * window_size
        delta_mean /= true_sub_count

    # --- Prepare final output ---
    out = {
        "baseline_prob": baseline_prob,
        "delta_probs": delta_mean,
        "mutated_probs": mutated_mean,
        "per_aa_results": per_aa_results,
        "sequence": cache_mutations["truncated_sequence"][0],
        "protein_name": cache_mutations["protein"][0],
        "substitute_aas": substitute_aas,
    }

    return out



# def multi_aa_scanning(model, tokenizer, 
#                       single_protein_info, 
#                       window_size: int = 3, 
#                       substitute_aas: list[str] = None,
#                       normalise_true_substitution: bool = False,
#                       protein_max_length: int = 1000,
#                       device: str = "cuda") -> dict:
#     """
#     Perform scanning by replacing a window of residues with multiple amino acids.
    
#     For each position in the protein sequence, substitutes the surrounding window
#     with each amino acid in substitute_aas and computes the resulting probabilities.
#     Results are averaged across all substitute amino acids per position.
    
#     Args:
#         model: Trained model for inference
#         tokenizer: Tokenizer instance for encoding sequences
#         single_protein_info: Dict containing protein data with keys:
#             - 'truncated_sequence': list with single protein sequence string
#             - 'label': tensor with single label value
#             - 'set': list with split name (e.g., 'test')
#             - 'protein': list with protein name
#         window_size: Size of mutation window, must be odd (default: 3)
#         substitute_aas: List of amino acids to substitute (default: ["A", "R", "E", "F"])
#         normalise_true_substitution: Whether to normalize by number of true substitutions
#         protein_max_length: Maximum sequence length for tokenizer
#         device: Device for inference ('cuda' or 'cpu')
    
#     Returns:
#         dict with keys:
#             - 'baseline_prob': float, baseline probability for wild-type
#             - 'delta_probs': array, mean delta prob per position
#             - 'mutated_probs': array, mean mutated prob per position
#             - 'per_aa_results': dict mapping AA -> {'delta_probs', 'mutated_probs'}
#             - 'sequence': original truncated sequence
#             - 'protein_name': protein identifier
#             - 'substitute_aas': amino acids tested
#     """
#     import warnings
#     import numpy as np
#     import torch
#     from torch.utils.data import DataLoader
#     from tqdm import tqdm
    
#     if substitute_aas is None:
#         substitute_aas = ["A", "R", "E", "F"]
    
#     assert window_size % 2 == 1, "window_size must be odd"
    
#     # Extract data from input dict
#     original_sequence = single_protein_info['truncated_sequence'][0]
#     seq_len = len(original_sequence)
#     protein_name = single_protein_info['protein'][0]
    
#     model.eval()
    
#     # ========== GET BASELINE PROBABILITY ==========
    
#     baseline_encoded = tokenizer(
#         [original_sequence],
#         padding='max_length',
#         max_length=protein_max_length,
#         truncation=True,
#         return_tensors="pt"
#     )
    
#     with torch.no_grad():
#         baseline_outputs = model(
#             baseline_encoded['input_ids'].to(device),
#             attention_mask=baseline_encoded['attention_mask'].to(device)
#         )
#         baseline_logits = baseline_outputs.logits[0]
#         baseline_probs = torch.softmax(baseline_logits, dim=-1)
#         baseline_prob = baseline_probs[1].cpu().item()  # Probability of class 1
    
#     # ========== GENERATE MUTATED SEQUENCES ==========
    
#     seq_array = list(original_sequence)
#     half_window = window_size // 2
    
#     mutated_sequences = []
#     mutation_metadata = []
    
#     for pos in tqdm(range(seq_len), desc="Generating mutations"):
#         # Calculate window boundaries
#         window_start = max(0, pos - half_window)
#         window_end = min(seq_len, pos + half_window + 1)
        
#         for sub_aa in substitute_aas:
#             # Create mutated sequence
#             mut_seq = seq_array.copy()
#             true_sub_count = 0
            
#             for idx in range(window_start, window_end):
#                 if seq_array[idx] != sub_aa:
#                     true_sub_count += 1
#                 mut_seq[idx] = sub_aa
            
#             mutated_sequences.append(''.join(mut_seq))
#             mutation_metadata.append({
#                 'position': pos,
#                 'substitute_aa': sub_aa,
#                 'true_sub_count': true_sub_count
#             })
    
#     n_mutations = len(mutated_sequences)
    
#     # ========== TOKENIZE AND EVALUATE MUTATED SEQUENCES ==========
    
#     encoded = tokenizer(
#         mutated_sequences,
#         padding='max_length',
#         max_length=protein_max_length,
#         truncation=True,
#         return_tensors="pt"
#     )
    
#     # Determine batch size
#     batch_size = max(1, n_mutations // 100)
    
#     mutated_probs = []
    
#     # Process in batches
#     for batch_idx in tqdm(range(0, n_mutations, batch_size), desc="Evaluating mutations"):
#         batch_end = min(batch_idx + batch_size, n_mutations)
        
#         batch_input_ids = encoded['input_ids'][batch_idx:batch_end].to(device)
#         batch_attention_mask = encoded['attention_mask'][batch_idx:batch_end].to(device)
        
#         with torch.no_grad():
#             batch_outputs = model(
#                 batch_input_ids,
#                 attention_mask=batch_attention_mask
#             )
#             batch_logits = batch_outputs.logits  # Shape: (batch_size, 2)
#             batch_softmax = torch.softmax(batch_logits, dim=-1)  # Shape: (batch_size, 2)
#             batch_class1_probs = batch_softmax[:, 1].cpu().numpy()  # Get class 1 probabilities
        
#         mutated_probs.extend(batch_class1_probs)
    
#     mutated_probs = np.array(mutated_probs)
    
#     # ========== PROCESS RESULTS ==========
    
#     delta_probs = mutated_probs - baseline_prob
    
#     # Reshape to [positions, amino_acids]
#     n_positions = seq_len
#     n_substitute = len(substitute_aas)
    
#     delta_reshaped = delta_probs.reshape(n_positions, n_substitute)
#     mutated_reshaped = mutated_probs.reshape(n_positions, n_substitute)
    
#     # Store per-AA results
#     per_aa_results = {
#         aa: {
#             'delta_probs': delta_reshaped[:, idx],
#             'mutated_probs': mutated_reshaped[:, idx]
#         }
#         for idx, aa in enumerate(substitute_aas)
#     }
    
#     # Average across amino acids
#     delta_mean = np.mean(delta_reshaped, axis=1)
#     mutated_mean = np.mean(mutated_reshaped, axis=1)
    
#     # Optional normalization by true substitution count
#     if normalise_true_substitution:
#         true_sub_array = np.array([m['true_sub_count'] for m in mutation_metadata])
#         true_sub_reshaped = true_sub_array.reshape(n_positions, n_substitute)
#         mean_true_subs = np.mean(true_sub_reshaped, axis=1)
        
#         # Avoid division by zero
#         delta_mean = np.divide(delta_mean, mean_true_subs, 
#                                where=mean_true_subs > 0, 
#                                out=np.zeros_like(delta_mean))
    
#     return {
#         'baseline_prob': baseline_prob,
#         'delta_probs': delta_mean,
#         'mutated_probs': mutated_mean,
#         'per_aa_results': per_aa_results,
#         'sequence': original_sequence,
#         'protein_name': protein_name,
#         'substitute_aas': substitute_aas
#     }



# def multi_aa_scanning(model, tokenizer, 
#                       single_protein_info, 
#                       window_size: int = 3, 
#                       substitute_aas=["A", "R", "E", "F"],
#                       normalise_true_substitution=False,
#                       protein_max_lenght=1000,
#                       device="cuda"):
#     """
#     Perform scanning by replacing a window of residues with multiple amino acids.
#     For each position, substitutes with each amino acid in substitute_aas and averages the results.
#     Window_size should be odd (so there is a center residue).
#     Returns baseline probability and delta probabilities mapped to positions.

#     Args:
#         model: Trained model
#         tokenizer: Tokenizer instance
#         single_protein_info: Series containing single protein data
#         window_size: Size of mutation window (must be odd)
#         device: Device to run inference on
#         substitute_aas: List of amino acids to substitute (default: ["A", "R", "E", "F"])
#         normalise_true_substitution: Whether to normalize by number of true substitutions
    
#     Returns:
#         dict: Contains baseline_prob, delta_probs, mutated_probs, and per_aa_results
#     """
#     assert window_size % 2 == 1, "window_size must be odd"


#     ### CALCULATE BASELINE PROB ###

#     # Create dataloader for single protein
#     single_protein_dl = my_dataset.create_dataloader(
#         single_protein_info, 
#         split_name=single_protein_info["set"][0], 
#         batch_size=1,  # only 1 prtein
#         shuffle=False
#     )

#     print(next(iter(single_protein_dl)))

#     # Calculate baseline prob
#     with warnings.catch_warnings(): 
#         warnings.simplefilter("ignore", category=UserWarning)
#         baseline_dict = mf.evaluate_model(model, single_protein_dl, device, split_name="Single protein", verbose=False, 
#                                         from_precomputed_embs=False # in A-scannng always recopute
#                                         )
#     baseline_p = baseline_dict["probs_class1"][0].item() 


#     ### GENERATE ALL MUTATED SEQUENCES FOR ALL AMINO ACIDS ###
    
#     # Sequence to mutate (use TRUNCATED sequence)
#     original_sequence = single_protein_info['truncated_sequence'][0]
#     seq_list = list(original_sequence)
#     trunc_len = len(seq_list)

#     truncated_mutated_sequences = []
#     mutation_info = []  # Store info about each mutation (position, AA used)

#     # Iterate through each position and each substitute amino acid
#     for i in tqdm(range(trunc_len), desc="Generating mutations"):

#         # Define window boundaries
#         half_w = window_size // 2
#         start = max(0, i - half_w)
#         end = min(trunc_len, i + half_w + 1)

#         # For each position, create mutations with each substitute amino acid
#         for sub_aa in substitute_aas:
#             true_sub_count = 0
            
#             # Create mutated sequence
#             mutated_seq_list = seq_list.copy()
#             for j in range(start, end):
#                 if seq_list[j] != sub_aa:  # Count true substitutions
#                     true_sub_count += 1
#                 mutated_seq_list[j] = sub_aa
            
#             truncated_mutated_sequences.append(''.join(mutated_seq_list))
#             mutation_info.append({
#                 'position': i,
#                 'substitute_aa': sub_aa,
#                 'true_sub_count': true_sub_count
#             })

#     total_mutations = len(truncated_mutated_sequences)


#     ### PREPROCESS ALL MUTATED PROTEINS ###

#         # SAME PRPCEDURE DONE AT THE CRREATION OF DATASET
#         # No NEED TO TRUNCATE AS THE MIUTATED SEEUCNES ARE MUTATED ON THE ALREDY TRUNCATED

#     # ecnode
#     encoded = tokenizer(
#             list(truncated_mutated_sequences),
#             padding='max_length',
#             max_length=protein_max_lenght,
#             truncation=True,
#             return_tensors="pt"
#         )
#     input_ids_tensor = encoded["input_ids"]          # shape: (N, L)
#     attention_mask_tensor = encoded["attention_mask"]

#     # save all information needed to tothe model
#     cache_data = {
#         'protein': [single_protein_info['protein'][0]] * total_mutations,  # Same protein name
#         'label': torch.tensor([single_protein_info['label'][0].item()] * total_mutations),  # Same label
#         'set': [single_protein_info['set'][0]] * total_mutations,  # Same set
#         'sequence': truncated_mutated_sequences,
#         'truncated_sequence': truncated_mutated_sequences,
#         'input_ids': input_ids_tensor,
#         'attention_mask': attention_mask_tensor
#     }

#     # define a batchsize
#     batch_size = max(1, int(total_mutations // 100))
    
#     mutated_dl = my_dataset.create_dataloader(
#         cache_data,
#         split_name=single_protein_info["set"][0], 
#         batch_size=batch_size,         
#         shuffle=False
#     )

#     ### CALCULATE PROBS FOR ALL MUTATED PROTEINS ###

#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", category=UserWarning)
#         print("Evaluating model on mutated sequences...")
#         mutated_dict = mf.evaluate_model(model, mutated_dl, device, split_name="Multi-AA Scan", verbose=False)
    
#     # The result is a tensor of probabilities
#     mutated_probs = mutated_dict["probs_class1"].cpu().numpy()


#     ### CALCULATE DELTAS AND AVERAGE ACROSS AMINO ACIDS ###
    
#     delta_p_all = mutated_probs - baseline_p
    
#     # Organize results by position and amino acid
#     n_positions = trunc_len
#     n_aas = len(substitute_aas)
    
#     # Reshape to [positions, amino_acids]
#     delta_p_reshaped = delta_p_all.reshape(n_positions, n_aas)
#     mutated_probs_reshaped = mutated_probs.reshape(n_positions, n_aas)
    
#     # Store per-AA results for detailed analysis
#     per_aa_results = {}
#     for aa_idx, aa in enumerate(substitute_aas):
#         per_aa_results[aa] = {
#             'delta_probs': delta_p_reshaped[:, aa_idx],
#             'mutated_probs': mutated_probs_reshaped[:, aa_idx]
#         }
    
#     # Calculate mean across amino acids for each position
#     delta_p_mean = np.mean(delta_p_reshaped, axis=1)
#     mutated_probs_mean = np.mean(mutated_probs_reshaped, axis=1)
    
#     # Optional normalization by true substitution count
#     if normalise_true_substitution:
#         # Calculate mean true substitution count per position
#         true_sub_counts = np.array([info['true_sub_count'] for info in mutation_info])
#         true_sub_counts_reshaped = true_sub_counts.reshape(n_positions, n_aas)
#         mean_true_sub_counts = np.mean(true_sub_counts_reshaped, axis=1)
        
#         delta_p_mean = np.where(mean_true_sub_counts > 0, 
#                                 delta_p_mean / mean_true_sub_counts, 
#                                 0.0)

#     return {
#         'baseline_prob': baseline_p,
#         'delta_probs': delta_p_mean,  # Mean across all substitute AAs
#         'mutated_probs': mutated_probs_mean,  # Mean across all substitute AAs
#         'per_aa_results': per_aa_results,  # Individual results per amino acid
#         'sequence': single_protein_info['truncated_sequence'],
#         'protein_name': single_protein_info['protein'],
#         'substitute_aas': substitute_aas
#     }





def plot_multi_aa_scan(scan_results, sigma=3, threshold=True, figsize=(20, 6), 
                       highlight_residues=True, top_n=10, show_sequence=True,
                       style='darkgrid', palette='RdBu_r', show_per_aa=False, plot_range=None):
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
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', bbox=props,
            family='monospace')
    
    if plot_range is not None:
        plt.xlim(plot_range[0], plot_range[1])  # Show only first 50 residues

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



# def alanine_scanning(model, tokenizer, single_protein_info, window_size: int = 3, device="cuda", SUBSTITUTE_AA="A", normalise_true_substitution=False):
#     """
#     Perform scanning by replacing a window of residues with alanine.
#     Window_size should be odd (so there is a center residue).
#     Returns baseline probability and delta probabilities mapped to positions.

#     Args:
#         model: Trained model
#         tokenizer: Tokenizer instance
#         single_protein_info: Series containing single protein data
#         window_size: Size of mutation window (must be odd)
#         device: Device to run inference on
#         SUBSTITUTE_AA: Amino acid to substitute (default: "A" for alanine)
    
#     Returns:
#         dict: Contains baseline_prob, delta_probs, and mutated_probs
#     """
#     assert window_size % 2 == 1, "window_size must be odd"


#     ### CLAUCLATE BASELINE PROB ###

#     # Create a temporary dataframe with the single protein
#     temp_df = pd.DataFrame([single_protein_info])

#     # Create dataloader for single protein
#     single_protein_dl = my_dataset.create_dataloader(
#         temp_df, 
#         set_name=single_protein_info['set'], 
#         batch_size=1, 
#         shuffle=False
#     )

#     # calculate baseline peob
#     with warnings.catch_warnings(): 
#         warnings.simplefilter("ignore", category=UserWarning)
#         baseline_dict = evaluate_model(model, single_protein_dl, device, split_name="Single protein", verbose=False)
#     baseline_p = baseline_dict["probs_class1"][0].item() 


#     ### GENERATE ALL MUTATED PROBS ###
    
#     # Sequence to mutate (use TRUNCATED sequence)
#     seq_list = list(single_protein_info['trunc_sequence'])
#     trunc_len = len(seq_list)

#     mutated_sequences = []
#     true_substitution_count = [] # how many true substitutions (not substituting A->A)

#     # Iterate through each position to generate the mutated sequence for that position
#     for i in tqdm(range(trunc_len), desc="Generating mutations"):

#         # Define window boundaries
#         half_w = window_size // 2
#         start = max(0, i - half_w)
#         end = min(trunc_len, i + half_w + 1)

#         true_sub_count = 0
        
#         # Create mutated sequence
#         mutated_seq_list = seq_list.copy()
#         for j in range(start, end):
#             mutated_seq_list[j] = SUBSTITUTE_AA
#             if seq_list[j] != SUBSTITUTE_AA: # cpunt true substitutions
#                 true_sub_count += 1
#         mutated_sequences.append(''.join(mutated_seq_list)) # Conver to str
#         true_substitution_count.append(true_sub_count)


#     ### PREPROCESS ALL MUTATED PROTS ###

#     # Idea: creata a dataloder to re-use the evaluate_model() fucntion

#     # Create a list of processed data dictionaries for all mutations
#     all_mutated_data = []
    
#     # Preprocess all sequences (this still uses a loop but it's lightweight)
#     for mutated_seq in tqdm(mutated_sequences, desc="Preprocessing mutations"):

#         mutated_data = my_dataset.preprocess_sequence(
#             sequence=mutated_seq,
#             label=single_protein_info['label'], # Use the original label for all
#             protein_name=single_protein_info['protein'],
#             tokenizer=tokenizer,
#             protein_max_length=single_protein_info['inputs_ids_length'] # Use the original max length
#         )
#         mutated_data['set'] = single_protein_info['set']
#         all_mutated_data.append(mutated_data)

#     # Create one DataFrame from all processed data
#     mutated_df = pd.DataFrame(all_mutated_data)

#     # Create one DataLoader for ALL mutated sequences
#     # Batch size is set to the total number of mutations (1 batch total)
#     mutated_dl = my_dataset.create_dataloader(
#         mutated_df,
#         set_name=single_protein_info['set'], #  all samples have this set name
#         batch_size=int(trunc_len // 100),         
#         shuffle=False
#     )

#     ### CALCULATE PROBS ALL MUTATED PROTS ###

#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", category=UserWarning)
#         # Assuming 'evaluate_model' is accessible
#         print("Evaluating model on mutated sequences...")
#         mutated_dict = evaluate_model(model, mutated_dl, device, split_name="Batch Mutation Scan", verbose=False)
    
#     # The result is a tensor of probabilities [trunc_len]
#     mutated_probs = mutated_dict["probs_class1"].cpu().numpy()


#     ### CALCULATE DELTAS ###
#     delta_p = mutated_probs - baseline_p

#     if normalise_true_substitution:
#         true_substitution_count_np = np.array(true_substitution_count)
#         delta_p = np.where(true_substitution_count_np > 0, delta_p / true_substitution_count_np, 0.0)

#     return {
#         'baseline_prob': baseline_p,
#         'delta_probs': delta_p,
#         'mutated_probs': mutated_probs,
#         'sequence': single_protein_info['trunc_sequence'],
#         'protein_name': single_protein_info['protein']
#     }

# def plot_alanine_scan(delta_p, sequence, sigma=3, threshold=True, figsize=(20, 6), 
#                       highlight_residues=True, top_n=10, show_sequence=True,
#                       style='darkgrid', palette='RdBu_r', protein_name="N/A"):
#     """
#     Plot the Δp values across the protein sequence with optional smoothing,
#     threshold lines, and residue highlighting using Seaborn styling.

#     Args:
#         delta_p: array of Δp values (signed deltas from baseline).
#         sequence: protein sequence string.
#         sigma: Gaussian smoothing width (residues) for smoothing curve.
#         threshold: whether to plot threshold lines (mean ± 2*std).
#         figsize: tuple for figure size.
#         highlight_residues: whether to annotate top important residues.
#         top_n: number of top residues to highlight (both positive and negative).
#         show_sequence: whether to show amino acid letters on x-axis (only for short sequences).
#         style: seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks').
#         palette: color palette for gradient coloring.
#         protein_name: Name/ID of the protein (string, added for robustness).
#     """
#     # Set seaborn style
#     sns.set_style(style)
#     sns.set_context("notebook", font_scale=1.1)
    
#     positions = np.arange(1, len(sequence) + 1)
#     delta_p = np.array(delta_p)
    
#     # Smoothed signal
#     smooth_delta = gaussian_filter1d(delta_p, sigma=sigma)
    
#     # Statistics
#     mu = np.mean(delta_p)
#     std = np.std(delta_p)
#     cutoff = 2 * std
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Create color map based on delta_p values
#     max_abs_delta = np.max(np.abs(delta_p))     # Calculate the maximum absolute deviation from zero for a centered color map
#     norm = plt.Normalize(vmin=-max_abs_delta, vmax=max_abs_delta)
#     colors = plt.cm.RdBu_r(norm(delta_p))
    
#     # Plot bars with gradient coloring
#     bars = ax.bar(positions, delta_p, color=colors, alpha=0.6, 
#                   edgecolor='black', linewidth=0.5, label="Δp per residue")
    
#     # Plot smoothed curve
#     ax.plot(positions, smooth_delta, color="black", linewidth=3, 
#             label=f"Smoothed (σ={sigma})", zorder=5, alpha=0.8)
    
#     # Add threshold lines
#     if threshold:
#         ax.axhline(y=mu + cutoff, color="#2ecc71", linestyle="--", linewidth=2, 
#                    label=f"+2σ = {mu + cutoff:.3f}", alpha=0.8)
#         ax.axhline(y=mu - cutoff, color="#e74c3c", linestyle="--", linewidth=2, 
#                    label=f"-2σ = {mu - cutoff:.3f}", alpha=0.8)
#         ax.axhline(y=mu, color="#95a5a6", linestyle=":", linewidth=1.5, 
#                    label=f"Mean = {mu:.3f}", alpha=0.7)
    
#     # Highlight important residues
#     if highlight_residues and top_n > 0:
#         # Most negative deltas (most important for positive class)
#         neg_indices = np.argsort(delta_p)[:top_n]
#         for idx in neg_indices:
#             if delta_p[idx] < (mu - cutoff):
#                 ax.annotate(f'{sequence[idx]}{idx+1}', 
#                            xy=(positions[idx], delta_p[idx]),
#                            xytext=(0, -20), textcoords='offset points',
#                            ha='center', fontsize=9, color='white', weight='bold',
#                            bbox=dict(boxstyle='round,pad=0.4', 
#                                     facecolor='#e74c3c', 
#                                     edgecolor='darkred',
#                                     alpha=0.9, linewidth=2),
#                            arrowprops=dict(arrowstyle='->', 
#                                          color='darkred', 
#                                          lw=1.5,
#                                          connectionstyle='arc3,rad=0'))
        
#         # Most positive deltas (stabilizing residues)
#         pos_indices = np.argsort(delta_p)[-top_n:]
#         for idx in pos_indices:
#             if delta_p[idx] > (mu + cutoff):
#                 ax.annotate(f'{sequence[idx]}{idx+1}', 
#                            xy=(positions[idx], delta_p[idx]),
#                            xytext=(0, 20), textcoords='offset points',
#                            ha='center', fontsize=9, color='white', weight='bold',
#                            bbox=dict(boxstyle='round,pad=0.4', 
#                                     facecolor='#3498db', 
#                                     edgecolor='darkblue',
#                                     alpha=0.9, linewidth=2),
#                            arrowprops=dict(arrowstyle='->', 
#                                          color='darkblue', 
#                                          lw=1.5,
#                                          connectionstyle='arc3,rad=0'))
    
#     # Labels and styling
#     ax.set_xlabel("Residue Position", fontsize=13, fontweight='bold')
#     ax.set_ylabel("Δp (Change in Probability)", fontsize=13, fontweight='bold')
#     ax.set_title("Alanine Scanning Importance Map\n" + 
#                  "Negative Δp = Critical for function", 
#                  fontsize=15, fontweight='bold', pad=20)
    
#     # Show sequence on x-axis if short enough
#     if show_sequence and len(sequence) <= 50:
#         ax.set_xticks(positions)
#         ax.set_xticklabels(list(sequence), fontsize=9, family='monospace')
#         plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
#     else:
#         ax.set_xlim(0, len(sequence) + 1)
    
#     # Legend
#     ax.legend(loc='upper right', fontsize=10, framealpha=0.95, 
#              edgecolor='black', fancybox=True, shadow=True)
    
#     # Add colorbar
#     sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ax=ax, pad=0.01, aspect=30)
#     cbar.set_label('Δp Value', fontsize=11, fontweight='bold')
    
#     # Add statistics box with seaborn styling
#     stats_text = (f'Statistics:\n'
#                  f'Mean: {mu:.4f}\n'
#                  f'Std: {std:.4f}\n'
#                  f'Min: {np.min(delta_p):.4f}\n'
#                  f'Max: {np.max(delta_p):.4f}\n'
#                  f'Range: {np.max(delta_p) - np.min(delta_p):.4f}')
    
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, 
#                 edgecolor='black', linewidth=1.5)
#     ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
#             fontsize=10, verticalalignment='top', bbox=props,
#             family='monospace')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print detailed summary
#     print(f"\n{'='*70}")
#     print(f"{'ALANINE SCANNING SUMMARY':^70}")
#     print(f"{'='*70}")
#     print(f"\n{'Sequence Information:':<30}")
#     print(f"  {'Length:':<25} {len(sequence)}")
#     print(f"  {'Protein:':<25} {protein_name}")
#     print(f"\n{'Statistical Summary:':<30}")
#     print(f"  {'Mean Δp:':<25} {mu:.4f}")
#     print(f"  {'Std Δp:':<25} {std:.4f}")
#     print(f"  {'Min Δp:':<25} {np.min(delta_p):.4f}")
#     print(f"  {'Max Δp:':<25} {np.max(delta_p):.4f}")
#     print(f"  {'Threshold (+2σ):':<25} {mu + cutoff:.4f}")
#     print(f"  {'Threshold (-2σ):':<25} {mu - cutoff:.4f}")
    
#     print(f"\n{'─'*70}")
#     print(f"{'TOP CRITICAL RESIDUES (Largest Negative Δp)':^70}")
#     print(f"{'─'*70}")
#     print(f"{'Rank':<8}{'Position':<12}{'Residue':<12}{'Δp':<15}{'Status':<20}")
#     print(f"{'─'*70}")
    
#     critical_indices = np.argsort(delta_p)[:top_n]
#     for rank, idx in enumerate(critical_indices, 1):
#         status = "⚠️  Beyond threshold" if delta_p[idx] < (mu - cutoff) else "Within range"
#         print(f"{rank:<8}{idx+1:<12}{sequence[idx]:<12}{delta_p[idx]:<15.4f}{status:<20}")
    
#     print(f"\n{'─'*70}")
#     print(f"{'TOP STABILIZING RESIDUES (Largest Positive Δp)':^70}")
#     print(f"{'─'*70}")
#     print(f"{'Rank':<8}{'Position':<12}{'Residue':<12}{'Δp':<15}{'Status':<20}")
#     print(f"{'─'*70}")
    
#     stabilizing_indices = np.argsort(delta_p)[-top_n:][::-1]
#     for rank, idx in enumerate(stabilizing_indices, 1):
#         status = "✓ Beyond threshold" if delta_p[idx] > (mu + cutoff) else "Within range"
#         print(f"{rank:<8}{idx+1:<12}{sequence[idx]:<12}{delta_p[idx]:<15.4f}{status:<20}")
    
#     print(f"{'='*70}\n")