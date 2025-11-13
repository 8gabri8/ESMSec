import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch
from scipy.ndimage import gaussian_filter1d
import re


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



import re
from tqdm import tqdm

def _split_into_tokens(seq):
    """
    Split a sequence into tokens:
      - Keeps <...> tokens whole (e.g., <mask>, <pad>)
      - Single amino acids (A, R, N, ...) are single tokens
    """
    tokens = []
    i = 0
    while i < len(seq):
        if seq[i] == '<':
            j = seq.find('>', i+1)
            if j == -1:
                tokens.extend(list(seq[i:]))
                break
            tokens.append(seq[i:j+1])
            i = j + 1
        else:
            tokens.append(seq[i])
            i += 1
    return tokens

def _join_tokens(tokens):
    """Join tokens back into a string."""
    return "".join(tokens)

def create_all_mutations(truncated_seq, window_size=3, substitute_aas=["A", "R", "E", "F", "<mask>"]):
    """
    Token-safe mutation generator.

    Args:
        truncated_seq: str, protein sequence (can include special tokens like <mask>)
        window_size: int, odd number for symmetric window
        substitute_aas: list of AA strings or special tokens ("<mask>", etc.)

    Returns:
        mutated_sequences: list of mutated sequences
        names: list of mutation names ("pos_A", "pos_<mask>", etc.)
    """
    assert window_size % 2 == 1, "window_size must be odd"

    opposite_map = { #àcharge, size, and hydrophobicity, high Grantham distance (>150) for nearly all replacements.
        "A": "F",  # small → bulky aromatic
        "R": "F",  # positive, polar → hydrophobic bulky
        "N": "F",  # polar → hydrophobic
        "D": "F",  # acidic → hydrophobic
        "C": "R",  # small, sulfur → charged
        "Q": "F",  # polar → hydrophobic
        "E": "F",  # acidic → hydrophobic
        "G": "F",  # smallest → bulky
        "H": "F",  # aromatic basic → nonaromatic hydrophobic
        "I": "D",  # hydrophobic → charged acidic
        "L": "D",  # hydrophobic → acidic
        "K": "F",  # positive → hydrophobic
        "M": "D",  # sulfur hydrophobic → acidic
        "F": "D",  # aromatic → acidic
        "P": "W",  # rigid → bulky aromatic
        "S": "F",  # small polar → hydrophobic
        "T": "F",  # polar → hydrophobic
        "W": "D",  # aromatic → acidic
        "Y": "D",  # aromatic polar → acidic
        "V": "D"   # hydrophobic → acidic
    }

    tokens = _split_into_tokens(truncated_seq)
    seq_len = len(tokens)
    half_window = window_size // 2

    mutated_sequences = []
    names = []

    for pos in tqdm(range(seq_len), desc="Generating mutations"):
        window_start = max(0, pos - half_window)
        window_end = min(seq_len, pos + half_window + 1)

        if "inverse" in substitute_aas:
            mut_tokens = tokens.copy()
            for idx in range(window_start, window_end):
                aa = mut_tokens[idx]
                if opposite_map and aa in opposite_map:
                    mut_tokens[idx] = opposite_map[aa]
            mutated_sequences.append(_join_tokens(mut_tokens))
            names.append(f"{pos}_inverse")
        else:
            for sub_aa in substitute_aas:
                mut_tokens = tokens.copy()

                # Mutate tokens in the window
                for idx in range(window_start, window_end):
                    mut_tokens[idx] = sub_aa

                mutated_sequences.append(_join_tokens(mut_tokens))
                names.append(f"{pos}_{sub_aa}")

    return mutated_sequences, names



def multi_aa_scanning_tmp(
                        model,
                        baseline_prob,
                        mutations_dl,
                        names_mutations, 
                        substitute_aas,
                        wt_seq,
                        prot_name,
                        device="cuda",
):

    model.eval()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mutated_eval = mf.evaluate_model(
            net=model,
            dl=mutations_dl,
            device=device,
            loss_fn=model.loss_fn,
            split_name="mutations",
            verbose=False,
            from_precomputed_embs=True #ATTENTION
        )

    mutated_probs_linear = np.array(mutated_eval["probs_class1"])  # shape (n_mutations,)
    delta_probs_linear = mutated_probs_linear - baseline_prob

    # prepare containers
    per_aa_results = {aa: {"delta_probs": [], "mutated_probs": []} for aa in substitute_aas}

    # fill results per AA
    for d_prob, mut_name in zip(delta_probs_linear, names_mutations):
        parts = mut_name.split("_")
        sub_aa = parts[1]
        if sub_aa in per_aa_results:
            per_aa_results[sub_aa]["delta_probs"].append(float(d_prob))
            per_aa_results[sub_aa]["mutated_probs"].append(float(d_prob + baseline_prob))

    # cauclte mean across sub_aa
    to_calc = []
    for sub_aa in substitute_aas:
        to_calc.append(per_aa_results[sub_aa]["delta_probs"])
    delta_probs_array = np.array(to_calc)
    delta_probs_mean = np.mean(delta_probs_array, axis=0)
        
    out = {# serialisable types
        "per_aa_results": per_aa_results,
        "baseline_prob": float(baseline_prob), #scalar
        "delta_probs_mean": delta_probs_mean.tolist(), #vector
        "sequence": str(wt_seq),
        "protein_name": str(prot_name),
        "substitute_aas": list(substitute_aas),
        "mutated_probs_linear": mutated_probs_linear.tolist(),
        "delta_probs_linear": delta_probs_linear.tolist(),
    }

    return out


def plot_multi_aa_scan(scan_results,
                              sigma=3,
                              show_per_aa=True,
                              xlim=None,
                              figsize=(30, 6),
                              palette='Tab10', 
                              ax=None, 
                              sub_aa_for_mean=None
                              ):
    """
    Plot mutation scanning results.
    
    Args:
        scan_results (dict): Output from multi_aa_scanning_tmp()
        sigma (float): Gaussian smoothing sigma for mean line
        show_per_aa (bool): Show individual AA substitution lines
        xlim (tuple): Optional (xmin, xmax) to zoom on x-axis
        figsize (tuple): Figure size
        palette (str): Color palette name
    """
    
    # Extract data
    per_aa_results = scan_results["per_aa_results"]
    substitute_aas = scan_results["substitute_aas"]
    sequence = scan_results["sequence"]
    protein_name = scan_results["protein_name"]
    delta_probs_mean = scan_results["delta_probs_mean"]

    # recalcuate mean wiht a subset of sub_aa
    if sub_aa_for_mean is not None:
        tmp = []
        for aa in sub_aa_for_mean:
            tmp.append(per_aa_results[aa]["delta_probs"])
        delta_probs_mean = np.array(tmp).mean(axis=0)
    
    # Positions start from 1
    positions = np.arange(1, len(delta_probs_mean) + 1)
    
    # Calculate statistics
    std = np.std(delta_probs_mean)
    mean = np.mean(delta_probs_mean)
    threshold_upper = 2 * std
    threshold_lower = -2 * std
    
    # Smooth mean
    smooth_mean = gaussian_filter1d(delta_probs_mean, sigma=sigma)
    
    # Find outlier positions (using smoothed mean)
    outlier_mask = (smooth_mean > threshold_upper) | (smooth_mean < threshold_lower)
    outlier_positions = positions[outlier_mask]
    outlier_values = smooth_mean[outlier_mask]
    
    # Only create a new figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # use parent figure    
    
    # Color palette
    colors = sns.color_palette("tab10", len(substitute_aas))
    
    # Plot individual AA lines
    if show_per_aa:
        for sub_aa, color in zip(substitute_aas, colors):
            ax.plot(positions, per_aa_results[sub_aa]["delta_probs"], 
                   color=color, alpha=1, linewidth=2.5, label=f'→{sub_aa}')
    
    # Plot mean as bars (thin, semi-transparent)
    ax.bar(positions, delta_probs_mean, 
           color='gray', alpha=0.3, width=1.0, 
           edgecolor='none', label='Mean Δp (raw)', zorder=2)
    
    # Plot smoothed mean line (thick, prominent)
    ax.plot(positions, smooth_mean, 
           color='black', linewidth=3, 
           label=f'Mean Δp (smoothed, σ={sigma})', zorder=5)
    
    # Zero line
    ax.axhline(0, color='darkgray', linestyle='-', linewidth=1, alpha=0.7, zorder=1)
    
    # Threshold lines
    ax.axhline(threshold_upper, color='red', linestyle='--', 
              linewidth=2, alpha=0.8, label=f'+2σ = {threshold_upper:.3f}', zorder=3)
    ax.axhline(threshold_lower, color='blue', linestyle='--', 
              linewidth=2, alpha=0.8, label=f'-2σ = {threshold_lower:.3f}', zorder=3)
    
    # Highlight outlier positions
    if len(outlier_positions) > 0:
        ax.scatter(outlier_positions, outlier_values, 
                  color='red', s=120, marker='o', 
                  edgecolors='darkred', linewidths=2.5,
                  zorder=10, label=f'Outliers (n={len(outlier_positions)})')
        
        # Annotate outlier positions (if not too many)
        if len(outlier_positions) <= 20:
            for pos, val in zip(outlier_positions, outlier_values):
                aa_name = sequence[pos - 1]  # positions start at 1, but sequence is 0-indexed
                ax.annotate(f'{aa_name}{int(pos)}', 
                           xy=(pos, val), 
                           xytext=(0, 10 if val > 0 else -15),
                           textcoords='offset points',
                           ha='center', 
                           fontsize=9,
                           color='darkred',
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='yellow', 
                                   alpha=0.7, 
                                   edgecolor='darkred'))
    
    # Set y-limits
    ax.set_ylim(-3 * std, 3 * std)
    
    # Set x-limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(1, len(scan_results["delta_probs_mean"]))
    
    # Labels and title
    ax.set_xlabel('Position (Amino Acid)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Δp (mutated - baseline)', fontsize=14, fontweight='bold')
    ax.set_title(f'{protein_name} - Multi-AA Mutation Scan - Sub aa for mean {sub_aa_for_mean if sub_aa_for_mean is not None else "all"}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add secondary x-axis with AA sequence
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    
    # Determine step size for AA labels based on sequence length and xlim
    if xlim is not None:
        visible_start = max(1, int(xlim[0]))
        visible_end = min(len(sequence), int(xlim[1]))
        visible_length = visible_end - visible_start + 1
    else:
        visible_start = 1
        visible_end = len(sequence)
        visible_length = len(sequence)
    
    # Show every nth AA to avoid overcrowding
    if visible_length <= 50:
        step = 1
    elif visible_length <= 100:
        step = 2
    elif visible_length <= 200:
        step = 5
    else:
        step = max(10, visible_length // 50)
    
    tick_positions = np.arange(visible_start, visible_end + 1, step)
    tick_labels = [sequence[pos - 1] for pos in tick_positions]
    
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=9, family='monospace')
    ax2.set_xlabel('Wild-type AA', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', which='major', length=5)
    
    # Legend
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95, 
             edgecolor='black', fancybox=True, shadow=True)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Improve tick labels
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Add subtle background color
    ax.set_facecolor('#f9f9f9')
    
    plt.tight_layout()

    if ax is None:
        plt.show()
    else:
        # Don't show or close anything if plotting into an external axis
        pass
    
    if ax is None:
        #Print summary
        print(f"{'='*60}")
        print(f"Outlier positions (|Δp| > 2σ):")
        print(f"{'='*60}")
        if len(outlier_positions) > 0:
            for pos in outlier_positions:
                wt_aa = sequence[pos - 1]
                delta_val = smooth_mean[pos - 1]
                print(f"  Position {pos:3d} ({wt_aa}): Δp = {delta_val:+.4f}")
        else:
            print("  No outliers found.")
        print(f"{'='*60}\n")

    return fig, ax

