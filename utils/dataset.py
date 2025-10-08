from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

class ProteinDataset(Dataset):
    def __init__(self, names, labels, input_ids=None, attention_mask=None, embs=None):

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.names = names  # can be strings
        self.embs = embs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # Must take into account None fields (None if the attribute is None)
        input_id_item = self.input_ids[idx] if self.input_ids is not None else None
        attn_mask_item = self.attention_mask[idx] if self.attention_mask is not None else None
        emb_item = self.embs[idx] if self.embs is not None else None

        return (
            input_id_item,
            attn_mask_item,
            self.labels[idx],
            self.names[idx],
            emb_item
        )
    
def create_dataloader(processed_df, set_name, batch_size, shuffle=False, pin_memory=True):
    """
    Create a DataLoader from processed DataFrame for a specific set.
    
    Args:
        processed_df: DataFrame with preprocessed sequences
        set_name: 'train', 'validation', or 'test'
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    # Filter by set
    set_idx = processed_df['set'] == set_name
    
    # Stack tensors
    input_ids = torch.stack([t.long() for t in processed_df.loc[set_idx, 'input_ids'].tolist()]) 
    attention_mask = torch.stack([t.long() for t in processed_df.loc[set_idx, 'attention_mask'].tolist()])
    labels = torch.tensor(processed_df.loc[set_idx, 'label'].values, dtype=torch.long)
    names = processed_df.loc[set_idx, 'protein'].tolist()
    embs = torch.stack([t.long() for t in processed_df.loc[set_idx, 'embedding'].tolist()]) 
    
    # Create dataset
    dataset = ProteinDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        names=names,
        embs=embs
    )
    
    # Create and return DataLoader
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        pin_memory=pin_memory
    )


def truncate_sequence(sequence, max_length=1000):
    # If sequence > 1000 residues: take first 500 + last 500 residues
    if len(sequence) <= max_length:
        return sequence
    else:
        half_length = (max_length) // 2 
        truncated_sequence = sequence[:half_length] + sequence[-half_length:]

    return truncated_sequence


def preprocess_sequence(sequence, label, protein_name, tokenizer, protein_max_length=1000, config=None):
    """
    Preprocess a single protein sequence.
    
    Args:
        sequence: Raw protein sequence string
        label: Classification label
        protein_name: Name/ID of the protein
        tokenizer: Tokenizer instance
        config: Configuration dictionary
        
    Returns:
        Dictionary containing all preprocessed data for one sequence
    """
    # Truncate sequence
    trunc_sequence = truncate_sequence(sequence, max_length=protein_max_length)
    
    # Tokenize single sequence
    tokenized = tokenizer(
        trunc_sequence,
        padding='max_length',
        max_length=protein_max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    # Extract tensors (squeeze to remove batch dimension)
    input_ids = tokenized['input_ids'].squeeze(0)
    attention_mask = tokenized['attention_mask'].squeeze(0)
    
    # Calculate lengths
    sequence_length = len(sequence)
    trunc_sequence_length = len(trunc_sequence)
    inputs_ids_length = len(input_ids)
    inputs_ids_length_no_pad = (input_ids != tokenizer.pad_token_id).sum().item()
    
    return {
        'sequence': sequence,
        'trunc_sequence': trunc_sequence,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': label,
        'protein': protein_name,
        'sequence_length': sequence_length,
        'trunc_sequence_length': trunc_sequence_length,
        'inputs_ids_length': inputs_ids_length,
        'inputs_ids_length_no_pad': inputs_ids_length_no_pad
    }


