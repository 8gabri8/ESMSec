from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import numpy as np

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


# class ProteinDataset(Dataset):
#     def __init__(self, names, labels, input_ids=None, attention_mask=None, embs=None):
#         """
#         Generic dataset. Assumes that all data passed here is already filtered for the split.
#         """
#         self.names = names
#         self.labels = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels
#         self.input_ids = input_ids
#         self.attention_mask = attention_mask
#         self.embs = embs

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         input_id_item = self.input_ids[idx] if self.input_ids is not None else None
#         attn_mask_item = self.attention_mask[idx] if self.attention_mask is not None else None
#         emb_item = self.embs[idx] if self.embs is not None else None
#         label = self.labels[idx]
#         name = self.names[idx]

#         return input_id_item, attn_mask_item, label, name, emb_item
    
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


    
# def create_dataloader(cache_data, split_name, batch_size, shuffle=False):

#     # Get indices for the split
#     indices = [i for i, s in enumerate(cache_data["set"]) if s == split_name]

#     # Slice data for this split
#     names = [cache_data["protein"][i] for i in indices]
#     labels = cache_data["label"][indices] if isinstance(cache_data["label"], torch.Tensor) else torch.tensor([cache_data["label"][i] for i in indices], dtype=torch.long)

#     input_ids = cache_data.get("input_ids", None)
#     attention_mask = cache_data.get("attention_mask", None)
#     embs = cache_data.get("embedding", None)

#     if isinstance(input_ids, torch.Tensor):
#         input_ids = input_ids[indices]
#     if isinstance(attention_mask, torch.Tensor):
#         attention_mask = attention_mask[indices]
#     if isinstance(embs, torch.Tensor):
#         embs = embs[indices]

#     # Create dataset
#     dataset = ProteinDataset(
#         names=names,
#         labels=labels,
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         embs=embs
#     )

#     # Wrap in DataLoader
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         drop_last=False,
#         num_workers=4,
#         pin_memory=True
#     )


def truncate_sequence(sequence, max_length=1000):
    # If sequence > 1000 residues: take first 500 + last 500 residues
    if len(sequence) <= max_length:
        return sequence
    else:
        half_length = (max_length) // 2 
        truncated_sequence = sequence[:half_length] + sequence[-half_length:]

    return truncated_sequence


# def preprocess_sequence(sequence, label, protein_name, tokenizer, protein_max_length=1000, config=None):
#     """
#     Preprocess a single protein sequence.
    
#     Args:
#         sequence: Raw protein sequence string
#         label: Classification label
#         protein_name: Name/ID of the protein
#         tokenizer: Tokenizer instance
#         config: Configuration dictionary
        
#     Returns:
#         Dictionary containing all preprocessed data for one sequence
#     """
#     # Truncate sequence
#     trunc_sequence = truncate_sequence(sequence, max_length=protein_max_length)
    
#     # Tokenize single sequence
#     tokenized = tokenizer(
#         trunc_sequence,
#         padding='max_length',
#         max_length=protein_max_length,
#         truncation=True,
#         return_tensors="pt"
#     )
    
#     # Extract tensors (squeeze to remove batch dimension)
#     input_ids = tokenized['input_ids'].squeeze(0)
#     attention_mask = tokenized['attention_mask'].squeeze(0)
    
#     # Calculate lengths
#     sequence_length = len(sequence)
#     trunc_sequence_length = len(trunc_sequence)
#     inputs_ids_length = len(input_ids)
#     inputs_ids_length_no_pad = (input_ids != tokenizer.pad_token_id).sum().item()
    
#     return {
#         'sequence': sequence,
#         'trunc_sequence': trunc_sequence,
#         'input_ids': input_ids,
#         'attention_mask': attention_mask,
#         'label': label,
#         'protein': protein_name,
#         'sequence_length': sequence_length,
#         'trunc_sequence_length': trunc_sequence_length,
#         'inputs_ids_length': inputs_ids_length,
#         'inputs_ids_length_no_pad': inputs_ids_length_no_pad
#     }


