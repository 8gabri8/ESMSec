
from torch.optim import AdamW
from torch import nn
from torch.optim import lr_scheduler


def truncate_sequence(sequence, max_length=1000):
    # If sequence > 1000 residues: take first 500 + last 500 residues
    if len(sequence) <= max_length:
        return sequence
    else:
        half_length = (max_length) // 2 
        truncated_sequence = sequence[:half_length] + sequence[-half_length:]

    return truncated_sequence


def train(net, train_dl, valid_dl, test_dl, config):

    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []
    test_loss_history = []
    test_acc_history = []

    optimizer = AdamW(net.feature_fn.parameters(), lr=config["LR"])
    exp_lr = lr_scheduler.StepLR(optimizer, step_size=config["LR_DECAY_STEPS"], gamma=config["LR_DECAY_GAMMA"])
    loss_fn = nn.CrossEntropyLoss()




    pass