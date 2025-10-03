
import random
import time
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from model import SecFeature
#from utils.utils import eval_print
import numpy as np
import pandas as pd
from torch.optim import Adam, lr_scheduler
from sklearn.linear_model import LogisticRegression
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers import pipeline
from datasets import Dataset, DatasetDict
# 加载模型和 tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

start = time.time()


# Already trained model
checkpoint =  "facebook/esm2_t6_8M_UR50D" # Original "ESM2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint, cache_dir="/home/gdallagl/myworkdir/data/esm2-models")
print("ESM model type", type(model))
# max sequence length for ESM2 is 1024, here we use 1000 to be safe
max_length = 1000


# Define Model (ESM+ClassificatioHead)
class EsmDeepSec(nn.Module):

    def __init__(self, model):
        super(EsmDeepSec, self).__init__()

        # ESM base
        self.model = model  

        ESM_hidden_dim = self.model.config.hidden_size
        print("\nESM hidden dim", ESM_hidden_dim, "\n")

        # Classification Head
        self.feature_fn = SecFeature(in_features=ESM_hidden_dim)


    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad(): # Freeze ESM2 parameters
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        last_hidden_state = outputs.last_hidden_state # Shape: [batch, seq_len (with special tokens), hidden_dim]

        features = self.feature_fn(last_hidden_state)

        return features


train_loss_history = []
train_acc_history = []
valid_loss_history = []
valid_acc_history = []
test_loss_history = []
test_acc_history = []

def train_and_predict(train_dl, valid_dl, test_dl, lr, num_iter, eval_size, device, args):
    # dl --> dataloader
    # num_iter --> iteraztions, ie One forward and backward pass through one batch
    # eval_size --> evaluation frequency (in iterations)

    global train_loss_history
    global train_acc_history
    global valid_loss_history
    global valid_acc_history
    global test_loss_history
    global test_acc_history

    # Initialise model, optimizer and loss function
    net = EsmDeepSec(model).to(device)
    optimizer = AdamW(net.feature_fn.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for p in net.model.parameters():
        p.requires_grad = False

    # exp_lr = lr_scheduler.StepLR(optimizer, step_size=700, gamma=0.1)
    best_loss = 10.
    best_acc = 0. # used for model selection
    valid_prob = None
    test_prob = None
    t0 = time.time()
    iter_idx = 1
    net.train()

    while iter_idx <= num_iter:

        print("Iteration number: ", iter_idx)

        # print("iter_idx",iter_idx)
        for seq, label in train_dl: # take one BATCH

            # set model to trainign mode
            net.train()
            #move to device
            seq, label = seq.to(device), label.to(device)
            # Forward apss
            output = net(seq) # seq --> [batch_size, sequence_length], TOKENISEd protein sequences
            # compute loss
            loss = loss_fn(output, label) # crossEntripyLoss() expects RAW logits, so it is correct passing direclty "outpu"
            # clear previous gradients
            optimizer.zero_grad(set_to_none=True)  # 将梯度张量设置为None，而不是将其清零
            # compute gradients (backpropgation)
            loss.backward()
            # paramers update
            optimizer.step()

            iter_idx += 1

            # EVALUATION
            if iter_idx % eval_size == 0:

                # change mode to eval (avoid dropouts layers, ..)
                net.eval()

                with torch.no_grad(): # avpid computing grads

                    print("Eavluationf on Validation Set")

                    ### TRAINING SET EVALUATION ###
                    train_loss = torch.zeros(1, dtype=torch.float16, device=device)
                    train_acc = torch.zeros(1, dtype=torch.float16, device=device)

                    for [seq, attention_mask, label] in train_dl:
                        seq, attention_mask, label = seq.to(device), attention_mask.to(device), label.to(device)
                        output = net(seq, attention_mask=attention_mask)
                        probs = torch.softmax(output, dim=-1) # coonvert to probs
                        predict = torch.argmax(probs, dim=-1) # take label (is the index of max)
                        loss = loss_fn(output, label) # alredy compute SINGLE BATCH AVERAGE

                        train_loss += loss
                        correct = torch.eq(predict, label) # tensor of 0/1
                        train_acc += correct.sum() # count how many correct in this batch

                    train_loss /= len(train_dl)  # Average loss PER BATCH
                    train_acc /= len(train_dl.dataset) # Average accurracy PER SAMPLE (of this set)

                    train_loss = train_loss.item()  # Convert tensor to Python scalar
                    train_acc = train_acc.item()
                    train_loss_history.append(train_loss) # Save in lists
                    train_acc_history.append(train_acc)


                    ### VALIDATION SET EVALUATION ###
                    valid_loss = torch.zeros(1, dtype=torch.float16, device=device)
                    valid_acc = torch.zeros(1, dtype=torch.float16, device=device)
                    tmp_prob = [] # predicted probabilities for the ONLY POSTIVE class (ie class with label 1)
                    valid_label = [] # true labels

                    for [seq, attention_mask, label] in valid_dl:

                        seq, attention_mask, label = seq.to(device), attention_mask.to(device), label.to(device)
                        output = net(seq, attention_mask=attention_mask)
                        probs = torch.softmax(output, dim=-1)
                        predict = torch.argmax(probs, dim=-1)
                        loss = loss_fn(output, label)

                        valid_loss += loss
                        correct = torch.eq(predict, label)
                        valid_acc += correct.sum()

                        tmp_prob.append(probs[:, 1]) #prob class1, for later ROC AUC
                        valid_label.append(label)

                    valid_loss /= len(valid_dl)
                    valid_acc /= len(valid_dl.dataset)

                    tmp_prob = torch.cat(tmp_prob, dim=0).cpu()
                    valid_label = torch.cat(valid_label, dim=0).cpu()

                    valid_loss = valid_loss.item()
                    valid_acc = valid_acc.item()

                    valid_acc_history.append(valid_acc)
                    valid_loss_history.append(valid_loss)

                    valid_prob = tmp_prob  #  Save best validation probabilities
                    va_label = valid_label  # Save corresponding labels


                    ### MODEL SELECTION ###
                    ### TEST SET EVALUATION --> only if the best model changes ###
                    if valid_acc > best_acc:

                        print("Evaluation on Test Set")

                        best_acc = valid_acc

                        ### TEST SET EVALUATION ###
                        test_prob_list = []
                        test_label_list = []

                        test_loss = torch.zeros(1, dtype=torch.float16, device=device)
                        test_acc = torch.zeros(1, dtype=torch.float16, device=device)

                        for [seq, attention_mask, label] in test_dl:

                            seq, attention_mask, label = seq.to(device), label.to(attention_mask), label.to(device)
                            output = net(seq, attention_mask=attention_mask)
                            probs = torch.softmax(output, dim=-1)[:, 1] # Only positive class prob
                            test_prob_list.append(probs.cpu())
                            test_label_list.append(label.cpu())
                          
                            probs1 = torch.softmax(output, dim=-1)  # Full probability distribution
                            predict = torch.argmax(probs1, dim=-1)
                            loss = loss_fn(output, label)

                            test_loss += loss
                            correct = torch.eq(predict, label)
                            test_acc += correct.sum()

                        test_loss /= len(test_dl)
                        test_acc /= len(test_dl.dataset)
                        test_loss = test_loss.item()
                        test_acc = test_acc.item()
                        test_acc_history.append(test_acc)
                        test_loss_history.append(test_loss)

                        test_prob = torch.cat(test_prob_list, dim=0).cpu() # Concatenates all batches along the first dimension
                        test_label = torch.cat(test_label_list, dim=0).cpu()
                        test_predict = (test_prob.numpy() > 0.5).astype(np.int16)

                        print(f"[{args.fluid}-test]  labels={test_label}, preds={test_predict}, probs={test_prob}")


                t = time.time() - t0
                t0 = time.time() # reset timer

                # go back to train mode
                net.train()

                # Performace report
                print('[iter {:05d} {:.0f}s] train loss({:.4f}) acc({:.4f}); valid loss({:.4f}), acc({:.4f})'
                      .format(iter_idx, t, train_loss, train_acc, valid_loss, valid_acc))


                valid_predict = (valid_prob.numpy() > 0.5).astype(np.int16) # if p(class1)>0.5 label as 1, else 0


                # printing info
                #eval_print(va_label, valid_predict, valid_prob, '{:s}-valid'.format(args.fluid))
                #eval_print(test_label, test_predict, test_prob, '{:s}-test'.format(args.fluid))
                print(f"[{args.fluid}-valid] labels={va_label}, preds={valid_predict}, probs={valid_prob}")


                ### GPU monitoring
                if torch.cuda.is_available():
                    # 创建一个小张量并将其移到 GPU
                    tensor = torch.randn(1).cuda()

                    # 获取当前 GPU 的显存使用情况
                    allocated_memory = torch.cuda.memory_allocated()
                    cached_memory = torch.cuda.memory_reserved()

                    print(f"Allocated memory: {allocated_memory / 1024 ** 3:.2f} GB")
                    print(f"Cached memory: {cached_memory / 1024 ** 3:.2f} GB")
                else:
                    print("No GPU available.")

    # claean up
    torch.cuda.empty_cache() 

    return valid_prob, test_prob, va_label, test_label,valid_predict,test_predict



def truncate_sequence(sequence, max_length=1000):
    # If sequence > 1000 residues: take first 500 + last 500 residues
    if len(sequence) <= max_length:
        return sequence
    else:
        half_length = (max_length) // 2 
        truncated_sequence = sequence[:half_length] + sequence[-half_length:]

    return truncated_sequence


def main(args):

    # Initializations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True 

    # # Load data
    # train_data = pd.read_csv(/data/Seminal_tr.csv')
    # valid_data = pd.read_csv(/data/Seminal_va.csv')
    # test_data = pd.read_csv(/data/Seminal_te.csv')
    # # Extract sequences and labels
    # train = train_data['sequence'].tolist()
    # train_labels = train_data['label'].tolist()
    # valid = valid_data['sequence'].tolist()
    # valid_labels = valid_data['label'].tolist()
    # test = test_data['sequence'].tolist()
    # test_labels = test_data['label'].tolist()

    # Load the full dataset
    data = pd.read_csv("/home/gdallagl/myworkdir/data/ESMSec/protein/CSF_my_dataset.csv")

    # Split based on the 'set' column
    train_data = data[data['set'] == 'train']
    valid_data = data[data['set'] == 'validation']
    test_data = data[data['set'] == 'test']

    # Extract sequences and labels
    train_sequences = train_data['sequence'].tolist()
    train_labels = train_data['label'].tolist()

    valid_sequences = valid_data['sequence'].tolist()
    valid_labels = valid_data['label'].tolist()

    test_sequences = test_data['sequence'].tolist()
    test_labels = test_data['label'].tolist()

    # Preporcess sequences
    train1 = [truncate_sequence(seq) for seq in train_sequences] 
    valid1 = [truncate_sequence(seq) for seq in valid_sequences] 
    test1 = [truncate_sequence(seq) for seq in test_sequences]  

    # Tokenize sequences
    tokenized_train = tokenizer(train1, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt") # output is a dict
    tokenized_valid = tokenizer(valid1, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")
    tokenized_test = tokenizer(test1, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")
    # Convert labesl to tensors
    labels_tensor_train = torch.tensor(train_labels)
    labels_tensor_valid = torch.tensor(valid_labels)
    labels_tensor_test = torch.tensor(test_labels)
  
    # Create datasets and dataloaders
    train_dataset = TensorDataset(tokenized_train["input_ids"], tokenized_train["attention_mask"], labels_tensor_train)
    valid_dataset = TensorDataset(tokenized_valid["input_ids"], tokenized_valid["attention_mask"], labels_tensor_valid)
    test_dataset = TensorDataset(tokenized_test["input_ids"], tokenized_test["attention_mask"], labels_tensor_test)
    train_dl = DataLoader(train_dataset, args.bs, shuffle=True, pin_memory=True)
    valid_dl = DataLoader(valid_dataset, args.bs, shuffle=True, pin_memory=True)
    test_dl = DataLoader(test_dataset, args.bs, shuffle=True, pin_memory=True)
  

    ### Training loop ###
    valid_prob, test_prob, va_label, test_label,valid_predict,test_predict= train_and_predict(
        train_dl,
        valid_dl,
        test_dl,
        args.lr,
        args.num_iter,
        args.eval_size,
        device,
        args
    )


    ### PLOTTING ###
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_loss_history)), train_loss_history, label='Train Loss')
    plt.plot(range(len(valid_loss_history)), valid_loss_history, label='Valid Loss')
    plt.plot(range(len(test_loss_history)), test_loss_history, label='Test Loss')
    plt.title('Loss History')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_acc_history)), train_acc_history, label='Train Accuracy')
    plt.plot(range(len(valid_acc_history)), valid_acc_history, label='Valid Accuracy')
    plt.plot(range(len(test_acc_history)), test_acc_history, label='Test Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('Seminal.png')

    valid_prob = valid_prob.numpy()
    
    test_prob = test_prob.numpy()
  
    valid_predict = (valid_prob > 0.5).astype(np.int16)
    
    test_predict = (test_prob > 0.5).astype(np.int16)
   
    #eval_print(valid_label, valid_predict, valid_prob, '{:s}-valid'.format(args.fluid))
    #eval_print(test_label, test_predict, test_prob, '{:s}-test '.format(args.fluid))
    print(f"[{args.fluid}-valid] labels={va_label}, preds={valid_predict}, probs={valid_prob}")
    print(f"[{args.fluid}-test]  labels={test_label}, preds={test_predict}, probs={test_prob}")





if __name__ == '__main__':

    import argparse  

    parser = argparse.ArgumentParser()

    parser.add_argument('--fluid', default='CSF', type=str)      # Dataset name
    parser.add_argument('--num-iter', default=500, type=int)       # Total iterations
    parser.add_argument('--bs', default=32, type=int)               # Batch size
    parser.add_argument('--lr', default=5e-5, type=float)           # Learning rate
    parser.add_argument('--eval-size', default=100, type=int)       # Eval frequency
    parser.add_argument('--seed', default=43215, type=int)          # Random seed

    args = parser.parse_args() 
    main(args)
