
import random
import time
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from utils.attention import SecFeature
from utils.utils import eval_print
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
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "ESM2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start = time.time()

# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

# 定义最大序列长度
max_length = 1000


class EsmDeepSec(nn.Module):

    def __init__(self, model):
        super(EsmDeepSec, self).__init__()
        self.model = model  # torch.Size([1, 1022, 1280])

        self.feature_fn = SecFeature(


        )

    def forward(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)

        last_hidden_state = outputs.last_hidden_state

        features = self.feature_fn(last_hidden_state)

        return features


train_loss_history = []
train_acc_history = []
valid_loss_history = []
valid_acc_history = []
test_loss_history = []
test_acc_history = []
def train_and_predict(train_dl, valid_dl, test_dl, lr, num_iter, eval_size, device):
    global train_loss_history
    global train_acc_history
    global valid_loss_history
    global valid_acc_history
    global test_loss_history
    global test_acc_history


    net = EsmDeepSec(model).to(device)

    optimizer = AdamW(
        net.feature_fn.parameters(),
        lr=lr
    )
    loss_fn = nn.CrossEntropyLoss()
    # exp_lr = lr_scheduler.StepLR(optimizer, step_size=700, gamma=0.1)
    best_loss = 10.
    best_acc = 0.
    valid_prob = None
    test_prob = None
    t0 = time.time()
    iter_idx = 1
    net.train()

    while iter_idx <= num_iter:
        # print("iter_idx",iter_idx)
        for seq, label in train_dl:
            net.train()
            seq, label = seq.to(device), label.to(device)
            # print("train模式")
            output = net(seq)
            # print("输出output:",output)
            loss = loss_fn(output, label)
            # print("loss:",loss)
            optimizer.zero_grad(set_to_none=True)  # 将梯度张量设置为None，而不是将其清零
            # print("optimize完成")
            loss.backward()
            # print("loss back完成")
            optimizer.step()
            # exp_lr.step()
            # print("optimizer.step完成")
            iter_idx += 1
            # print("iter_ids+:",iter_idx)
            if iter_idx % eval_size == 0:
                net.eval()
                # print("eval模式")
                with torch.no_grad():
                    # print("eval冻结参数完成")
                    # evaluate on train dataset
                    train_loss = torch.zeros(1, dtype=torch.float16, device=device)
                    train_acc = torch.zeros(1, dtype=torch.float16, device=device)
                    for [seq, label] in train_dl:
                        seq, label = seq.to(device), label.to(device)
                        output = net(seq)
                        # print("eval_output:",output)
                        probs = torch.softmax(output, dim=-1)
                        # print("eval_probs:",probs)
                        predict = torch.argmax(probs, dim=-1)
                        # print("eval_predict:", predict)
                        loss = loss_fn(output, label)
                        # print("eval_loss:", loss)
                        train_loss += loss

                        correct = torch.eq(predict, label)
                        train_acc += correct.sum()
                    train_loss /= len(train_dl)
                    train_acc /= len(train_dl.dataset)
                    train_loss = train_loss.item()  # 将平均训练损失（tensor类型）转换为Python标量。
                    # print("train loss",train_loss)
                    train_acc = train_acc.item()
                    # print("train_acc",train_acc)
                    train_loss_history.append(train_loss)
                    train_acc_history.append(train_acc)

                    # evaluate on test dataset
                    valid_loss = torch.zeros(1, dtype=torch.float16, device=device)
                    valid_acc = torch.zeros(1, dtype=torch.float16, device=device)
                    tmp_prob = []
                    valid_label = []

                    for [seq, label] in valid_dl:
                        seq, label = seq.to(device), label.to(device)

                        output = net(seq)

                        probs = torch.softmax(output, dim=-1)

                        # print("valid probs:",probs)
                        predict = torch.argmax(probs, dim=-1)
                        # print("valid=predict",predict)
                        loss = loss_fn(output, label)

                        valid_loss += loss
                        correct = torch.eq(predict, label)

                        valid_acc += correct.sum()
                        # print('predict', predict)
                        # print('label',label)
                        # print('num', correct.sum())

                        tmp_prob.append(probs[:, 1])
                        valid_label.append(label)
                    valid_loss /= len(valid_dl)
                    # print("len valid_dl",len(valid_dl))
                    valid_acc /= len(valid_dl.dataset)

                    tmp_prob = torch.cat(tmp_prob, dim=0).cpu()
                    # valid_label= torch.cat(label, dim=0).cpu()
                    valid_label = torch.cat(valid_label, dim=0).cpu()

                    valid_loss = valid_loss.item()
                    # print("valid loss",valid_loss)
                    valid_acc = valid_acc.item()
                    # eval_print(valid_dl.dataset.tensors[1], valid_predict, valid_prob, '{:s}-valid'.format(args.fluid))
                    # print("valid acc",valid_acc)
                    valid_acc_history.append(valid_acc)
                    valid_loss_history.append(valid_loss)

                    if valid_acc > best_acc:
                        best_acc = valid_acc
                        valid_prob = tmp_prob  # 记录当前的验证集正类预测概率
                        va_label = valid_label

                        test_prob_list = []
                        test_label_list = []


                        test_loss = torch.zeros(1, dtype=torch.float16, device=device)
                        test_acc = torch.zeros(1, dtype=torch.float16, device=device)

                        for [seq, label] in test_dl:
                            seq, label = seq.to(device), label.to(device)
                            output = net(seq)
                            probs = torch.softmax(output, dim=-1)[:, 1]
                            test_prob_list.append(probs.cpu())
                            test_label_list.append(label.cpu())
                          
                            probs1 = torch.softmax(output, dim=-1)
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


                t = time.time() - t0
                t0 = time.time()
                net.train()
                print('[iter {:05d} {:.0f}s] train loss({:.4f}) acc({:.4f}); valid loss({:.4f}), acc({:.4f})'
                      .format(iter_idx, t, train_loss, train_acc, valid_loss, valid_acc))

                test_prob = torch.cat(test_prob_list, dim=0).cpu()
                test_label = torch.cat(test_label_list, dim=0).cpu()
                valid_predict = (valid_prob.numpy() > 0.5).astype(np.int16)
                test_predict = (test_prob.numpy() > 0.5).astype(np.int16)

                eval_print(va_label, valid_predict, valid_prob, '{:s}-valid'.format(args.fluid))
                eval_print(test_label, test_predict, test_prob, '{:s}-test'.format(args.fluid))

                if torch.cuda.is_available():
                    # 创建一个小张量并将其移到 GPU
                    tensor = torch.randn(1).cuda()

                    # 获取当前 GPU 的显存使用情况
                    allocated_memory = torch.cuda.memory_allocated()
                    cached_memory = torch.cuda.memory_cached()

                    print(f"Allocated memory: {allocated_memory / 1024 ** 3:.2f} GB")
                    print(f"Cached memory: {cached_memory / 1024 ** 3:.2f} GB")
                else:
                    print("No GPU available.")


    torch.cuda.empty_cache() 


    return valid_prob, test_prob, va_label, test_label,valid_predict,test_predict



def truncate_sequence(sequence, max_length=1000):
    
    if len(sequence) <= max_length:
        return sequence

   
    else:
        half_length = (max_length) // 2 
        truncated_sequence = sequence[:half_length] + sequence[-half_length:]

    return truncated_sequence


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True 
    train_data = pd.read_csv(/data/Seminal_tr.csv')
    valid_data = pd.read_csv(/data/Seminal_va.csv')
    test_data = pd.read_csv(/data/Seminal_te.csv')

    train = train_data['sequence'].tolist()
  
    train_labels = train_data['label'].tolist()

    valid = valid_data['sequence'].tolist()
    valid_labels = valid_data['label'].tolist()
    test = test_data['sequence'].tolist()
    test_labels = test_data['label'].tolist()

    train1 = [truncate_sequence(seq) for seq in train] 
    valid1 = [truncate_sequence(seq) for seq in valid] 
    test1 = [truncate_sequence(seq) for seq in test]  

   
    tokenized_train = tokenizer(train1, padding='max_length', max_length=max_length, truncation=True,
                                return_tensors="pt")
    tokenized_valid = tokenizer(valid1, padding='max_length', max_length=max_length, truncation=True,
                                return_tensors="pt")
    tokenized_test = tokenizer(test1, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")
 
    labels_tensor_train = torch.tensor(train_labels)
    labels_tensor_valid = torch.tensor(valid_labels)
    labels_tensor_test = torch.tensor(test_labels)
  
    train_dataset = TensorDataset(tokenized_train["input_ids"], labels_tensor_train)
    valid_dataset = TensorDataset(tokenized_valid["input_ids"], labels_tensor_valid)
    test_dataset = TensorDataset(tokenized_test["input_ids"], labels_tensor_test)

    train_dl = DataLoader(train_dataset, args.bs, shuffle=True, pin_memory=True)
    valid_dl = DataLoader(valid_dataset, args.bs, shuffle=True, pin_memory=True)
    test_dl = DataLoader(test_dataset, args.bs, shuffle=True, pin_memory=True)
  

    valid_prob, test_prob, va_label, test_label,valid_predict,test_predict= train_and_predict(
        train_dl,
        valid_dl,
        test_dl,
        args.lr,
        args.num_iter,
        args.eval_size,
        device,

    )


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
   
    eval_print(valid_label, valid_predict, valid_prob, '{:s}-valid'.format(args.fluid))
    eval_print(test_label, test_predict, test_prob, '{:s}-test '.format(args.fluid))


if __name__ == '__main__':
    import argparse  


    parser = argparse.ArgumentParser()

    parser.add_argument('--fluid', default='Seminal', type=str)
    parser.add_argument('--num-iter', default=2960, type=int)  
    parser.add_argument('--bs', default=32, type=int)  

    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--eval-size', default=148, type=int) 
    parser.add_argument('--seed', default=43215, type=int)

    args = parser.parse_args() 
    main(args)
