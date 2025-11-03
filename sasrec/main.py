#%%
import os
import sys
import time
import torch
import argparse
import numpy as np
from datetime import datetime

from model import SASRec
from utils import set_seed, set_device
from evaluate import evaluate, evaluate_valid
from dataset import data_partition, build_index, WarpSampler


#%%
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="ml-1m", type=str)
parser.add_argument('--data-dir', default="../data/sasrec/data", type=str)
parser.add_argument('--weights-dir', default="weights", type=str)
parser.add_argument('--evaluate-interval', default=20, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden-units', default=50, type=int)
parser.add_argument('--num-blocks', default=2, type=int)
parser.add_argument('--num-epochs', default=1000, type=int)
parser.add_argument('--num-heads', default=1, type=int)
parser.add_argument('--dropout-rate', default=0.2, type=float)
parser.add_argument('--l2-emb', default=0.0, type=float)
parser.add_argument('--state-dict-path', default=None, type=str)
parser.add_argument('--norm-first', action='store_true', default=False)
parser.add_argument('--random-seed', default=0, type=int)
parser.add_argument("--device", type=str, default="none")
try:
    args = parser.parse_args()
except: 
    args = parser.parse_args([])

args.expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(args.random_seed)
args.device = set_device(args.device)

folder = args.dataset + '_' + args.weights_dir
os.makedirs(folder, exist_ok=True)
fname = 'sasrec.lr={}.block={}.head={}.unit={}.maxlen={}.batch={}.expt={}.pth'
args.fname = fname.format(args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen, args.batch_size, args.expt_num)


#%%
try:
    import wandb
except: 
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb

wandb_login = False
try:
    wandb_login = wandb.login(key = open(f"{args.data_dir}/wandb_key.txt", 'r').readline())
except:
    pass

if wandb_login:
    configs = vars(args)
    wandb_var = wandb.init(project="seq_rec", config=configs)
    expt_name = f"sasrec_{args.expt_num}"
    wandb.run.name = expt_name


#%% data loading
item_seqs, user_seqs = build_index(args.data_dir, args.dataset)
dataset = data_partition(args.data_dir, args.dataset)
[item_seqs_train, item_seqs_valid, item_seqs_test, num_users, num_items] = dataset

total_batch = (num_users-1) // args.batch_size + 1
total_seq_len = 0.0
for u in item_seqs_train:
    total_seq_len += len(item_seqs_train[u])
print('average sequence length: %.2f' % (total_seq_len / num_users))

sampler = WarpSampler(item_seqs_train, num_users, num_items, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=1)


#%%
model = SASRec(num_users, num_items, args).to(args.device)

for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_normal_(param.data)
    except:
        pass
model.pos_emb.weight.data[0, :] = torch.zeros_like(model.pos_emb.weight.data[0, :])
model.item_emb.weight.data[0, :] = torch.zeros_like(model.item_emb.weight.data[0, :])

bce_criterion = torch.nn.BCEWithLogitsLoss()
adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))


#%%
best_valid_ndcg, best_valid_hr = 0.0, 0.0
best_test_ndcg, best_test_hr = 0.0, 0.0
best_epoch = 0
model.train()
for epoch in range(1, args.num_epochs + 1):

    for step in range(total_batch):
        user_ids, log_seqs, pos_seqs, neg_seqs = sampler.next_batch()
        user_ids, log_seqs, pos_seqs, neg_seqs = np.array(user_ids), np.array(log_seqs), np.array(pos_seqs), np.array(neg_seqs)
        pos_logits, neg_logits = model(user_ids, log_seqs, pos_seqs, neg_seqs)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

        adam_optimizer.zero_grad()
        indices = np.where(pos_seqs != 0)
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])
        for param in model.item_emb.parameters():
            loss += args.l2_emb * torch.norm(param)
        loss.backward()
        adam_optimizer.step()
        print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

    if epoch % args.evaluate_interval == 0:
        model.eval()
        print('Evaluating', end='')
        t_valid = evaluate_valid(model, dataset, args)
        t_test = evaluate(model, dataset, args)
        print(f"epoch:{epoch}, valid (NDCG@10: {t_valid[0]}, HR@10: {t_valid[1]}), test (NDCG@10: {t_test[0]}, HR@10: {t_test[1]})")

        if t_valid[0] > best_valid_ndcg:
            best_valid_ndcg = t_valid[0]
            best_valid_hr = t_valid[1]
            best_test_ndcg = t_test[0]
            best_test_hr = t_test[1]
            best_epoch = epoch

            torch.save(model.state_dict(), os.path.join(folder, args.fname))

        if wandb_login:
            wandb_var.log(
                {
                "valid_ndcg@10": t_valid[0],
                "valid_hr@10": t_valid[1],
                "test_ndcg@10": t_test[0],
                "test_HR@10": t_test[1],
                "best_valid_ndcg@10": best_valid_ndcg,
                "best_valid_HR@10": best_valid_hr,
                "best_test_ndcg@10": best_test_ndcg,
                "best_test_HR@10": best_test_hr,
                "best_epoch": best_epoch,
                }
            )

        model.train()
    
sampler.close()
print("Done")

# %%
