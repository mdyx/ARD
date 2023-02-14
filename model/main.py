import os
import time
import torch
import argparse
import numpy as np
from model import ARDSAS
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--log_dir', required=True)
parser.add_argument('--device', default='cpu', type=str)

parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--dropout_rate', default=0.6, type=float)
parser.add_argument('--ard_dr', default=0.3, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--lr_decay', default=1., type=float)

def lr_decay(optim, epoch, lr, lr_decay):
    lr = lr * (lr_decay ** (epoch // 10))
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def main():
    args = parser.parse_args()
    dir_road = '../log/' + args.dataset
    log_road = dir_road + '/' + args.log_dir + '.txt'
    if not os.path.isdir(dir_road):
        os.makedirs(dir_road)
    
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    
    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    # print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(log_road, 'w')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = ARDSAS(usernum, itemnum, args).to(args.device)
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        epoch_loss = 0.
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits, ind_loss, cent_loss = model(u, seq, pos, neg, epoch)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            
            lr_decay(adam_optimizer, epoch, args.lr, args.lr_decay)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            epoch_loss += loss.item()
            loss += 0.5 * ind_loss
            loss += 0.5 * cent_loss
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            
            loss.backward()
            adam_optimizer.step()
        
        epoch_loss /= num_batch
        
        print("[{}/{}] loss = {:.4f}".format(epoch, args.num_epochs, epoch_loss))
        
        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            # print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('epoch {}: valid (NDCG@10: {:.4f}, Recall@10: {:.4f}), test (NDCG@10: {:.4f}, Recall@10: {:.4f})'.format(epoch, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            
            f.write('epoch {}: loss: {:.4f}, valid (NDCG@10: {:.4f}, Recall@10: {:.4f}), test (NDCG@10: {:.4f}, Recall@10: {:.4f})\n'.format(epoch, epoch_loss, t_valid[0], t_valid[1], t_test[0], t_test[1]))
            f.flush()
            
            t0 = time.time()
            model.train()
    
    f.close()
    sampler.close()
    print("Done")

if __name__ == '__main__':
    main()
