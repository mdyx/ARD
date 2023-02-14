import numpy as np
import torch

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)
    
    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class ARDSAS(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(ARDSAS, self).__init__()
        
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        
        self.temperature = 0.2
        self.hidden_units = args.hidden_units
        self.aspect_layers = torch.nn.ModuleList()
        for _ in range(16):
            self.aspect_layers.append(torch.nn.Linear(args.hidden_units, args.hidden_units, bias = False))
        
        self.weight = torch.nn.Linear(2 * args.hidden_units, 1)
        self.softmax = torch.nn.Softmax(dim = 2)
        self.softmax2 = torch.nn.Softmax(dim = 1)
        self.dropout = torch.nn.Dropout(p = args.ard_dr)
        
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
    
    def log2feats(self, log_seqs, is_train, tar, epoch):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        
        # padding mask
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
        
        ind_loss, cent_loss = 0., 0.
        mask = timeline_mask
        [N, L, E] = seqs.shape
        
        if is_train:
            mask = torch.cat([mask, mask[:, [-1]]], dim = 1)
        
        if is_train:
            kv = torch.cat([seqs, tar[0][:, [-1], :]], dim = 1)
        else:
            kv = seqs
        kv_save = kv.clone()
        
        asp_list = []
        for i in range(16):
            asp_i = self.aspect_layers[i](kv) + kv
            asp_i = asp_i / (asp_i.norm(dim = 2, keepdim = True) + 1e-8)
            
            asp_i = asp_i * ~mask.unsqueeze(-1)
            asp_list.append(asp_i.unsqueeze(2))
        
        temp_list = []
        for i in range(16):
            asp_i = self.aspect_layers[i](kv_save)
            asp_i = asp_i * ~mask.unsqueeze(-1)
            temp_list.append(asp_i.unsqueeze(2))
        
        cent_loss = ((kv_save * ~mask.unsqueeze(-1)) - torch.cat(temp_list, dim = 2).mean(2)).norm(dim = 2).sum() / (~mask).sum()
        
        kv = torch.cat(asp_list, dim = 2)
        
        norm_x = kv / (kv.norm(dim = 3, keepdim = True) + 1e-8)
        
        pos = (norm_x * norm_x).sum(dim = 3)
        ttl = torch.matmul(norm_x, norm_x.transpose(2, 3))
        
        pos = torch.exp(pos / self.temperature)
        ttl = torch.exp(ttl / self.temperature).sum(dim = 3)
        
        mi = -torch.log(pos / ttl)
        
        mi = mi * (~mask.unsqueeze(-1))
        
        mi = mi.sum(dim = 2)
        ind_loss = mi.sum() / (~mask).sum()
        
        q = kv.sum(2).cumsum(dim = 1)
        q = q - kv.sum(2)
        div = 16 * ((~mask).cumsum(dim = 1) - (~mask).long()).unsqueeze(-1) + 1e-8
        q = q / div
        q = q * ~mask.unsqueeze(-1)
        
        q = q.unsqueeze(2)
        weight = self.weight(torch.cat([q.repeat(1, 1, 16, 1), kv], dim = 3)).squeeze(3)
        
        weight = self.softmax(weight)
        
        weight = weight.unsqueeze(3)
        re_all = (kv * weight).sum(2)
        
        re_embedding = re_all[:, :L, :].clone()
        
        re_embedding = re_embedding + kv[:, :L, :, :].clone().mean(dim = 2)
        
        if is_train:
            mask = mask[:, :L]
        
        if is_train:
            neg_kv = tar[1]
            asp_list = []
            for i in range(16):
                asp_i = self.aspect_layers[i](neg_kv) + neg_kv
                asp_i = asp_i / (asp_i.norm(dim = 2, keepdim = True) + 1e-8)
                asp_i = asp_i * ~mask.unsqueeze(-1)
                asp_list.append(asp_i.unsqueeze(2))
            neg_kv = torch.cat(asp_list, dim = 2)
            
            q = q[:, 1:, :, :].clone() * (~mask).unsqueeze(-1).unsqueeze(-1)
            
            weight2 = self.weight(torch.cat([q.repeat(1, 1, 16, 1), neg_kv], dim = 3)).squeeze(3)
            
            weight2 = self.softmax(weight2)
            
            weight2 = weight2.unsqueeze(3)
            neg = (neg_kv * weight2).sum(2)
            
            pos = re_all[:, 1:, :].clone() * (~mask.unsqueeze(-1))
            pos = self.dropout(pos)
            pos = pos + (kv[:, 1:, :, :] * ~mask.unsqueeze(-1).unsqueeze(-1)).clone().mean(2)
            neg = self.dropout(neg)
            neg = neg + neg_kv.mean(2)
            
            tar = [pos, neg]
        else:
            tar_kv = tar
            asp_list = []
            for i in range(16):
                asp_i = self.aspect_layers[i](tar_kv) + tar_kv
                asp_i = asp_i / (asp_i.norm(dim = 1, keepdim = True) + 1e-8)
                asp_list.append(asp_i.unsqueeze(1))
            tar_kv = torch.cat(asp_list, dim = 1)
            
            q = kv.sum((1, 2))
            length = (~mask).sum(1)
            div = (length * 16).unsqueeze(-1)
            q = q / div
            
            q = q.unsqueeze(1)
            weight = self.weight(torch.cat([q.repeat(tar_kv.shape[0], 16, 1), tar_kv], dim = 2)).squeeze(2)
            
            weight = self.softmax2(weight)
            
            weight = weight.unsqueeze(2)
            tar = (tar_kv * weight).sum(1)
            
            tar = tar + tar_kv.mean(1)
        seqs = re_embedding
        
        if len(self.attention_layers) > 0:
            seqs *= self.item_emb.embedding_dim ** 0.5
            positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
            seqs = seqs + self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        seqs *= ~timeline_mask.unsqueeze(-1)
        
        # attention mask
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
            
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)
        
        log_feats = self.last_layernorm(seqs)
        
        return log_feats, tar, ind_loss, cent_loss
    
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, epoch): # for training
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        pos_embs = pos_embs * ~timeline_mask.unsqueeze(-1)
        neg_embs = neg_embs * ~timeline_mask.unsqueeze(-1)
        
        log_feats, [pos_embs, neg_embs], ind_loss, cent_loss = self.log2feats(log_seqs, True, [pos_embs, neg_embs], epoch)
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
        return pos_logits, neg_logits, ind_loss, cent_loss
    
    def predict(self, user_ids, log_seqs, item_indices): # for inference
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        
        log_feats, item_embs, ind_loss, cent_loss = self.log2feats(log_seqs, False, item_embs, -1)
        
        final_feat = log_feats[:, -1, :]
        
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        
        return logits
