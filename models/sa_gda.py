import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from models.layers import pool, MyRNN, GCNLayer, DenseGCN
from models.layers import PositionAwareAttention
from models.layers import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils

class SegAttnClassifier(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super(SegAttnClassifier, self).__init__()
        self.opt = opt
        self.prune = opt['prune']
        self.emb_matrix = emb_matrix
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim']) if opt['pe_dim'] > 0 else None
        
        self.in_dim = opt['emb_dim']
        if self.pos_emb:
            self.in_dim += opt['pos_dim']
        if self.ner_emb:
            self.in_dim += opt['ner_dim']
        if self.pe_emb:
            self.in_dim += 2 * opt['pe_dim']
        
        self.mem_dim = opt['hidden_dim']
        
        self.rnn = MyRNN(self.in_dim, self.mem_dim // 2, opt['rnn_layer'],
            bidirectional=True, use_cuda=opt['cuda'])
 
        self.in_drop = nn.Dropout(opt['in_dropout'])
        self.drop = nn.Dropout(opt['dropout'])

        self.conv_2 = nn.Conv1d(self.mem_dim, self.mem_dim, 2, padding=0)
        self.conv_3 = nn.Conv1d(self.mem_dim, self.mem_dim, 3, padding=1)

        self.l0 = nn.Linear(self.mem_dim * 3, self.mem_dim)
        self.l1 = nn.Linear(self.mem_dim * 2, self.mem_dim)
        self.l2 = nn.Linear(self.mem_dim * 2, self.mem_dim)
        self.l3 = nn.Linear(self.mem_dim * 3, self.mem_dim)
        self.l4 = nn.Linear(self.mem_dim * 4, self.mem_dim)
        self.classifier = nn.Linear(self.mem_dim, opt['num_class'])

        self.init_embeddings()

    def init_embeddings(self):
        nn.init.xavier_normal_(self.l0.weight)
        nn.init.xavier_normal_(self.l1.weight)
        nn.init.xavier_normal_(self.l2.weight)
        nn.init.xavier_normal_(self.l3.weight)
        nn.init.xavier_normal_(self.l4.weight)
        if self.opt['pe_dim'] > 0:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data.uniform_(-1.0, 1.0)
        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data.uniform_(-1.0, 1.0)

        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)

        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)
        word_embs = self.emb(words)
        embs = [word_embs]

        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        if self.opt['pe_dim'] > 0:
            embs += [self.pe_emb(subj_pos + constant.MAX_LEN)]
            embs += [self.pe_emb(obj_pos + constant.MAX_LEN)]
        
        # Embedding of word and pos and ner and position
        embs = torch.cat(embs, dim=2)
        # Dorpout of input
        embs = self.in_drop(embs)

        # Encoder Layer
        rnn_outputs, hidden = self.rnn(embs, masks)
        hidden = torch.cat([hidden[-1, :, :], hidden[-2, :, :]], dim=-1)

        # 2-gram
        segment_2 = F.relu(self.conv_2(rnn_outputs.permute(0, 2, 1)).permute(0, 2, 1))
        pad = torch.zeros(embs.size(0), 1, self.mem_dim).cuda()
        segment_2 = torch.cat([segment_2, pad], dim=1)
        # 3-gram
        segment_3 = F.relu(self.conv_3(rnn_outputs.permute(0, 2, 1)).permute(0, 2, 1))
        
        # Mask of subj and obj to get max hidden varibale of subj and obj
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)
        e1 = pool(rnn_outputs, subj_mask, "max")
        e2 = pool(rnn_outputs, obj_mask, "max")

        out_list = []
        # hidden entities to get information for relation 
        query_ent = self.l0(torch.cat([hidden, e1, e2], dim=-1))
        out_list.append(attention(query_ent, rnn_outputs, rnn_outputs, masks))

        h1 = attention(e1, rnn_outputs, rnn_outputs, masks)
        h2 = attention(e2, rnn_outputs, rnn_outputs, masks)
        h0 = attention(hidden, rnn_outputs, rnn_outputs, masks)
        # out_list.append(h0)

        query_men = self.l1(torch.cat([h1, h2], dim=-1))
        men_out = attention(query_men, rnn_outputs, rnn_outputs, masks)
        out_list.append(men_out)

        out_seg2 = attention(query_men, segment_2, segment_2, masks)
        out_seg3 = attention(query_men, segment_3, segment_3, masks)
        out_list.append(out_seg2)
        out_list.append(out_seg3)

        # query_seg = self.l2(torch.cat([men_out, h0], dim=-1))
        # out_seg2 = attention(query_seg, segment_2, segment_2, masks)
        # out_seg3 = attention(query_seg, segment_3, segment_3, masks)
        # out_list.append(out_seg2)
        # out_list.append(out_seg3)
        

        outputs = torch.cat(out_list, dim=-1)

        outputs = F.relu(self.l4(outputs))

        outputs = self.drop(outputs)
        outputs = self.classifier(outputs)
        return outputs


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    query = query.unsqueeze(1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value).squeeze()

def Qattention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value).squeeze()