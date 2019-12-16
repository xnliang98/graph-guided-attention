import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils import constant, torch_utils

class MTAClassifier(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super(MTAClassifier, self).__init__()
        self.opt = opt
        
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

def pool(h, mask, pool_type='max'):
    if pool_type == 'max':
        h = h.masked_fill(mask, -1e9)
        return torch.max(h, 1)[0]
    elif pool_type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

class MyRNN(nn.Module):
    """ A wrapper of rnn layer."""
    def __init__(self,
                input_size, 
                hidden_dim, 
                num_layers=1, 
                bidirectional=False, 
                dropout=0, 
                batch_first=True, 
                use_cuda=True,
                rnn_type='lstm'): # rnn, lstm, gru
        super(MyRNN, self).__init__()

        self.use_cuda = use_cuda
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.num_layers = num_layers
        self.direct = 2 if bidirectional else 1
        self.batch_first = batch_first
    
        rnn_type = rnn_type.upper()
        args = dict(input_size=input_size, hidden_size=hidden_dim, 
                        num_layers=num_layers, bidirectional=bidirectional, 
                        dropout=dropout, batch_first=batch_first)
        self.rnn = getattr(nn, rnn_type)(**args)

    def rnn_zero_state(self, batch_size):
        total_layers = self.num_layers * self.direct
        state_shape = (total_layers, batch_size, self.hidden_dim)
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0        
    
    def forward(self, inputs, masks=None):
        batch_size = inputs.size(0)
        h0, c0 = self.rnn_zero_state(batch_size)
        if masks is None:
            rnn_outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        else:
            seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
            rnn_inputs = pack_padded_sequence(inputs, seq_lens, batch_first=self.batch_first)
            rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
            rnn_outputs, _ = pad_packed_sequence(rnn_outputs, batch_first=self.batch_first)
        
        return rnn_outputs, ht
