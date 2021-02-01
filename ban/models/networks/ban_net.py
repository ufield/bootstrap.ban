from copy import deepcopy
import itertools
import os
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
import block
from block.models.networks.vqa_net import factory_text_enc
from block.models.networks.vqa_net import mask_softmax
from block.models.networks.mlp import MLP
from .dev.slp import SLP
from .dev.fc_layer import FCLayer
from .dev.attention import LowLankBilinearPooling, BilinearAttentionMap
from .dev.classifier import Classifier

import pdb

class BanNet(nn.Module):

    def __init__(self,
            txt_enc={},
            q_max_length=14,
            glimpse=2,
            objects=36,
            feat_dims={},
            biattention={},
            wid_to_word={},
            word_to_wid={},
            aid_to_ans=[],
            ans_to_aid={}):
        super(BanNet, self).__init__()
        # self.self_q_att = self_q_att
        self.glimpse = glimpse
        self.q_max_length = q_max_length
        self.objects = objects
        # self.classif = classif
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        # Modules

        self.txt_enc = factory_text_enc(self.wid_to_word, txt_enc)
        self.v_att = BilinearAttentionMap(**feat_dims, glimpse=glimpse)

        self.b_net = []
        self.q_prj = []
        self.c_prj = []

        for i in range(glimpse):
            # self.b_net.append(LowLankBilinearPooling(
            #                     feat_dims['v_dim'], feat_dims['q_dim'], feat_dims['h_dim'],
            #                     None, k=1))
            self.b_net.append(LowLankBilinearPooling(**feat_dims, h_out=None, k=1))
            self.q_prj.append(FCLayer(feat_dims['h_dim'], feat_dims['h_dim'], '', .2))
            self.c_prj.append(FCLayer(objects + 1, feat_dims['h_dim'], 'ReLU', .0))

        self.b_net = nn.ModuleList(self.b_net)
        self.q_prj = nn.ModuleList(self.q_prj)
        self.c_prj = nn.ModuleList(self.c_prj)

        self.classifier = Classifier(feat_dims['h_dim'], feat_dims['h_dim']*2, 3000, 0.5)

        # ここで skipthoughts している
        # if self.self_q_att:
        #     self.q_att_linear0 = nn.Linear(2400, 512)
        #     self.q_att_linear1 = nn.Linear(512, 2)

        # if self.shared:
        #     self.cell = MuRelCell(**cell)
        # else:
        #     self.cells = nn.ModuleList([MuRelCell(**cell) for i in range(self.n_step)])

        # self.slp = SLP(**self.classif['slp'])

        # if 'fusion' in self.classif:
        #     self.classif_module = block.factory_fusion(self.classif['fusion'])
        # elif 'mlp' in self.classif:
        #     self.classif_module = MLP(self.classif['mlp'])
        # else:
        #     raise ValueError(self.classif.keys())

        Logger().log_value('nparams',
            sum(p.numel() for p in self.parameters() if p.requires_grad),
            should_print=True)

        # Logger().log_value('nparams_txt_enc',
        #     self.get_nparams_txt_enc(),
        #     should_print=True)

        # self.buffer = None


    # def get_nparams_txt_enc(self):
    #     params = [p.numel() for p in self.txt_enc.parameters() if p.requires_grad]
    #     if self.self_q_att:
    #         params += [p.numel() for p in self.q_att_linear0.parameters() if p.requires_grad]
    #         params += [p.numel() for p in self.q_att_linear1.parameters() if p.requires_grad]
    #     return sum(params)

    # def set_buffer(self):
    #     self.buffer = {}
    #     if self.shared:
    #         self.cell.pairwise.set_buffer()
    #     else:
    #         for i in range(self.n_step):
    #             self.cell[i].pairwise.set_buffer()

    # def set_pairs_ids(self, n_regions, bsize, device='cuda'):
    #     if self.shared and self.cell.pairwise:
    #         self.cell.pairwise_module.set_pairs_ids(n_regions, bsize, device=device)
    #     else:
    #         for i in self.n_step:
    #             if self.cells[i].pairwise:
    #                 self.cells[i].pairwise_module.set_pairs_ids(n_regions, bsize, device=device)

    def forward(self, batch):
        # batch のもとは、 https://github.com/Cadene/block.bootstrap.pytorch/blob/master/block/datasets/vqa2.py で作成
        v = batch['visual']            # v.shape:  torch.Size([12, 36, 2048])
        q = batch['question']          # q.shape:  torch.Size([12, question_length_batch_max])
        # l = batch['lengths'].data      # l.shape:  torch.Size([12, 1])
        b = batch['norm_coord']        # c.shape:  torch.Size([12, 36, 4])

        q = self.pad_trim_question(q, self.q_max_length)
        q_emb = self.process_question(q)  # q_emb.shape: torch.Size([12, 14, 2400])


        boxes = b[:,:,:4].transpose(1,2)

        b_emb = [0] * self.glimpse
        attn, logits = self.v_att(v, q_emb)   # attn.shape: torch.Size([12, 2, 36, 14])
        # attn, logits = self.v_att.forward(v, q_emb)

        # pdb.set_trace()

        # pdb.set_trace()
        # TODO: Counter module.

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g](v, q_emb, attn[:, g, :, :])  # batch x h_dim x h_dim, eq. (5) in paper
            # b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, attn[:, g, :, :])  # batch x h_dim x h_dim, eq. (5) in paper

            # atten, _ = logits[:, g, :, :].max(2)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb

            # embed = self.

        # b_emb[0].shape: torch.Size([12, 512])
        # q_emb.shape: torch.Size([12, 14, 2400])
        # pdb.set_trace()

        logits = self.classifier(q_emb.sum(1))
        # q.shape: torch.Size([bsize, 14, 2400])

        bsize = q.shape[0]
        n_regions = v.shape[1] # v.shape = (batch, 36, 2048)

        # q_expand = q[:,None,:].expand(bsize, n_regions, q.shape[1]) # q.shape[1] = 4800?
        # q_expand = q_expand.contiguous().view(bsize*n_regions, -1)  # q_expand.shape:  torch.Size([432, 4800])

        # logits = self.slp(q) # 暫定的
        # logits = torch.zeros([bsize,3000]) # 暫定的
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # logits = logits.to(device)
        out = {'logits': logits}
        # mm = v


        # print('v.shape: ', v.shape)
        # print('q.shape: ', q.shape)
        # print('q_expand.shape: ', q_expand.shape)
        # print('l.shape: ', l.shape)
        # print('c.shape: ', c.shape)
        # pdb.set_trace()

        # for i in range(self.n_step):
        #     cell = self.cell if self.shared else self.cells[i] # デフォルトではcell を share している
        #     mm = cell(q_expand, mm, c)       # cell -> in: , out: (batch_size, 2048)

        #     if self.buffer is not None: # for visualization
        #         self.buffer[i] = deepcopy(cell.pairwise.buffer)

        # if self.agg['type'] == 'max':
        #     mm = torch.max(mm, 1)[0]
        # elif self.agg['type'] == 'mean':
        #     mm = mm.mean(1)

        # if 'fusion' in self.classif:
        #     logits = self.classif_module([q, mm])
        # elif 'mlp' in self.classif:
        #     logits = self.classif_module(mm)

        # out = {'logits': logits}   # logits.shape: torch.Size([12, 3000])
        return out

    def pad_trim_question(self, q, max_length):
        '''
            max_length 以下の question は 0 padding
            max_length 以上の question は trim
        '''
        tmp = torch.zeros([q.shape[0], max_length], dtype=torch.int)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tmp = tmp.to(device)
        q = torch.cat((q, tmp), 1)
        q = q[:, :max_length]
        return q

    def process_question(self, q):
        q_emb = self.txt_enc.embedding(q)
        q, _ = self.txt_enc.rnn(q_emb)
        return q

    # def process_question(self, q, l):
    #     q_emb = self.txt_enc.embedding(q)

    #     # q_emb.shape:  torch.Size([12, question_length, 620])
    #     print('q_emb.shape: ', q_emb.shape)

    #     q, _ = self.txt_enc.rnn(q_emb)

    #     # q.shape: torch.Size([12, question_length, 2400])
    #     print('q.shape: ', q.shape)


    #     if self.self_q_att:
    #         q_att = self.q_att_linear0(q)
    #         q_att = F.relu(q_att)
    #         q_att = self.q_att_linear1(q_att)
    #         q_att = mask_softmax(q_att, l)
    #         #self.q_att_coeffs = q_att
    #         if q_att.size(2) > 1:
    #             q_atts = torch.unbind(q_att, dim=2)
    #             q_outs = []
    #             for q_att in q_atts:
    #                 q_att = q_att.unsqueeze(2)
    #                 q_att = q_att.expand_as(q)
    #                 q_out = q_att*q
    #                 q_out = q_out.sum(1)
    #                 q_outs.append(q_out)
    #             q = torch.cat(q_outs, dim=1)
    #         else:
    #             q_att = q_att.expand_as(q)
    #             q = q_att * q
    #             q = q.sum(1)
    #     else:
    #         # l contains the number of words for each question
    #         # in case of multi-gpus it must be a Tensor
    #         # thus we convert it into a list during the forward pass
    #         l = list(l.data[:,0])
    #         q = self.txt_enc._select_last(q, l)

    #     return q

    def process_answers(self, out):
        batch_size = out['logits'].shape[0]
        _, pred = out['logits'].data.max(1)
        pred.squeeze_()
        out['answers'] = [self.aid_to_ans[pred[i]] for i in range(batch_size)]
        out['answer_ids'] = [pred[i] for i in range(batch_size)]
        return out
