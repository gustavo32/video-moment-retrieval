import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class EncoderText(nn.Module):
    def __init__(self, vocab, args):
        super(EncoderText, self).__init__()

        # word embedding
        self.embed = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)

        # caption embedding
        self.bi_gru = args.bi_gru
        self.rnn = nn.GRU(args.word_dim, args.embed_size, args.num_layers, batch_first=True, bidirectional=args.bi_gru)


    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.bi_gru:
            cap_emb = (cap_emb[:,:,:cap_emb.size(2)//2] + cap_emb[:,:,cap_emb.size(2)//2:])/2

        return cap_emb


class SCAN(nn.Module):
    def __init__(self, args, vocab):
        super(SCAN, self).__init__()
        self.batch_size = args.batch_size
        self.margin = args.margin
        self.embed_size = args.embed_size
        self.img_dim = args.img_dim
        self.encode_vis = nn.Linear(self.img_dim, self.embed_size)
        self.word_dim = args.word_dim
        self.encode_text = EncoderText(vocab, args)
        self.criterion = nn.MarginRankingLoss(margin=self.margin)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.softmax = nn.Softmax(dim=2)
        self.lambda_lse = 6.0
        self.segments = self.get_segments()

    def xattn_score_i2t(self, vis_feats, desc_feats):
        combined_scores = []
        desc_feats = desc_feats.repeat(len(vis_feats), 1, 1)
        for idx in range(desc_feats.shape[1]):
            word = desc_feats[:, idx, :]
            word = word.unsqueeze(1)
            word = word.repeat(1, vis_feats.shape[1], 1)
            word_seg_scores = self.cos(word, vis_feats)
            word_seg_scores = word_seg_scores.unsqueeze(-1)
            combined_scores.append(word_seg_scores)
        combined_scores = torch.cat(combined_scores, dim=-1)
        combined_scores = self.softmax(combined_scores)  # torch.Size([64, 48, N])
        combined_scores = combined_scores.unsqueeze(-1)

        seg_sentence_feats = combined_scores.repeat(1, 1, 1, vis_feats.shape[-1]) \
                             * desc_feats.unsqueeze(1).repeat(1, vis_feats.shape[1], 1, 1)
        seg_sentence_feats = torch.sum(seg_sentence_feats, dim=2)

        seg_sen_scores = self.cos(vis_feats, seg_sentence_feats)

        if not self.training:
            final_scores = []
            for seg in self.segments:
                tmp_scores = seg_sen_scores[:, seg[0]*25:(seg[1]+1)*25].clone()
                tmp_scores.mul_(self.lambda_lse).exp_()
                tmp_scores = tmp_scores.sum(dim=1, keepdim=True)
                tmp_scores = torch.log(tmp_scores)/self.lambda_lse
                final_scores.append(tmp_scores)
            final_scores = torch.cat(final_scores, dim=1)
            return final_scores

        seg_sen_scores.mul_(self.lambda_lse).exp_()
        seg_sen_scores = seg_sen_scores.sum(dim=1, keepdim=True)
        seg_sen_scores = torch.log(seg_sen_scores) / self.lambda_lse

        return seg_sen_scores

    def get_segments(self):
        segments = []
        for i in range(6):
            for j in range(i, 6):
                segments.append([i, j])
            pass

        return segments

    def forward(self, images, descriptions, img_lens, cap_lens, gts):
        """
        :param video_sample: Visual features and descriptions
        :return: vis_hidden: Includes both visual and optical flow features
                 desc_hidden: hidden states of descriptions where N is the number of
                 annotations and hn is the number of words in the description
        """

        images = images.float().cuda()
        descriptions = descriptions.cuda()

        vis_embed = self.encode_vis(images)
        desc_embed = self.encode_text(descriptions, cap_lens)

        if self.training:
            total_loss = 0.
            for idx in range(len(desc_embed)):
                num_words = cap_lens[idx]
                desc = desc_embed[idx, :num_words, :].unsqueeze(0)
                rel_scores = self.xattn_score_i2t(vis_embed, desc)
                rel_scores = rel_scores.squeeze()
                actual_vid_scores = rel_scores[idx]
                nm_vid_scores = rel_scores[np.arange(len(rel_scores)) != idx]
                actual_vid_scores = actual_vid_scores.repeat(len(nm_vid_scores))
                target = torch.FloatTensor(nm_vid_scores.size()).fill_(1)
                target = Variable(target.cuda())
                loss_sim = self.criterion(actual_vid_scores, nm_vid_scores, target)
                total_loss += loss_sim

            return total_loss

        else:
            scores = []
            for idx in range(len(desc_embed)):
                tmp_vis_feats = vis_embed[idx].unsqueeze(0)
                num_words = cap_lens[idx]
                desc = desc_embed[idx, :num_words, :].unsqueeze(0)
                rel_scores = self.xattn_score_i2t(tmp_vis_feats, desc)
                scores.append(rel_scores)
            scores = torch.cat(scores, dim=0)
            return scores, gts
