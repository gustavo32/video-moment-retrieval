import time
import numpy as np
from itertools import product
import argparse

import logging
import tensorboard_logger as tb_logger

import torch
from torchtext.vocab import GloVe

import data
from model import SCAN, func_attention, cosine_similarity
from evaluation import AverageMeter, LogCollector


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/',
                        help='path to datasets')
    parser.add_argument('--pretrained_path', default='rgb_vgg_fc7_features/',
                        help='path to extracted features')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--margin', default=0.7, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=10, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=50, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=256, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='./runs/runX/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/runX/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=4096, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--raw_feature_norm', default="clipped_l2norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--agg_func', default="LogSumExp",
                        help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--cross_attn', default="t2i",
                        help='t2i|i2t')
    parser.add_argument('--precomp_enc_type', default="basic",
                        help='basic|weight_norm')
    parser.add_argument('--bi_gru', action='store_true',
                        help='Use bidirectional GRU.')
    parser.add_argument('--lambda_lse', default=6., type=float,
                        help='LogSumExp temp.')
    parser.add_argument('--lambda_softmax', default=9., type=float,
                        help='Attention softmax temperature.')
    opt = parser.parse_args()

    opt.bi_gru = True
    opt.max_violation = True
    opt.agg_func = "Mean"
    print(opt)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    vocab = GloVe(name="6B", dim=50)
    opt.vocab_size = len(vocab)
    opt.val_step = 500

    # Load data loaders
    train_loader, val_loader = data.get_loaders(vocab, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = SCAN(opt, vocab)
    # model = torch.nn.DataParallel(model)
    # model.cuda()

    # Train the Model
    for epoch in range(opt.num_epochs):
        print(opt.logger_name)
        print(opt.model_name)
        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

    # evaluate on validation set
    miou = validate(opt, val_loader, model)


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model)


def iou(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return float(intersection)/union


def rank(preds, gt):
    return preds.index(list(gt)) + 1


def validate(opt, val_loader, model):
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    proposals = np.asarray(list(product(range(6), range(6))))
    proposals = proposals[proposals[:, 1] >= proposals[:, 0]]

    all_proposals = []
    all_y_true = []
    for i, val_data in enumerate(val_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb = model.forward_emb(*val_data)

        batch_proposals = []
        for j in range(len(img_emb)):
            row_sim = []
            for p in proposals:
                if p[1] <= (val_data[2][j] - 1) // 25:
                    proposal_img_emb = img_emb[j, p[0]*25:min((p[1]+1)*25, val_data[2][j])].unsqueeze(0).contiguous()
                    cap_i = cap_emb[j, :val_data[3][j], :].unsqueeze(0).contiguous()
                    weiContext, attn = func_attention(proposal_img_emb, cap_i, opt, smooth=opt.lambda_softmax)
                    sim = cosine_similarity(proposal_img_emb, weiContext, dim=2)
                    if opt.agg_func == 'LogSumExp':
                        sim.mul_(opt.lambda_lse).exp_()
                        sim = sim.sum(dim=0, keepdim=True)
                        sim = torch.log(sim) / opt.lambda_lse
                    elif opt.agg_func == 'Mean':
                        sim = sim.sum(dim=0, keepdim=True)
                    row_sim.append(sim.view(-1))
                else:
                    row_sim.append(torch.tensor([-5.]).cuda()) #arrumar isso

            row_sim = torch.stack(row_sim, 0)
            ind = torch.argsort(row_sim.view(-1), descending=True).cpu().numpy()
            batch_proposals.append(torch.tensor(proposals[ind]))

        all_proposals.append(torch.cat(batch_proposals))
        all_y_true.append(val_data[-1])

    all_proposals = torch.cat(all_proposals).view(-1, 21, 2)
    all_y_true = torch.cat(all_y_true)

    average_iou = []
    average_ranks = []
    for gts, preds in zip(all_y_true, all_proposals):
        pred = preds[0]
        ious = [iou(pred, gt) for gt in gts]
        average_iou.append(np.mean(np.sort(ious)[-3:]))
        list_preds = preds.numpy().tolist()
        ranks = [rank(list_preds, gt) for gt in gts.numpy()]
        average_ranks.append(np.mean(np.sort(ranks)[:3]))

    rank1 = np.sum(np.array(average_ranks) <= 1) / float(len(average_ranks))
    rank3 = np.sum(np.array(average_ranks) <= 3) / float(len(average_ranks))
    rank5 = np.sum(np.array(average_ranks) <= 5) / float(len(average_ranks))
    miou = np.mean(average_iou)
    print("Average rank@1: %f" % rank1)
    print("Average rank@3: %f" % rank3)
    print("Average rank@5: %f" % rank5)
    print("Average iou: %f" % miou)
    return rank1, rank5, miou

# Frame level attention with word
# Average rank@1: 0.144498
# Average rank@3: 0.323206
# Average rank@5: 0.421053
# Average iou: 0.256846

# Word level attention with image sequence
# Average rank@1: 0.102632
# Average rank@3: 0.321531
# Average rank@5: 0.538995
# Average iou: 0.244398

if __name__ == '__main__':
    main()
