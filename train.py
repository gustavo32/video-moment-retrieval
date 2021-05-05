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
    parser.add_argument('--video_features_path', default='rgb_vgg_fc7_features/',
                        help='path to video extracted features')
    parser.add_argument('--audio_features_path', default='audio_mfcc_features/',
                        help='path to audio extracted features')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--margin', default=0.7, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=10, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--img_dim', default=4096, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--audio_dim', default=40, type=int,
                        help='Dimensionality of the audio embedding.')
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
    parser.add_argument('--bi_gru', action='store_true',
                        help='Use bidirectional GRU.')
    parser.add_argument('--lambda_lse', default=6., type=float,
                        help='LogSumExp temp.')
    parser.add_argument('--lambda_softmax', default=9., type=float,
                        help='Attention softmax temperature.')
    opt = parser.parse_args()

    opt.bi_gru = True
    opt.max_violation = True
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


def moving_average(x, window):
    return np.convolve(x, np.ones(window), 'valid') / window


def each_pred_moving_average(scores, period):
    length = len(scores)
    mean_pad_scores = np.pad(scores, (period // 2, period // 2), "mean")
    ma = moving_average(mean_pad_scores, period)
    argmax_ma = np.argmax(ma)
    proposals = ma > scores.mean()

    start_pred = argmax_ma
    while start_pred >= 0 and proposals[start_pred]:
        start_pred -= 1

    end_pred = argmax_ma
    while end_pred < length and proposals[end_pred]:
        end_pred += 1

    return [start_pred + 1, end_pred - 1]


def prediction_by_moving_average(scores_videos, period):
    return np.apply_along_axis(each_pred_moving_average, 1, scores_videos, period=period)


def validate(opt, val_loader, model):
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    all_y_true = []
    all_preds_5 = []
    all_preds_9 = []
    all_preds_15 = []
    all_preds_30 = []
    for i, val_data in enumerate(val_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        with torch.no_grad():
            video_emb, cap_emb = model.forward_emb(*val_data)

        batch_preds_5 = []
        batch_preds_9 = []
        batch_preds_15 = []
        batch_preds_30 = []
        for j in range(len(video_emb)):
            cap_i = cap_emb[j, :val_data[3][j], :].unsqueeze(0).contiguous()
            img_i = video_emb[j].unsqueeze(0).contiguous()
            weiContext, attn = func_attention(img_i, cap_i, opt, smooth=opt.lambda_softmax)
            sim = cosine_similarity(img_i, weiContext, dim=2)
            batch_preds_5.append(torch.tensor(each_pred_moving_average(sim.cpu().numpy(), 5)) // 25)
            batch_preds_9.append(torch.tensor(each_pred_moving_average(sim.cpu().numpy(), 9)) // 25)
            batch_preds_15.append(torch.tensor(each_pred_moving_average(sim.cpu().numpy(), 15)) // 25)
            batch_preds_30.append(torch.tensor(each_pred_moving_average(sim.cpu().numpy(), 30)) // 25)

        all_preds_5.append(torch.vstack(batch_preds_5))
        all_preds_9.append(torch.vstack(batch_preds_9))
        all_preds_15.append(torch.vstack(batch_preds_15))
        all_preds_30.append(torch.vstack(batch_preds_30))
        all_y_true.append(val_data[-1])

    all_preds_5 = torch.vstack(all_preds_5)
    all_preds_9 = torch.vstack(all_preds_9)
    all_preds_15 = torch.vstack(all_preds_15)
    all_preds_30 = torch.vstack(all_preds_30)
    all_y_true = torch.cat(all_y_true)

    average_iou = []
    average_ranks = []
    for gts, pred in zip(all_y_true.numpy(), all_preds_5.numpy()):
        ious = [iou(pred, gt) for gt in gts]
        average_iou.append(np.mean(np.sort(ious)[-3:]))
        ranks = [1 if tuple(gt) == tuple(pred) else 2 for gt in gts]
        average_ranks.append(np.mean(np.sort(ranks)[:3]))
    rank1 = np.sum(np.array(average_ranks) <= 1) / float(len(average_ranks))
    miou = np.mean(average_iou)
    print("PERIOD 5:")
    print("Average rank@1: %f" % rank1)
    print("Average iou: %f" % miou)


    average_iou = []
    average_ranks = []
    for gts, pred in zip(all_y_true.numpy(), all_preds_9.numpy()):
        ious = [iou(pred, gt) for gt in gts]
        average_iou.append(np.mean(np.sort(ious)[-3:]))
        ranks = [1 if tuple(gt) == tuple(pred) else 2 for gt in gts]
        average_ranks.append(np.mean(np.sort(ranks)[:3]))
    rank1 = np.sum(np.array(average_ranks) <= 1) / float(len(average_ranks))
    miou = np.mean(average_iou)
    print("PERIOD 9:")
    print("Average rank@1: %f" % rank1)
    print("Average iou: %f" % miou)


    average_iou = []
    average_ranks = []
    for gts, pred in zip(all_y_true.numpy(), all_preds_15.numpy()):
        ious = [iou(pred, gt) for gt in gts]
        average_iou.append(np.mean(np.sort(ious)[-3:]))
        ranks = [1 if tuple(gt) == tuple(pred) else 2 for gt in gts]
        average_ranks.append(np.mean(np.sort(ranks)[:3]))
    rank1 = np.sum(np.array(average_ranks) <= 1) / float(len(average_ranks))
    miou = np.mean(average_iou)
    print("PERIOD 15:")
    print("Average rank@1: %f" % rank1)
    print("Average iou: %f" % miou)


    average_iou = []
    average_ranks = []
    for gts, pred in zip(all_y_true.numpy(), all_preds_30.numpy()):
        ious = [iou(pred, gt) for gt in gts]
        average_iou.append(np.mean(np.sort(ious)[-3:]))
        ranks = [1 if tuple(gt) == tuple(pred) else 2 for gt in gts]
        average_ranks.append(np.mean(np.sort(ranks)[:3]))
    miou = np.mean(average_iou)
    print("PERIOD 30:")
    print("Average rank@1: %f" % rank1)
    print("Average iou: %f" % miou)

    return rank1, miou

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
