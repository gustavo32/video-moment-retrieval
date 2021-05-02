import time
import numpy as np
from itertools import product
import argparse

import logging
import tensorboard_logger as tb_logger

import torch
from torchtext.vocab import GloVe
import torch.backends.cudnn as cudnn

import data
from model import SCAN
from evaluation import AverageMeter, LogCollector

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../VideoMomentRetrieval/data/',
                        help='path to datasets')
    parser.add_argument('--pretrained_path', default='../VideoMomentRetrieval/rgb_vgg_fc7_features/',
                        help='path to extracted features')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--margin', default=0.7, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=10, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=32, type=int,
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
    parser.add_argument('--workers', default=16, type=int,
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
    print(opt)

    opt.cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    cudnn.benchmark = True

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    vocab = GloVe(name="6B", dim=50)
    opt.vocab_size = len(vocab)
    opt.val_step = 500

    # Load data loaders
    train_loader, val_loader = data.get_loaders(vocab, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = SCAN(opt, vocab)
    model = torch.nn.DataParallel(model)
    model.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=opt.learning_rate)
    num_parameters = sum([p.data.nelement() for p in model.parameters() if p.requires_grad])
    print('  + Number of params: {}'.format(num_parameters))

    # Train the Model
    for epoch in range(opt.num_epochs):
        print(opt.logger_name)
        print(opt.model_name)
        # train for one epoch
        # test(opt, val_loader, model)
        train(opt, train_loader, model, epoch, optimizer)
        test(opt, val_loader, model)

    # evaluate on validation set
    test(opt, val_loader, model)


def train(opt, train_loader, model, epoch, optimizer):
    # average meters to record the training statistics
    model.train()
    losses = AverageMeter()

    for i, train_data in enumerate(train_loader):

        # Update the model
        loss = model(*train_data)
        loss = torch.mean(loss)

        num_items = len(train_data[0])
        losses.update(loss.data, num_items)

        # measure elapsed time
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % opt.log_step == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f})'.format(
                    epoch, i * num_items, len(train_loader.dataset),
                    losses.val, losses.avg))


def actual_iou(top_five_indices, gt_times, segments):
    combined_ious = []
    for i in range(5):
        seg_idx = top_five_indices[:, i]
        tmp_segs = torch.index_select(segments, 1, seg_idx)
        tmp_segs = tmp_segs[:, 0, :] # torch.Size([64, 2])
        ious = []
        for j in range(4):
            curr_gt = gt_times[:, j, :]
            gt_start = curr_gt[:, 0]
            gt_end = curr_gt[:, 1] + 1
            pred_start = tmp_segs[:, 0]
            pred_end = tmp_segs[:, 1] + 1
            intersection = np.minimum(gt_end.cpu().numpy(), pred_end.cpu().numpy()) + 1 - np.maximum(gt_start.cpu().numpy(), pred_start.cpu().numpy())
            intersection = np.maximum(0., intersection)
            union = np.maximum(gt_end.cpu().numpy(), pred_end.cpu().numpy()) + 1 - np.minimum(pred_start.cpu().numpy(), gt_start.cpu().numpy())
            iou_val = np.divide(intersection, union)
            iou_val = torch.from_numpy(iou_val)
            ious.append(iou_val.unsqueeze(-1))
        ious = torch.cat(ious, dim=1)
        combined_ious.append(ious.unsqueeze(1))
    combined_ious = torch.cat(combined_ious, dim=1)
    return combined_ious



def iou(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) + 1 - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) + 1 - min(pred[0], gt[0])
    return float(intersection)/union


def rank(preds, gt):
    return preds.index(list(gt)) + 1


def test(opt, test_loader, model):
    model.eval()

    segments = []
    for i in range(6):
        for j in range(i, 6):
            segments.append([i, j])
    segments = np.array(segments)
    segments = torch.from_numpy(segments).cuda()

    miou = 0.
    total_iou = 0.
    num_correct = 0.
    acc_5 = 0.
    num_samples = 0.
    total_miou = []

    all_y_true = []
    all_proposals = []
    for i, val_data in enumerate(test_loader):
        num_samples += len(val_data[0])

        confidence_scores, gt_flipped = model(*val_data)  # torch.Size([64, 21])

        tmp_segs = segments.clone().unsqueeze(0)
        tmp_segs = tmp_segs.repeat(len(val_data[0]), 1, 1)

        top_five, ind_five = torch.topk(confidence_scores, 5, dim=1, largest=True)
        ious = actual_iou(ind_five, gt_flipped, tmp_segs)

        top_one_ious = ious[:, 0, :]
        mean_top_one, mean_ind_one = torch.topk(top_one_ious, 3, dim=1)  # torch.Size([64, 3])
        mean_top_one = torch.mean(mean_top_one, dim=1)
        miou += torch.sum(mean_top_one)
        sat = mean_top_one >= 0.5
        num_correct += torch.sum(sat)

        mean_top_five, mean_ind_five = torch.topk(ious, 3, dim=2)
        mean_top_five = torch.mean(mean_top_five, dim=2)
        sat_five = mean_top_five >= 0.5
        sat_five = torch.sum(sat_five, dim=1)
        sat_five = sat_five >= 1
        acc_5 += torch.sum(sat_five)

        indices = torch.argsort(confidence_scores, dim=1)

        all_y_true.append(gt_flipped)
        all_proposals.append(segments[indices])

    all_y_true = torch.vstack(all_y_true)
    all_proposals = torch.vstack(all_proposals)

    acc = (float(num_correct) / num_samples) * 100
    acc_5 = (float(acc_5) / num_samples) * 100
    miou = (float(miou) / num_samples) * 100
    # final_miou = np.mean(total_miou) * 100
    print("")
    print('R@1 accuracy: ' + str(acc) + '\n' + 'R@5 accuracy: ' + str(acc_5) + '\n' + 'miou: ' + str(miou))
    print("")

    average_iou = []
    average_ranks = []
    for gts, preds in zip(all_y_true.cpu().numpy(), all_proposals.cpu().numpy()):
        pred = preds[0]
        ious = [iou(pred, gt) for gt in gts]
        average_iou.append(np.mean(np.sort(ious)[-3:]))
        list_preds = preds.tolist()
        ranks = [rank(list_preds, gt) for gt in gts]
        average_ranks.append(np.mean(np.sort(ranks)[:3]))

    rank1 = np.sum(np.array(average_ranks) <= 1) / float(len(average_ranks))
    rank3 = np.sum(np.array(average_ranks) <= 3) / float(len(average_ranks))
    rank5 = np.sum(np.array(average_ranks) <= 5) / float(len(average_ranks))
    miou = np.mean(average_iou)
    print("Average rank@1: %f" % rank1)
    print("Average rank@3: %f" % rank3)
    print("Average rank@5: %f" % rank5)
    print("Average iou: %f" % miou)


if __name__ == '__main__':
    main()
