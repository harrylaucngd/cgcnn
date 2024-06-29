import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from early_stopping import EarlyStopping

from cgcnn.data_modify import CIFData
from cgcnn.data_modify import collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet

import predict
# 修改了本来的命令行输入
# 按照要求，是将原始数据集8:2分成训练集与测试集，再在训练集里进行五折交叉验证


class Args:
    def __init__(self):
        self.data_options = ['DACs-data-pretrain']
        self.task = 'regression'
        self.disable_cuda = False
        self.workers = 0
        self.epochs = 500
        self.start_epoch = 0
        self.batch_size = 256  # 512
        self.lr = 0.000991  # 0.0007  # 0.0010098097217444582
        self.lr_milestones = [100]
        self.momentum = 0.7519  # 0.6907078649506996
        self.weight_decay = 1.032e-5  # 0.08  # 0.00012726161062048998
        self.print_freq = 100
        self.resume = ''
        self.train_ratio = 0.8
        self.train_size = None
        self.val_ratio = 0
        self.val_size = None
        self.test_ratio = 0.2
        self.test_size = None
        self.optim = 'SGD'
        self.atom_fea_len = 128
        self.h_fea_len = 128
        self.n_conv = 4
        self.n_h = 1   # 池化后的隐藏层数量
        self.cuda = torch.cuda.is_available()
        self.is_5cv = True


args = Args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.

# 修改到你的本地目录
figure_save_path = 'C:\\Users\\86159\\Desktop\\智能化工大作业'
save_name = 'early_17'


def main():
    global args, best_mae_error, figure_save_path, save_name

    # load data
    dataset = CIFData(*args.data_options)
    collate_fn = collate_pool
    train_loaders, val_loaders, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True,
        is_5cv=args.is_5cv
    )

    # obtain target value normalizer
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    all_train_losses = []
    all_val_losses = []
    train_maes = []
    train_r2s = []
    test_maes = []
    test_r2s = []

    cv_num = 5

    # 增加5折交叉验证，准确性存疑，具体交叉验证的数据划分在data.py中存在改动
    for fold in range(cv_num):
        print('-------------Fold {}---------------'.format(fold+1))

        # build model
        structures, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=args.atom_fea_len,
                                    n_conv=args.n_conv,
                                    h_fea_len=args.h_fea_len,
                                    n_h=args.n_h,
                                    classification=True if args.task ==
                                                           'classification' else False)
        if args.cuda:
            model.cuda()

        # define loss func and optimizer
        if args.task == 'classification':
            criterion = nn.NLLLoss()
        else:
            criterion = nn.MSELoss()
        if args.optim == 'SGD':
            optimizer = optim.SGD(model.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.optim == 'Adam':
            optimizer = optim.Adam(model.parameters(), args.lr,
                                   weight_decay=args.weight_decay)
        else:
            raise NameError('Only SGD or Adam is allowed as --optim')

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_mae_error = checkpoint['best_mae_error']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                normalizer.load_state_dict(checkpoint['normalizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                                gamma=0.5)

        train_loader = train_loaders[fold]
        val_loader = val_loaders[fold]

        train_losses = []
        val_losses = []
        train_target = []
        train_output = []

        best_mae_for_fold = 1e10

        # 加入早停
        patience = 30
        early_stopping = EarlyStopping(patience, verbose=True, delta=0)

        for epoch in range(args.start_epoch, args.epochs):
            # train for one epoch
            train_loss, train_target, train_output, train_mae = train(train_loader, model, criterion, optimizer, epoch, normalizer)
            train_losses.append(train_loss)

            # evaluate on validation set
            mae_error, val_loss = validate(val_loader, model, criterion, normalizer)
            val_losses.append(val_loss)

            if mae_error != mae_error:
                print('Exit due to NaN')
                sys.exit(1)

            scheduler.step()

            # remember the best mae_eror and save checkpoint
            if args.task == 'regression':
                is_best = mae_error < best_mae_error
                best_mae_error = min(mae_error, best_mae_error)
                is_best_for_fold = mae_error < best_mae_for_fold
                best_mae_for_fold = min(mae_error, best_mae_for_fold)
            else:
                is_best = mae_error > best_mae_error
                best_mae_error = max(mae_error, best_mae_error)
                is_best_for_fold = mae_error < best_mae_for_fold
                best_mae_for_fold = min(mae_error, best_mae_for_fold)
            save_checkpoint({
                'fold': fold + 1,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mae_error': best_mae_error,
                'optimizer': optimizer.state_dict(),
                'normalizer': normalizer.state_dict(),
                'args': vars(args)
            }, is_best)

            save_checkpoint_for_fold({
                'fold': fold + 1,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mae_error': best_mae_error,
                'optimizer': optimizer.state_dict(),
                'normalizer': normalizer.state_dict(),
                'args': vars(args)
            }, is_best_for_fold)

            if epoch > 200:
                eachepoch_val_loss = np.average(val_loss)
                # 早停法
                early_stopping(eachepoch_val_loss, {
                    'fold': fold + 1,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_mae_error': best_mae_error,
                    'optimizer': optimizer.state_dict(),
                    'normalizer': normalizer.state_dict(),
                    'args': vars(args)
                }, fold+1)
                # 若满足 early stopping 要求
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        # 训练集评估
        print('---------Evaluate Model on Train Set---------------')
        r2_train = r2_score(train_target, train_output)
        train_maes.append(float(train_mae.avg))
        train_r2s.append(r2_train)
        print(' ** Train MAE {mae:.3f}'.format(mae=float(train_mae.avg)))
        print(' ** Train R^2 {r2:.3f}'.format(r2=r2_train))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot((0, 1), (0, 1), linewidth=1, transform=ax.transAxes, ls='--', c='k', label="1:1 line", alpha=0.5)
        ax.plot(train_target, train_output, 'o', c='b', markersize=3, alpha=0.5)
        bbox = dict(boxstyle="round", fc='1', alpha=0.)
        bbox = bbox
        ax.set_xlabel('DFT Values')
        ax.set_ylabel("Predicted values")
        ax.set_title("DFT values vs Predicted Values in Train Set")
        ax.tick_params(labelsize=7)
        x_major_locator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(x_major_locator)
        y_major_locator = MultipleLocator(0.5)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.set(xlim=(-3, 3), ylim=(-3, 3))
        plt.text(-1.3, 2.8, 'R² = {r2:.2f}\nMAE = {mae:.2f}'.format(r2=r2_train, mae=float(train_mae.avg)), fontsize=12,
                 va='top', ha='right')
        plt.savefig(os.path.join(figure_save_path, 'train_{}_{}.png'.format(save_name, fold+1)))
        plt.close()

        # 测试集评估
        # test best model
        print('---------Evaluate Model on Test Set---------------')
        best_checkpoint = torch.load('model_best_fold.pth.tar')
        model.load_state_dict(best_checkpoint['state_dict'])
        test_mae, test_r2, test_targets, test_preds = validate(test_loader, model, criterion, normalizer, test=True)
        r2 = r2_score(test_targets, test_preds)
        print(' ** R^2 {r2:.3f}'.format(r2=r2))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot((0, 1), (0, 1), linewidth=1, transform=ax.transAxes, ls='--', c='k', label="1:1 line", alpha=0.5)
        ax.plot(test_targets, test_preds, 'o', c='b', markersize=3, alpha=0.5)
        bbox = dict(boxstyle="round", fc='1', alpha=0.)
        bbox = bbox
        ax.set_xlabel('DFT Values')
        ax.set_ylabel("Predicted values")
        ax.tick_params(labelsize=7)
        ax.set_title("DFT values vs Predicted Values in Test Set")
        x_major_locator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(x_major_locator)
        y_major_locator = MultipleLocator(0.5)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.set(xlim=(-3, 3), ylim=(-3, 3))
        plt.text(-1.3, 2.8, 'R² = {r2:.2f}\nMAE = {mae:.2f}'.format(r2=r2, mae=float(test_mae)), fontsize=12,
                 va='top', ha='right')
        plt.savefig(os.path.join(figure_save_path, 'test_{}_{}.png'.format(save_name, fold + 1)))
        plt.close()
        test_maes.append(test_mae)
        test_r2s.append(test_r2)

    # 增加loss曲线绘制
    for i in range(cv_num):
        plt.plot(all_train_losses[i], label='Train Fold {}'.format(i+1))
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.savefig(os.path.join(figure_save_path, 'train_loss_{}.png'.format(save_name)))
    plt.close()

    for i in range(cv_num):
        plt.plot(all_val_losses[i], label='Val Fold {}'.format(i+1))
    plt.xlabel('Epochs')
    plt.ylabel('Val Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.savefig(os.path.join(figure_save_path, 'Val_loss_{}.png'.format(save_name)))
    plt.close()

    print('---------5-fold Cross-Validation Results---------------')
    print('mean_train_mae:', np.array(train_maes).mean())
    print('mean_train_r2:', np.array(train_r2s).mean())
    print('mean_test_mae:', np.array(test_maes).mean())
    print('mean_test_r2:', np.array(test_r2s).mean())

    # 预测
    # predict.main()


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    global figure_save_path, save_name
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                )

                return losses.avg, target.view(-1).tolist(), normalizer.denorm(output.data.cpu()).view(-1).tolist(), mae_errors
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, accu=accuracies,
                    prec=precisions, recall=recalls, f1=fscores,
                    auc=auc_scores)
                )

                return losses.avg, target.view(-1).tolist(), normalizer.denorm(output.data.cpu()).view(-1).tolist(), mae_errors


def validate(val_loader, model, criterion, normalizer, test=False):
    global figure_save_path, save_name
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    if args.task == 'regression':
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors))
        if test:
            # 增加输出r2
            r2 = r2_score(test_targets, test_preds)
            return mae_errors.avg, r2, test_targets, test_preds
        else:
            return mae_errors.avg, losses.avg
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
        return auc_scores.avg, losses.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_checkpoint_for_fold(state, is_best, filename='checkpoint_fold.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_fold.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
