# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from datasets.custom.dataset import PoseDataset as PoseDataset_custom
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or linemod')
    parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
    parser.add_argument('--batch_size', type=int, default = 8, help='batch size')
    parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
    parser.add_argument('--lr', default=0.0001, help='learning rate')
    parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
    parser.add_argument('--w', default=0.3, help='regularize confidence')
    parser.add_argument('--w_rate', default=0.3, help='regularize confidence refiner decay')
    parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
    parser.add_argument('--refine_epoch', default=4, type=int, help='epoch to start the training of iterative refinement')
    parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
    parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
    parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
    parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
    parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
    parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
    parser.add_argument('--image_size', type=int, default=300, help="square side length of cropped image")

    parser.add_argument('--num_rot_bins', type=int, default = 90, help='number of bins discretizing the rotation around front')
    parser.add_argument('--profile', action="store_true", default=False, help='should we performance profile?')
    opt = parser.parse_args()

    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.num_points = 500 #number of points on the input pointcloud
        opt.outf = 'trained_models/ycb' #folder to save trained models
        opt.log_dir = 'experiments/logs/ycb' #folder to save logs
        opt.repeat_epoch = 1 #number of repeat times for one epoch training
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
        opt.log_dir = 'experiments/logs/linemod'
        opt.repeat_epoch = 20
    elif opt.dataset == 'custom':
        opt.num_objects = 1
        opt.num_points = 500
        opt.outf = 'trained_models/custom'
        opt.log_dir = 'experiments/logs/custom'
        opt.repeat_epoch = 1
    else:
        print('Unknown dataset')
        return

    estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects, num_rot_bins = opt.num_rot_bins)
    estimator.cuda()
    estimator = nn.DataParallel(estimator)
    refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects, num_rot_bins = opt.num_rot_bins)
    refiner.cuda()
    refiner = nn.DataParallel(refiner)

    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

    if opt.resume_refinenet != '':
        refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
        opt.refine_start = True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        #opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start, opt.num_rot_bins, opt.image_size, perform_profiling=opt.profile)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start, opt.num_rot_bins, perform_profiling=opt.profile)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start, opt.num_rot_bins, opt.image_size, perform_profiling=opt.profile)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start, opt.num_rot_bins, perform_profiling=opt.profile)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
    
    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_rot_bins)
    criterion_refine = Loss_refine(opt.num_rot_bins)

    best_test = np.Inf

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()

    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_loss_avg = 0.0
        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                
                if opt.profile:
                    print("starting training sample {0} {1}".format(i, datetime.now()))

                points, choose, img, front_r, rot_bins, front_orig, t, model_points, idx = data
                points, choose, img, front_r, rot_bins, front_orig, t, model_points, idx = Variable(points).cuda(), \
                                                                 Variable(choose).cuda(), \
                                                                 Variable(img).cuda(), \
                                                                 Variable(front_r).cuda(), \
                                                                 Variable(rot_bins).cuda(), \
                                                                 Variable(front_orig).cuda(), \
                                                                 Variable(t).cuda(), \
                                                                 Variable(model_points).cuda(), \
                                                                 Variable(idx).cuda()
                pred_front, pred_rot_bins, pred_t, pred_c, emb = estimator(img, points, choose, idx)

                if opt.profile:
                    print("finished forward pass {0} {1}".format(i, datetime.now()))

                loss, new_points, new_rot_bins, new_t, new_front_orig, new_front_r = criterion(pred_front, pred_rot_bins, pred_t, pred_c, front_r, rot_bins, front_orig, t, idx, model_points, points, opt.w, opt.refine_start)

                if opt.profile:
                    print("finished loss {0} {1}".format(i, datetime.now()))

                if opt.refine_start:
                    for ite in range(0, opt.iteration):
                        pred_front, pred_rot_bins, pred_t = refiner(new_points, emb, idx)
                        loss, new_points, new_rot_bins, new_t, new_front_orig, new_front_r = criterion_refine(pred_front, pred_rot_bins, pred_t, new_front_r, new_rot_bins, new_front_orig, new_t, idx, new_points)
                        loss.backward()
                else:
                    loss.backward()

                if opt.profile:
                    print("finished refiner steps {0} {1}".format(i, datetime.now()))

                train_loss_avg += loss.item()
                train_count += opt.batch_size

                logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_loss:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count, train_loss_avg))
                optimizer.step()
                optimizer.zero_grad()
                train_loss_avg = 0

                if opt.profile:
                    print("finished optimizer step {0} {1}".format(i, datetime.now()))

                if train_count != 0 and (train_count / opt.batch_size) % 1000 == 0:
                    if opt.refine_start:
                        torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
                    else:
                        torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

                if opt.profile:
                    print("finished training sample {0} {1}".format(i, datetime.now()))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))


        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_loss = 0.0
        test_count = 0
        estimator.eval()
        refiner.eval()

        for j, data in enumerate(testdataloader, 0):
            points, choose, img, front_r, rot_bins, front_orig, t, model_points, idx = data
            points, choose, img, front_r, rot_bins, front_orig, t, model_points, idx = Variable(points).cuda(), \
                                                                 Variable(choose).cuda(), \
                                                                 Variable(img).cuda(), \
                                                                 Variable(front_r).cuda(), \
                                                                 Variable(rot_bins).cuda(), \
                                                                 Variable(front_orig).cuda(), \
                                                                 Variable(t).cuda(), \
                                                                 Variable(model_points).cuda(), \
                                                                 Variable(idx).cuda()
            pred_front, pred_rot_bins, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            loss, new_points, new_rot_bins, new_t = criterion(pred_front, pred_rot_bins, pred_t, pred_c, front_r, rot_bins, front_orig, t, idx, model_points, points, opt.w, opt.refine_start)
                
            if opt.refine_start:
                for ite in range(0, opt.iteration):
                    pred_front, pred_rot_bins, pred_t = refiner(new_points, emb, idx)
                    loss, new_points, new_rot_bins, new_t = criterion_refine(pred_front, pred_rot_bins, pred_t, front_r, new_rot_bins, front_orig, new_t, idx, new_points)

            test_loss += loss.item()
            logger.info('Test time {0} Test Frame No.{1} loss:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, loss))

            test_count += 1

        test_loss = test_loss / test_count
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg Loss: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_loss))
        if test_loss <= best_test:
            best_test = test_loss
            if opt.refine_start:
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_loss))
            else:
                torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_loss))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        # if best_test < opt.decay_margin and not opt.decay_start:
        #     opt.decay_start = True
        #     opt.lr *= opt.lr_rate
        #     opt.w *= opt.w_rate
        #     optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

        if epoch >= opt.refine_epoch and not opt.refine_start:
            opt.refine_start = True
            #opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)

            if opt.dataset == 'ycb':
                dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start, opt.num_rot_bins, opt.image_size)
            elif opt.dataset == 'linemod':
                dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start, opt.num_rot_bins)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
            if opt.dataset == 'ycb':
                test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start, opt.num_rot_bins, opt.image_size)
            elif opt.dataset == 'linemod':
                test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start, opt.num_rot_bins)
            testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
            
            opt.sym_list = dataset.get_sym_list()
            opt.num_points_mesh = dataset.get_num_points_mesh()

            print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    main()
