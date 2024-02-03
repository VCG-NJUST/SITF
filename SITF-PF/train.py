
# coding=utf-8

from __future__ import print_function
from tensorboardX import SummaryWriter
from Pointfilter_Network_Architecture import pointfilternet
from DataLoader1 import PointcloudPatchDataset, RandomPointcloudPatchSampler, my_collate
from Pointfilter_Utils  import parse_arguments, adjust_learning_rate, compute_bilateral_loss_with_repulsion,compute_original_3_loss

from npy2ply import npy2ply
import os
import numpy as np
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn
from addnoise import addgaussiannoise,adduniformnoise
from denoise import denoise,denoise1
from normal import Normal

torch.backends.cudnn.benchmark = True

def train(opt):
    if not os.path.exists(opt.summary_dir):
        os.makedirs(opt.summary_dir)
    if not os.path.exists(opt.network_model_dir):
        os.makedirs(opt.network_model_dir)
    print("Random Seed: ", opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    '''
    #model_filename = 'D:/jk/self-supervision/Summary/pre_train_model/power gs/model_full_ae.pth'
    model_filename = 'D:/jk/self-supervision/Summary/Train/model_full_ae_0.pth'
    checkpoint = torch.load(model_filename)
    pretrain_dict = checkpoint['state_dict']
    model_dict = denoisenet.state_dict() 
    #pretrain_dict={k:v for k,v in pretrain_dict.items() if k in model_dict}
    #pretrain_dict.pop('decoder.fc3.weight')
    #pretrain_dict.pop('decoder.fc3.bias')
    model_dict.update(pretrain_dict)
    denoisenet.load_state_dict(model_dict)
    '''


    # z0 = torch.from_numpy(np.zeros([parameters.batchSize,3]))
    # pts_list = train_datasampler.list_index
    for i in range(1,6):
        #npy2ply(opt.trainset)else:
        Normal(opt.trainset,shapes_list_file='gs_noise_1.txt')
        addgaussiannoise()
        #adduniformnoise()
        for num in range(3):
            denoisenet = pointfilternet(num=num).cuda()
            optimizer = optim.Adam(
                denoisenet.parameters()
                , lr=opt.lr
                , betas=(0.9, 0.999)
                , eps=10e-8
            )
            if i>0:
                model_filename =os.path.join('D:/jk/self-supervision/Summary/Train', 'model_full_ae_'+str(i-1)+'_'+str(num)+'.pth')
                checkpoint = torch.load(model_filename)
                pretrain_dict = checkpoint['state_dict']
                model_dict = denoisenet.state_dict()
                #pretrain_dict={k:v for k,v in pretrain_dict.items() if k in model_dict}
                # pretrain_dict.pop('decoder.fc3.weight')
                # pretrain_dict.pop('decoder.fc3.bias')
                model_dict.update(pretrain_dict)
                denoisenet.load_state_dict(model_dict)

            train_writer = SummaryWriter(opt.summary_dir)

            # optionally resume from a checkpoint
            if opt.resume:
                if os.path.isfile(opt.resume):
                    print("=> loading checkpoint '{}'".format(opt.resume))
                    checkpoint = torch.load(opt.resume)
                    opt.start_epoch = checkpoint['epoch']
                    denoisenet.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))
                else:
                    print("=> no checkpoint found at '{}'".format(opt.resume))
            train_dataset = PointcloudPatchDataset(
                root=opt.trainset,
                ori_root=opt.origin,
                root_noise=opt.denoiseset,
                shapes_list_file='train_gs_1.txt',
                patch_radius=0.05,
                seed=opt.manualSeed,
                train_state='train',
                num=num
                )
            train_datasampler = RandomPointcloudPatchSampler(
                train_dataset,
                patches_per_shape=8000,
                seed=opt.manualSeed,
                identical_epochs=False)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                sampler=train_datasampler,
                shuffle=(train_datasampler is None),
                collate_fn=my_collate,
                batch_size=opt.batchSize,
                num_workers=int(opt.workers),
                pin_memory=True)
            num_batch = len(train_dataloader)
            for epoch in range(opt.start_epoch, opt.nepoch):
                adjust_learning_rate(optimizer, epoch, opt)
                print('lr is %.10f' % (optimizer.param_groups[0]['lr']))
                num_batch_current = 0
                patch_radius = train_dataset.patch_radius_absolute
                pred_pts = np.empty((0, 3), dtype='float32')

                for batch_ind, data_tuple in enumerate(train_dataloader):
                    denoisenet.train()
                    optimizer.zero_grad()
                    noise_patch, gt_patch,gt_normal, support_radius,origin_noise = data_tuple
                    noise_patch = noise_patch.float().cuda(non_blocking=True)
                    gt_patch = gt_patch.float().cuda(non_blocking=True)
                    gt_normal = gt_normal.float().cuda(non_blocking=True)
                    origin_noise = origin_noise.float().cuda(non_blocking=True)
                    support_radius = opt.support_multiple * support_radius
                    support_radius = support_radius.float().cuda(non_blocking=True)
                    support_angle = (opt.support_angle / 360) * 2 * np.pi
                    noise_patch = noise_patch.transpose(2, 1).contiguous()
                    origin_noise = origin_noise.transpose(2, 1).contiguous()
                    pred_pts = denoisenet(noise_patch,origin_noise)
                    #                else:
                    #                pred_pts = denoisenet(noise_patch)
                    #                pred_point = pred_pts.unsqueeze(2).repeat(1, 1, noise_patch.shape[2])
                    #                error = noise_patch - pred_point
                    #                errors.append(error)
                    # print(denoisenet.error.shape[0])
                    loss = 100 * compute_bilateral_loss_with_repulsion(pred_pts, gt_patch, gt_normal,
                                                                       support_radius, support_angle, opt.repulsion_alpha)

                    #loss = compute_original_3_loss(pred_pts, gt_patch, opt.repulsion_alpha)
                    loss.backward()
                    optimizer.step()
                    print('[%d: %d/%d] train loss: %f\n' % (epoch, batch_ind, num_batch, loss.item()))
                    train_writer.add_scalar('loss', loss.data.item(), epoch * num_batch + batch_ind)
                checpoint_state = {
                    'epoch': epoch + 1,
                    'state_dict': denoisenet.state_dict(),
                    'optimizer': optimizer.state_dict()}

                if epoch == (opt.nepoch - 1):

                    torch.save(checpoint_state, '%s/model_full_ae_%d_%d.pth' % (opt.network_model_dir,i,num))

                #if epoch % opt.model_interval == 0:

                    #torch.save(checpoint_state, '%s/model_full_ae_%d.pth' % (opt.network_model_dir, epoch))
            else:
                if num<2:
                    denoise1(opt,i,num)
        else:
            denoise(opt, i)



if __name__ == '__main__':
    parameters = parse_arguments()
    parameters.trainset = 'D:/jk/self-supervision/train'
    parameters.summary_dir = 'D:/jk/self-supervision/Summary/Train/logs'
    parameters.network_model_dir = 'D:/jk/self-supervision/Summary/Train'
    parameters.batchSize = 256
    parameters.lr = 1e-4
    parameters.workers = 4
    parameters.nepoch = 50
    print(parameters)
    train(parameters)
