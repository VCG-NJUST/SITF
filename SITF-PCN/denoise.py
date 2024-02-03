import torch
import numpy as np
from Pointfilter_Network_Architecture import ResPCPNet
from DataLoader1 import PointcloudPatchDataset,my_collate
from Pointfilter_Utils import parse_arguments
import os

def denoise(opt,i):

    with open(os.path.join(opt.denoiseset, 'gs_noise_1.txt'), 'r') as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))

    with torch.no_grad():
        for shape_id, shape_name in enumerate(shape_names):
            print(shape_name)
            original_noise_pts = np.load(os.path.join(opt.denoiseset, shape_name + '.npy'))
            np.save(os.path.join(opt.denoise_dir, shape_name + '.npy'), original_noise_pts.astype('float32'))
            for iter_index in range(2):
                print(iter_index)
                denoise_dataset = PointcloudPatchDataset(
                    root=opt.denoise_dir,
                    ori_root=opt.denoiseset,
                    shape_name=shape_name,
                    patch_radius=opt.patch_radius,
                    train_state='evaluation')
                denoise_dataloader = torch.utils.data.DataLoader(
                    denoise_dataset,
                    collate_fn=my_collate,
                    batch_size=opt.batchSize,
                    num_workers=int(opt.workers))

                pointfilter_eval = ResPCPNet(num=0).cuda()
                model_filename = os.path.join(opt.eval_dir, 'model_full_ae_'+str(i)+'_'+str(0)+'.pth')
                checkpoint = torch.load(model_filename)
                pointfilter_eval.load_state_dict(checkpoint['state_dict'])
                pointfilter_eval.cuda()
                pointfilter_eval.eval()

                patch_radius = denoise_dataset.patch_radius_absolute
                pred_pts = np.empty((0, 3), dtype='float32')
                for batch_ind, data_tuple in enumerate(denoise_dataloader):
                    noise_patch, noise_disp, origin_noise = data_tuple
                    noise_patch = noise_patch.float().cuda()
                    origin_noise = origin_noise.float().cuda()
                    noise_patch = noise_patch.transpose(2, 1).contiguous()
                    origin_noise = origin_noise.transpose(2, 1).contiguous()
                    predict,trans = pointfilter_eval(noise_patch, origin_noise)
                    predict = predict.unsqueeze(1)
                    predict = torch.bmm(predict,trans.transpose(2,1)).squeeze(1)
                    pred_pts = np.append(pred_pts,
                                         np.squeeze(predict.data.cpu().numpy()) * patch_radius + noise_disp.numpy(),
                                         axis=0)

                np.save(os.path.join(opt.denoise_dir, shape_name+'.npy'), pred_pts.astype('float32'))
            np.save(os.path.join(opt.trainset, shape_name + '.npy'), pred_pts.astype('float32'))
            #np.save(os.path.join(opt.origin, shape_name + '.npy'), pred_pts.astype('float32'))

def denoise1(opt,i,num):

    with open(os.path.join(opt.trainset, 'gs_2.txt'), 'r') as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))

    with torch.no_grad():
        for shape_id, shape_name in enumerate(shape_names):
            print(shape_name)

            denoise_dataset = PointcloudPatchDataset(
                root=opt.trainset,
                ori_root=opt.origin,
                shape_name=shape_name,
                patch_radius=opt.patch_radius,
                train_state='evaluation')
            denoise_dataloader = torch.utils.data.DataLoader(
                denoise_dataset,
                collate_fn=my_collate,
                batch_size=opt.batchSize,
                num_workers=int(opt.workers))

            pointfilter_eval = ResPCPNet(num=num).cuda()
            model_filename = os.path.join(opt.eval_dir, 'model_full_ae_'+str(i)+'_'+str(num)+'.pth')
            checkpoint = torch.load(model_filename)
            pointfilter_eval.load_state_dict(checkpoint['state_dict'])
            pointfilter_eval.cuda()
            pointfilter_eval.eval()

            patch_radius = denoise_dataset.patch_radius_absolute
            pred_pts = np.empty((0, 3), dtype='float32')
            for batch_ind, data_tuple in enumerate(denoise_dataloader):
                noise_patch, noise_disp, origin_noise = data_tuple
                noise_patch = noise_patch.float().cuda()
                origin_noise = origin_noise.float().cuda()
                noise_patch = noise_patch.transpose(2, 1).contiguous()
                origin_noise = origin_noise.transpose(2, 1).contiguous()
                predict,trans = pointfilter_eval(noise_patch, origin_noise)
                predict = predict.unsqueeze(1)
                predict = torch.bmm(predict, trans.transpose(2, 1)).squeeze(1)
                pred_pts = np.append(pred_pts,
                                     np.squeeze(predict.data.cpu().numpy()) * patch_radius + noise_disp.numpy(),
                                     axis=0)

            np.save(os.path.join(opt.trainset, shape_name+'.npy'), pred_pts.astype('float32'))

def denoise_1(num,shape_name,target):
    root='D:/jk/self-supervision/denoised'
    opt=parse_arguments()
    with torch.no_grad():
        np.save(os.path.join(root, shape_name + '.npy'), target.astype('float32'))
        for iter_index in range(2):
            print(iter_index)
            denoise_dataset = PointcloudPatchDataset(
                root=root,
                shape_name=shape_name,
                patch_radius=0.05,
                train_state='evaluation')
            denoise_dataloader = torch.utils.data.DataLoader(
                denoise_dataset,
                collate_fn=my_collate,
                batch_size=opt.batchSize,
                num_workers=int(opt.workers))

            pointfilter_eval = pid_pointfilternet().cuda()
            model_filename = os.path.join(opt.eval_dir, 'model_full_ae_' + str(num-1) + '.pth')
            checkpoint = torch.load(model_filename)
            pointfilter_eval.load_state_dict(checkpoint['state_dict'])
            pointfilter_eval.cuda()
            pointfilter_eval.eval()

            patch_radius = denoise_dataset.patch_radius_absolute
            pred_pts = np.empty((0, 3), dtype='float32')
            for batch_ind, data_tuple in enumerate(denoise_dataloader):
                noise_patch, noise_inv, noise_disp = data_tuple
                noise_patch = noise_patch.float().cuda()
                noise_inv = noise_inv.float().cuda()
                noise_patch = noise_patch.transpose(2, 1).contiguous()
                predict = pointfilter_eval(noise_patch)
                predict = predict.unsqueeze(2)
                predict = torch.bmm(noise_inv, predict).squeeze(1)
                pred_pts = np.append(pred_pts,
                                     np.squeeze(predict.data.cpu().numpy()) * patch_radius + noise_disp.numpy(),
                                     axis=0)
            np.save(os.path.join(root, shape_name + '.npy'), pred_pts.astype('float32'))
        return pred_pts

if __name__ == '__main__':

    parameters = parse_arguments()
    parameters.denoiseset = 'D:/jk/PCN/noise'
    #parameters.eval_dir = './Summary/pre_train_model/'
    parameters.batchSize = 128
    parameters.workers = 2
    parameters.trainset = 'D:/jk/PCN/train'
    parameters.denoise_dir= 'D:/jk/PCN/denoised'
    parameters.eval_dir='D:/jk/PCN/Summary/Train'
    parameters.eval_iter_nums = 2
    parameters.patch_radius = 0.05
    denoise(parameters,5)

