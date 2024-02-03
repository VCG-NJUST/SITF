import torch
import numpy as np
from Pointfilter_Network_Architecture import ResPCPNet
from DataLoader1 import PointcloudPatchDataset,my_collate
from Pointfilter_Utils import parse_arguments
import os

def eval(opt):

    with open(os.path.join(opt.testset, 'test_real_noise.txt'), 'r') as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))

    if not os.path.exists(parameters.save_dir):
        os.makedirs(parameters.save_dir)

    for shape_id, shape_name in enumerate(shape_names):
        print(shape_name)
        original_noise_pts = np.load(os.path.join(opt.testset, shape_name + '.npy'))
        np.save(os.path.join(opt.save_dir, shape_name + '.npy'), original_noise_pts.astype('float32'))
        for eval_index in range(2):
            print(eval_index)
            test_dataset = PointcloudPatchDataset(
                root=opt.save_dir,
                ori_root=opt.testset,
                shape_name=shape_name,
                patch_radius=opt.patch_radius,
                train_state='evaluation')
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                collate_fn=my_collate,
                batch_size=opt.batchSize,
                num_workers=int(opt.workers))

            pointfilter_eval = ResPCPNet(num=eval_index).cuda()
            model_filename = os.path.join(opt.eval_dir, 'model_full_ae_5_'+str(eval_index)+ '.pth')
            checkpoint = torch.load(model_filename)
            pointfilter_eval.load_state_dict(checkpoint['state_dict'])
            pointfilter_eval.cuda()
            pointfilter_eval.eval()

            patch_radius = test_dataset.patch_radius_absolute
            pred_pts = np.empty((0, 3), dtype='float32')
            for batch_ind, data_tuple in enumerate(test_dataloader):
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
            np.save(os.path.join(opt.save_dir, shape_name +'.npy'),pred_pts.astype('float32'))



if __name__ == '__main__':

    parameters = parse_arguments()
    parameters.testset = 'D:/jk/PCN/testaver'
    #parameters.eval_dir = './Summary/pre_train_model/'
    parameters.batchSize = 128
    parameters.workers = 2
    parameters.save_dir = 'D:/jk/PCN/result/'
    parameters.eval_dir = 'D:/jk/PCN/Summary/gs'
    parameters.eval_iter_nums = 2
    parameters.patch_radius = 0.05
    eval(parameters)

