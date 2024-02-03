
from sklearn.decomposition import PCA
import math
import torch
import argparse
import time
import numpy as np
import scipy.spatial as sp
##########################Parameters########################
#
#
#
#
###############################################################

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_arguments():
    parser = argparse.ArgumentParser()
    # naming / file handling
    parser.add_argument('--name', type=str, default='pcdenoising', help='training run name')
    parser.add_argument('--network_model_dir', type=str, default='./Summary/Models/Train',help='output folder (trained models)')
    parser.add_argument('--trainset', type=str, default='./Dataset/Train', help='training set file name')
    parser.add_argument('--testset', type=str, default='D:/jk/PCN/test', help='testing set file name')
    parser.add_argument('--denoiseset', type=str, default='D:/jk/PCN/noise', help='denoising set file name')
    parser.add_argument('--save_dir', type=str, default='D:/jk/PCN/result', help='')
    parser.add_argument('--denoise_dir', type=str, default='D:/jk/PCN/denoised', help='')
    parser.add_argument('--origin', type=str, default='D:/jk/PCN/origin', help='')
    parser.add_argument('--summary_dir', type=str, default='./Summary/Models/Train/logs', help='')

    # training parameters
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--manualSeed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--patch_per_shape', type=int, default=8000, help='')
    parser.add_argument('--patch_radius', type=float, default=0.05, help='')

    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--model_interval', type=int, default=1, metavar='N', help='how many batches to wait before logging training status')

    # others parameters
    parser.add_argument('--resume', type=str, default='', help='refine model at this path')
    parser.add_argument('--support_multiple', type=float, default=4.0, help='the multiple of support radius')
    parser.add_argument('--support_angle', type=int, default=15, help='')
    parser.add_argument('--gt_normal_mode', type=str, default='nearest', help='')
    parser.add_argument('--repulsion_alpha', type=float, default='0.97', help='')

    # evaluation parameters
    parser.add_argument('--eval_dir', type=str, default='D:/jk/PCN/Summary/Train', help='')
    parser.add_argument('--eval_iter_nums', type=int, default=2, help='')
    parser.add_argument('--outputs', type=str, nargs='+', default=['clean_points'], help='output of the network')
    parser.add_argument('--use_point_stn', type=int,
                        default=True, help='use point spatial transformer')
    parser.add_argument('--use_feat_stn', type=int,
                        default=None, help='use feature spatial transformer')
    parser.add_argument('--points_per_patch', type=int,
                        default=500, help='max. number of points per patch')  # 50

    return parser.parse_args()

###################Pre-Processing Tools########################
#
#
#
#
###############################################################


def get_principle_dirs(pts):

    pts_pca = PCA(n_components=3)
    pts_pca.fit(pts)
    principle_dirs = pts_pca.components_
    principle_dirs /= np.linalg.norm(principle_dirs, 2, axis=0)

    return principle_dirs


def pca_alignment(pts, random_flag=False):

    pca_dirs = get_principle_dirs(pts)

    if random_flag:

        pca_dirs *= np.random.choice([-1, 1], 1)

    rotate_1 = compute_roatation_matrix(pca_dirs[2], [0, 0, 1], pca_dirs[1])
    pca_dirs = np.array(rotate_1 * pca_dirs.T).T
    rotate_2 = compute_roatation_matrix(pca_dirs[1], [1, 0, 0], pca_dirs[2])
    pts = np.array(rotate_2 * rotate_1 * np.matrix(pts.T)).T

    inv_rotation = np.array(np.linalg.inv(rotate_2 * rotate_1))

    return pts, inv_rotation

def compute_roatation_matrix(sour_vec, dest_vec, sour_vertical_vec=None):
    # http://immersivemath.com/forum/question/rotation-matrix-from-one-vector-to-another/
    if np.linalg.norm(np.cross(sour_vec, dest_vec), 2) == 0 or np.abs(np.dot(sour_vec, dest_vec)) >= 1.0:
        if np.dot(sour_vec, dest_vec) < 0:
            return rotation_matrix(sour_vertical_vec, np.pi)
        return np.identity(3)
    alpha = np.arccos(np.dot(sour_vec, dest_vec))
    a = np.cross(sour_vec, dest_vec) / np.linalg.norm(np.cross(sour_vec, dest_vec), 2)
    c = np.cos(alpha)
    s = np.sin(alpha)
    R1 = [a[0] * a[0] * (1.0 - c) + c,
          a[0] * a[1] * (1.0 - c) - s * a[2],
          a[0] * a[2] * (1.0 - c) + s * a[1]]

    R2 = [a[0] * a[1] * (1.0 - c) + s * a[2],
          a[1] * a[1] * (1.0 - c) + c,
          a[1] * a[2] * (1.0 - c) - s * a[0]]

    R3 = [a[0] * a[2] * (1.0 - c) - s * a[1],
          a[1] * a[2] * (1.0 - c) + s * a[0],
          a[2] * a[2] * (1.0 - c) + c]

    R = np.matrix([R1, R2, R3])

    return R


def rotation_matrix(axis, theta):

    # Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.

    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.matrix(np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]]))


def patch_sampling(patch_pts, sample_num):

    if patch_pts.shape[0] > sample_num:

        sample_index = np.random.choice(range(patch_pts.shape[0]), sample_num, replace=False)

    else:

        sample_index = np.random.choice(range(patch_pts.shape[0]), sample_num)

    return sample_index

##########################Network Tools########################
#
#
#
#
###############################################################

def adjust_learning_rate(optimizer, epoch, opt):

    lr_shceduler(optimizer, epoch, opt.lr)
'''
def lr_shceduler(optimizer, epoch, init_lr):

    if epoch > 12:
        init_lr *= 0.5e-3
    elif epoch > 10:
        init_lr *= 1e-3
    elif epoch > 7:
        init_lr *= 1e-2
    elif epoch > 4:
        init_lr *= 1e-1
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr
'''
def lr_shceduler(optimizer, epoch, init_lr):

    if epoch > 36:
        init_lr *= 0.5e-3
    elif epoch > 32:
        init_lr *= 1e-3
    elif epoch > 24:
        init_lr *= 1e-2
    elif epoch > 16:
        init_lr *= 1e-1
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr


################################Ablation Study of Different Loss ###############################

def compute_original_1_loss(pts_pred, gt_patch_pts, gt_patch_normals, support_radius, alpha):

    pts_pred = pts_pred.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    dist_square = (pts_pred - gt_patch_pts).pow(2).sum(2)

    # avoid divided by zero
    weight = torch.exp(-1 * dist_square / (support_radius ** 2)) + 1e-12
    weight = weight / weight.sum(1, keepdim=True)

    # key loss
    project_dist = ((pts_pred - gt_patch_pts) * gt_patch_normals).sum(2)
    imls_dist = torch.abs((project_dist * weight).sum(1))

    # repulsion loss
    max_dist = torch.max(dist_square, 1)[0]

    # final loss
    dist = torch.mean((alpha * imls_dist) + (1 - alpha) * max_dist)

    return dist

def compute_original_2_loss(pred_point, gt_patch_pts, gt_patch_normals, support_radius, support_angle, alpha):

    # Compute Spatial Weighted Function
    pred_point = pred_point.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    dist_square = ((pred_point - gt_patch_pts) ** 2).sum(2)
    weight_theta = torch.exp(-1 * dist_square / (support_radius ** 2))

    ############# Get The Nearest Normal For Predicted Point #############
    nearest_idx = torch.argmin(dist_square, dim=1)
    pred_point_normal = torch.cat([gt_patch_normals[i, index, :] for i, index in enumerate(nearest_idx)])
    pred_point_normal = pred_point_normal.view(-1, 3)
    pred_point_normal = pred_point_normal.unsqueeze(1)
    pred_point_normal = pred_point_normal.repeat(1, gt_patch_normals.size(1), 1)
    ############# Get The Nearest Normal For Predicted Point #############

    # Compute Normal Weighted Function
    normal_proj_dist = (pred_point_normal * gt_patch_normals).sum(2)
    weight_phi = torch.exp(-1 * ((1 - normal_proj_dist) / (1 - np.cos(support_angle)))**2)

    # # avoid divided by zero
    weight = weight_theta * weight_phi + 1e-12
    weight = weight / weight.sum(1, keepdim=True)

    # key loss
    project_dist = torch.sqrt(dist_square)
    imls_dist = (project_dist * weight).sum(1)

    # repulsion loss
    max_dist = torch.max(dist_square, 1)[0]

    # final loss
    dist = torch.mean((alpha * imls_dist) + (1 - alpha) * max_dist)

    return dist

def compute_original_3_loss(pts_pred, gt_patch_pts):
    # PointCleanNet Loss
    pts_pred = pts_pred.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    m = (pts_pred - gt_patch_pts).pow(2).sum(2)
    min_dist = torch.min(m, 1)[0]
    max_dist = torch.max(m, 1)[0]
    alpha = 0.99
    dist = torch.mean((alpha * min_dist) + (1 - alpha) * max_dist)
    #print('min_dist: %f max_dist: %f' % (alpha * torch.mean(min_dist).item(), (1 - alpha) * torch.mean(max_dist).item()))
    return dist * 100


################################Ablation Study of Different Loss ###############################

def compute_bilateral_loss_with_repulsion(pred_point, gt_patch_pts, gt_patch_normals, support_radius, support_angle, alpha):
    #noise_dist = compute_original_3_loss(pred_point, origin_noise)
    # Our Loss
    pred_point = pred_point.unsqueeze(1).repeat(1, gt_patch_pts.size(1), 1)
    dist_square = ((pred_point - gt_patch_pts) ** 2).sum(2)
    weight_theta = torch.exp(-1 * dist_square / (support_radius ** 2))
    #print(dist_square.shape)
    nearest_idx = torch.min(dist_square, dim=1)[1]
    pred_point_normal = torch.cat([gt_patch_normals[i, index, :] for i, index in enumerate(nearest_idx)])
    pred_point_normal = pred_point_normal.view(-1, 3)
    pred_point_normal = pred_point_normal.unsqueeze(1)
    pred_point_normal = pred_point_normal.repeat(1, gt_patch_normals.size(1), 1)

    normal_proj_dist = (pred_point_normal * gt_patch_normals).sum(2)
    weight_phi = torch.exp(-1 * ((1 - normal_proj_dist) / (1 - np.cos(support_angle)))**2)

    # # avoid divided by zero
    weight = weight_theta * weight_phi + 1e-12
    weight = weight / weight.sum(1, keepdim=True)

    # key loss
    project_dist = torch.abs(((pred_point - gt_patch_pts) * gt_patch_normals).sum(2))
    imls_dist = (project_dist * weight).sum(1)

    # repulsion loss
    max_dist = torch.max(dist_square, 1)[0]

    # final loss
    dist = torch.mean((alpha * imls_dist) + (1 -alpha) * max_dist)

    return dist


# quaternion a + bi + cj + dk should be given in the form [a,b,c,d]
def batch_quat_to_rotmat(q, out=None):

    batchsize = q.size(0)

    if out is None:
        out = torch.FloatTensor(batchsize, 3, 3)

    # 2 / squared quaternion 2-norm
    s = 2/torch.sum(q.pow(2), 1)

    # coefficients of the Hamilton product of the quaternion with itself
    h = torch.bmm(q.unsqueeze(2), q.unsqueeze(1))

    out[:, 0, 0] = 1 - (h[:, 2, 2] + h[:, 3, 3]).mul(s)
    out[:, 0, 1] = (h[:, 1, 2] - h[:, 3, 0]).mul(s)
    out[:, 0, 2] = (h[:, 1, 3] + h[:, 2, 0]).mul(s)

    out[:, 1, 0] = (h[:, 1, 2] + h[:, 3, 0]).mul(s)
    out[:, 1, 1] = 1 - (h[:, 1, 1] + h[:, 3, 3]).mul(s)
    out[:, 1, 2] = (h[:, 2, 3] - h[:, 1, 0]).mul(s)

    out[:, 2, 0] = (h[:, 1, 3] - h[:, 2, 0]).mul(s)
    out[:, 2, 1] = (h[:, 2, 3] + h[:, 1, 0]).mul(s)
    out[:, 2, 2] = 1 - (h[:, 1, 1] + h[:, 2, 2]).mul(s)

    return out

def cos_angle(v1, v2):

    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)
