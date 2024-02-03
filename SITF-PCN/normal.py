import torch
import numpy as np
import numpy.linalg as LA
import os
import scipy.spatial as sp
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
def PCA(gt_patch):
    data_mean = np.mean(gt_patch, axis=0)
    data_patch = gt_patch - data_mean
    H = np.dot(data_patch.T, data_patch)
    vectors, values, vectors_t = LA.svd(H)
    return values,vectors
def Normal(root,shapes_list_file):
    with open(os.path.join(root,shapes_list_file)) as f:
        shape_names=f.readlines()
    shape_names=[x.strip() for x in shape_names]
    shape_names=list(filter(None,shape_names))
    for shape_ind,shape_name in enumerate(shape_names):
        normals = np.zeros(shape=(1, 3))
        gt=np.load(os.path.join(root,shape_name+'.npy'))
        gt_tree=sp.cKDTree(gt)
        for i in range(gt.shape[0]):
            index=gt_tree.query_ball_point(gt[i],r=0.03)
            #print(index)
            gt_patch=gt[index]
            W,V=PCA(gt_patch)
            normals=np.r_[normals,V[:,2].reshape(1,3)]
        normals=normals[1:]
        np.save(os.path.join(root,shape_name+'_normal.npy'),normals)
#Normal(root='E:/deep learning/pid_pf_point/noise')
def one_normal(pts):
    normals = np.zeros(shape=(1, 3))
    gt_tree = sp.cKDTree(pts)
    for i in range(pts.shape[0]):
        index = gt_tree.query_ball_point(pts[i], r=0.03)
        # print(index)
        gt_patch = pts[index]
        W, V = PCA(gt_patch)
        normals = np.r_[normals, V[:, 2].reshape(1, 3)]
    normals = normals[1:]
    return normals