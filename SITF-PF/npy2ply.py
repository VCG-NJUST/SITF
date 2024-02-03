import scipy.spatial as sp
import numpy as np
import torch
import os
#from Customer_Module.chamfer_distance.dist_chamfer import chamferDist
from plyfile import PlyData, PlyElement
#nnd = chamferDist()

def npy2ply(filename):

    with open(os.path.join(filename, 'gs_noise_1.txt'), 'r') as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))
    for shape_id, shape_name in enumerate(shape_names):
        pts = np.load(os.path.join(filename, shape_name+ '.npy'))
        vertex = [tuple(item) for item in pts]
        vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(os.path.join(filename, shape_name+ '.ply'))


   # Eval_With_Charmfer_Distance()
   # Eval_With_Mean_Square_Error()