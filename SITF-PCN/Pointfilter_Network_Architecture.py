from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import Pointfilter_Utils as utils


class STN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max', quaternion =False):
        super(STN, self).__init__()
        self.quaternion = quaternion
        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        if not quaternion:
            self.fc3 = nn.Linear(256, self.dim*self.dim)
        else:
            self.fc3 = nn.Linear(256, 4)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            if x.is_cuda:
                x_scales = Variable(torch.cuda.FloatTensor(x.size(0), 1024*self.num_scales, 1))
            else:
                x_scales = Variable(torch.FloatTensor(x.size(0), 1024*self.num_scales, 1))
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        if not self.quaternion:
            iden = Variable(torch.from_numpy(np.identity(self.dim, 'float32')).clone()).view(1, self.dim*self.dim).repeat(batchsize, 1)

            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden
            x = x.view(-1, self.dim, self.dim)
        else:
            # add identity quaternion (so the network can output 0 to leave the point cloud identical)
            iden = Variable(torch.FloatTensor([1, 0, 0, 0]))
            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden

            # convert quaternion to rotation matrix
            if x.is_cuda:
                trans = Variable(torch.cuda.FloatTensor(batchsize, 3, 3))
            else:
                trans = Variable(torch.FloatTensor(batchsize, 3, 3))
            x = utils.batch_quat_to_rotmat(x, trans)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_scales=1, num_points=500, use_point_stn=True, use_feat_stn=True, sym_op='max', get_pointfvals=False, point_tuple=1):
        super(PointNetfeat, self).__init__()
        self.num_points = num_points
        self.num_scales = num_scales
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.sym_op = sym_op
        self.get_pointfvals = get_pointfvals
        self.point_tuple = point_tuple

        if self.use_point_stn:
            # self.stn1 = STN(num_scales=self.num_scales, num_points=num_points, dim=3, sym_op=self.sym_op)
            self.stn1 = STN(num_scales=self.num_scales, num_points=num_points*self.point_tuple, dim=3, sym_op=self.sym_op, quaternion = True)

        if self.use_feat_stn:
            self.stn2 = STN(num_scales=self.num_scales, num_points=num_points, dim=64, sym_op=self.sym_op)

        self.conv0a = torch.nn.Conv1d(3*self.point_tuple, 64, 1)
        self.conv0b = torch.nn.Conv1d(64, 64, 1)

        # TODO remove
        # self.conv0c = torch.nn.Conv1d(64, 64, 1)
        # self.bn0c = nn.BatchNorm1d(64)
        # self.conv1b = torch.nn.Conv1d(64, 64, 1)
        # self.bn1b = nn.BatchNorm1d(64)


        self.bn0a = nn.BatchNorm1d(64)
        self.bn0b = nn.BatchNorm1d(64)
        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if self.num_scales > 1:
            self.conv4 = torch.nn.Conv1d(1024, 1024*self.num_scales, 1)
            self.bn4 = nn.BatchNorm1d(1024*self.num_scales)

        if self.sym_op == 'max':
            self.mp1 = torch.nn.MaxPool1d(num_points)
        elif self.sym_op == 'sum':
            self.mp1 = None
        else:
            raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

    def forward(self, x):

        # input transform
        if self.use_point_stn:
            # from tuples to list of single points
            x = x.view(x.size(0), 3, -1)
            trans = self.stn1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            x = x.contiguous().view(x.size(0), 3*self.point_tuple, -1)
        else:
            trans = None

        # mlp (64,64)
        x = F.relu(self.bn0a(self.conv0a(x)))
        x = F.relu(self.bn0b(self.conv0b(x)))
        # TODO remove
        #x = F.relu(self.bn0c(self.conv0c(x)))

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)
        else:
            trans2 = None

        # mlp (64,128,1024)
        x = F.relu(self.bn1(self.conv1(x)))
        # TODO remove
        #x = F.relu(self.bn1b(self.conv1b(x)))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # mlp (1024,1024*num_scales)
        if self.num_scales > 1:
            x = self.bn4(self.conv4(F.relu(x)))

        if self.get_pointfvals:
            pointfvals = x
        else:
            pointfvals = None # so the intermediate result can be forgotten if it is not needed

        # symmetric max operation over all points
        if self.num_scales == 1:
            if self.sym_op == 'max':
                x = self.mp1(x)
            elif self.sym_op == 'sum':
                x = torch.sum(x, 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

        else:
            if x.is_cuda:
                x_scales = Variable(torch.cuda.FloatTensor(x.size(0), 1024*self.num_scales**2, 1))
            else:
                x_scales = Variable(torch.FloatTensor(x.size(0), 1024*self.num_scales**2, 1))
            if self.sym_op == 'max':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*1024:(s+1)*self.num_scales*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            elif self.sym_op == 'sum':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*1024:(s+1)*self.num_scales*1024, :] = torch.sum(x[:, :, s*self.num_points:(s+1)*self.num_points], 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
            x = x_scales

        x = x.view(-1, 1024*self.num_scales**2)

        return x, trans, trans2, pointfvals


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, conv = False):
        super(BasicBlock, self).__init__()
        if conv:
            self.l1 = torch.nn.Conv1d(in_planes, planes, 1)
            self.l2 = torch.nn.Conv1d(planes, planes, 1)
        else:
            self.l1 = nn.Linear(in_planes,planes)
            self.l2 = nn.Linear(planes, planes)

        stdv = 0.001 # for working small initialisation
        # self.l1.weight.data.uniform_(-stdv, stdv)

        self.l1.weight.data.uniform_(-stdv, stdv)
        self.l2.weight.data.uniform_(-stdv, stdv)
        self.l1.bias.data.uniform_(-stdv, stdv)
        self.l2.bias.data.uniform_(-stdv, stdv)

        self.bn1 = nn.BatchNorm1d(planes, momentum = 0.01)
        self.shortcut = nn.Sequential()
        if in_planes != planes:
            if conv:
                self.l0 = nn.Conv1d(in_planes, planes, 1)
            else:
                self.l0 = nn.Linear(in_planes, planes)

            self.l0.weight.data.uniform_(-stdv, stdv)
            self.l0.bias.data.uniform_(-stdv, stdv)

            self.shortcut = nn.Sequential(self.l0,nn.BatchNorm1d(planes))
        self.bn2 = nn.BatchNorm1d(planes, momentum = 0.01)

    def forward(self, x):
            out = F.relu(self.bn1(self.l1(x)))
            out = self.bn2(self.l2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


class ResSTN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max', quaternion =False):
        super(ResSTN, self).__init__()
        self.quaternion = quaternion
        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.b1 = BasicBlock(self.dim, 64, conv = True)
        self.b2 = BasicBlock(64, 128, conv = True)
        self.b3 = BasicBlock(128, 1024, conv = True)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.bfc1 = BasicBlock(1024, 512)
        self.bfc2 = BasicBlock(512, 256)
        if not quaternion:
            self.bfc3 = BasicBlock(256, self.dim*self.dim)
        else:
            self.bfc3 = BasicBlock(256, 4)




    def forward(self, x):
        batchsize = x.size()[0]
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        # symmetric operation over all points
        x = self.mp1(x)

        x = x.view(-1, 1024*self.num_scales)

        x =self.bfc1(x)
        x = self.bfc2(x)
        x = self.bfc3(x)

        if not self.quaternion:
            iden = Variable(torch.from_numpy(np.identity(self.dim, 'float32')).clone()).view(1, self.dim*self.dim).repeat(batchsize, 1)

            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden
            x = x.view(-1, self.dim, self.dim)
        else:
            # add identity quaternion (so the network can output 0 to leave the point cloud identical)
            iden = Variable(torch.FloatTensor([1, 0, 0, 0]))
            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden

            # convert quaternion to rotation matrix
            if x.is_cuda:
                trans = Variable(torch.cuda.FloatTensor(batchsize, 3, 3))
            else:
                trans = Variable(torch.FloatTensor(batchsize, 3, 3))
            x = utils.batch_quat_to_rotmat(x, trans)
        return x

class ResPointNetfeat(nn.Module):
    def __init__(self, num_scales=1, num_points=500, use_point_stn=True, use_feat_stn=True, sym_op='max', get_pointfvals=False, point_tuple=1):
        super(ResPointNetfeat, self).__init__()
        self.num_points = num_points
        self.num_scales = num_scales
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.sym_op = sym_op
        self.get_pointfvals = get_pointfvals
        self.point_tuple = point_tuple

        if self.use_point_stn:
            # self.stn1 = STN(num_scales=self.num_scales, num_points=num_points, dim=3, sym_op=self.sym_op)
            self.stn1 = ResSTN(num_scales=self.num_scales, num_points=num_points*self.point_tuple, dim=3, sym_op=self.sym_op, quaternion=True)

        if self.use_feat_stn:
            self.stn2 = ResSTN(num_scales=self.num_scales, num_points=num_points, dim=64, sym_op=self.sym_op)

        self.b0a = BasicBlock(3*self.point_tuple, 64, conv = True)
        self.b0b = BasicBlock(64, 64, conv=True)

        self.b1 = BasicBlock(64, 64, conv = True)
        self.b2 = BasicBlock(64, 128, conv = True)


    def forward(self, x):

        # input transform
        if self.use_point_stn:
            # from tuples to list of single points
            x = x.view(x.size(0), 3, -1)
            trans = self.stn1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            x = x.contiguous().view(x.size(0), 3*self.point_tuple, -1)
        else:
            trans = None

        # mlp (64,64)
        x = self.b0a(x)
        x = self.b0b(x)

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)
        else:
            trans2 = None

        # mlp (64,128)
        x = self.b1(x)
        x = self.b2(x)
        return x, trans, trans2




def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def transformer_neighbors(x, feature, k=20, idx=None):
    '''
        input: x, [B,3,N]
               feature, [B,C,N]
        output: neighbor_x, [B,6,N,K]
                neighbor_feat, [B,2C,N,k]
    '''
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_x = x.view(batch_size * num_points, -1)[idx, :]
    neighbor_x = neighbor_x.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    position_vector = (x - neighbor_x).permute(0, 3, 1, 2).contiguous()  # B,3,N,k

    _, num_dims, _ = feature.size()

    feature = feature.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_feat = feature.view(batch_size * num_points, -1)[idx, :]
    neighbor_feat = neighbor_feat.view(batch_size, num_points, k, num_dims)
    neighbor_feat = neighbor_feat.permute(0, 3, 1, 2).contiguous()  # B,C,N,k

    return position_vector, neighbor_feat

class Point_Transformer(nn.Module):
    def __init__(self, input_features_dim):
        super(Point_Transformer, self).__init__()

        self.conv_theta1 = nn.Conv2d(3, input_features_dim, 1)
        self.conv_theta2 = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.bn_conv_theta = nn.BatchNorm2d(input_features_dim)

        self.conv_phi = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.conv_psi = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.conv_alpha = nn.Conv2d(input_features_dim, input_features_dim, 1)

        self.conv_gamma1 = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.conv_gamma2 = nn.Conv2d(input_features_dim, input_features_dim, 1)
        self.bn_conv_gamma = nn.BatchNorm2d(input_features_dim)

    def forward(self, xyz, features, k):

        position_vector, x_j = transformer_neighbors(xyz, features, k=k)

        delta = F.relu(self.bn_conv_theta(self.conv_theta2(self.conv_theta1(position_vector)))) # B,C,N,k
        # corrections for x_i
        x_i = torch.unsqueeze(features, dim=-1).repeat(1, 1, 1, k) # B,C,N,k

        linear_x_i = self.conv_phi(x_i) # B,C,N,k

        linear_x_j = self.conv_psi(x_j) # B,C,N,k

        relation_x = linear_x_i - linear_x_j + delta # B,C,N,k
        relation_x = F.relu(self.bn_conv_gamma(self.conv_gamma2(self.conv_gamma1(relation_x)))) # B,C,N,k

        weights = F.softmax(relation_x, dim=-1) # B,C,N,k
        features = self.conv_alpha(x_j) + delta # B,C,N,k

        f_out = weights * features # B,C,N,k
        f_out = torch.sum(f_out, dim=-1) # B,C,N

        return f_out


class ResPCPNet(nn.Module):
    def __init__(self, num_points=500, output_dim=3, use_point_stn=True, use_feat_stn=False, sym_op='max', get_pointfvals=False, point_tuple=1,num=0):
        super(ResPCPNet, self).__init__()
        self.num_points = num_points
        self.sym_op=sym_op
        self.feat1 = ResPointNetfeat(
            num_points=num_points,
            num_scales=1,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            get_pointfvals=get_pointfvals,
            point_tuple=point_tuple)
        self.feat2 = ResPointNetfeat(
            num_points=num_points,
            num_scales=1,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            get_pointfvals=get_pointfvals,
            point_tuple=point_tuple)
        self.b1 = BasicBlock(128, 1024, conv=True)
        self.b2 = BasicBlock(1024, 512)

        self.b3 = BasicBlock(512, 256)
        self.b4 = BasicBlock(256, output_dim)
        self.num=num
        self.attention = Point_Transformer(128)
    def forward(self, x,origin_noise):
        if self.num<1:
            x, trans,_ = self.feat1(x)
            x = self.b1(x)
            if self.sym_op == 'max':
                x,_ = torch.max(x,dim=-1)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
            x = self.b2(x)
            x = self.b3(x)
            x = self.b4(x)
        else:
            x, trans,_ = self.feat2(x)
            origin_noise = origin_noise.transpose(2, 1)
            origin_noise = torch.bmm(origin_noise, trans)
            origin_noise = origin_noise.transpose(2, 1)
            origin_noise = origin_noise.contiguous().view(origin_noise.size(0), 3, -1)
            x = self.attention(origin_noise, x, 4)
            x = self.b1(x)
            if self.sym_op == 'max':
                x,_ = torch.max(x,dim=-1)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
            x = self.b2(x)
            x = self.b3(x)
            x = self.b4(x)
        return x,trans

if __name__ == '__main__':


    model = pointfilternet().cuda()

    print(model)
