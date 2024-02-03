from __future__ import print_function
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class pointfilter_encoder(nn.Module):
    def __init__(self, input_dim=3, patch_nums=500, sym_op='max'):
        super(pointfilter_encoder, self).__init__()
        self.patch_nums = patch_nums
        self.sym_op = sym_op
        self.input_dim = input_dim

        self.conv2 = nn.Conv1d(self.input_dim, 64, kernel_size=1)
        self.conv3 = nn.Conv1d(64 , 128, kernel_size=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=1)




        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)



        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.activate(self.bn1(self.conv2(x)))

        x = self.activate(self.bn2(self.conv3(x)))

        x = self.activate(self.bn3(self.conv4(x)))


        #net3 = x  # 256


        '''
        if self.sym_op == 'sum':
            x = torch.sum(x, dim=-1)
        else:
            x, index = torch.max(x, dim=-1)
        '''

        return x#, index


class pointfilter_decoder(nn.Module):
    def __init__(self):
        super(pointfilter_decoder, self).__init__()

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)
        self.conv5 = nn.Conv1d(256, 512, kernel_size=1)
        self.conv6 = nn.Conv1d(512, 1024, kernel_size=1)


        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)


        self.dropout_1 = nn.Dropout(0.3)
        self.dropout_2 = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn4(self.conv5(x)))
        # net4 = x  # 512

        x = F.relu(self.bn5(self.conv6(x)))
        # net5 = x  # 1024
        x, index = torch.max(x, dim=-1)
        x = F.relu(self.bn1(self.fc1(x)))
        # x = self.dropout_1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        # x = self.dropout_2(x)
        #x = F.relu(self.bn3(self.fc3(x)))
        x= torch.tanh(self.fc3(x))

        return x

class PointAttentionNetwork(nn.Module):
    def __init__(self,C, ratio = 4):
        super(PointAttentionNetwork, self).__init__()
        self.bn1 = nn.BatchNorm1d(C//ratio)
        self.bn2 = nn.BatchNorm1d(C//ratio)
        self.bn3 = nn.BatchNorm1d(C//2)
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C//ratio, kernel_size=1, bias=False),
                                self.bn1,
                                nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C//ratio, kernel_size=1, bias=False),
                                self.bn2,
                                nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=C, out_channels=C//2, kernel_size=1, bias=False),
                                self.bn3,
                                nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b,c,n = x.shape

        a = self.conv1(x).permute(0,2,1) # b, n, c/ratio

        b = self.conv2(x) # b, c/ratio, n

        s = self.softmax(torch.bmm(a, b)) # b,n,n

        x = self.conv3(x) # b,c,n
        out = x + torch.bmm(x, s.permute(0, 2, 1))
        return out

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


class pointfilternet(nn.Module):
    def __init__(self, input_dim=3,num=0, patch_nums=500, sym_op='max'):
        super(pointfilternet, self).__init__()

        self.patch_nums = patch_nums
        self.sym_op = sym_op
        self.input_dim = input_dim
        self.encoder1 = pointfilter_encoder()
        self.decoder1 = pointfilter_decoder()
        self.encoder2 = pointfilter_encoder()
        self.decoder2 = pointfilter_decoder()
        #self.attention= PointAttentionNetwork(512)
        self.attention = Point_Transformer(256)
        self.num=num
    def forward(self, x,origin_noise):
        if self.num<1:
            out = self.encoder1(x)
            #print(out.shape)
            pred_pts=self.decoder1(out)
        else:
            out1 = self.encoder2(x)
            #out2 = self.encoder2(origin_noise)
            #print(origin_noise.shape)
            #out3 = torch.cat([out1,out2],dim=1)
            #print(out3.shape)
            out4 = self.attention(origin_noise,out1,4)
            #print(out4.shape)
            pred_pts=self.decoder2(out4)

        return pred_pts

if __name__ == '__main__':


    model = pointfilternet().cuda()

    print(model)
