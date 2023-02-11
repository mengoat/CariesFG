import math
import torch
import torch.nn as nn
import copy

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter

class DPS(nn.Module):
    def __init__(self,in_channel: int,num_classes: int,num_selects: int,global_feature_dim=1536):
        super(DPS, self).__init__()
        self.proj = nn.Conv2d(in_channel,global_feature_dim,kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Conv2d(global_feature_dim, global_feature_dim, 1),
            nn.BatchNorm2d(global_feature_dim),
            nn.ReLU(),
            nn.Conv2d(global_feature_dim, num_classes, 1)
        )
        self.fc = nn.Linear(in_channel,num_classes)
        self.num_selects = num_selects

    def forward(self,features):
        proj_features = self.proj(features)
        logits = self.classifier(proj_features)

        B, C, H, W = logits.size()
        logits = logits.view(B, C, -1).transpose(2, 1).contiguous()

        B, C, H, W = proj_features.size()
        proj_features = proj_features.view(B, C, -1).transpose(2, 1).contiguous()

        logits = torch.softmax(logits, dim=-1)
        logits_max, _ = torch.max(logits, dim=-1)
        logits_max, ranks = torch.sort(logits_max, descending=True)
        selection = ranks[:, :self.num_selects]

        selected_features = []
        for batch in range(B):
            selected_features.append(proj_features[batch][selection[batch]])
        selected_features = torch.stack(selected_features)

        return selected_features

class GCA(nn.Module):

    def __init__(self,
                 num_joints: int,
                 in_features: int,
                 num_classes: int,
                 use_global_token: bool = False):
        super(GCN, self).__init__()

        joints = [num_joints // 8, num_joints // 16, num_joints // 32]

        # 1
        self.pool1 = nn.Linear(num_joints, joints[0])

        A = torch.eye(joints[0]) / 100 + 1 / 100
        self.adj1 = nn.Parameter(copy.deepcopy(A))
        self.conv1 = nn.Conv1d(in_features, in_features, 1)
        self.batch_norm1 = nn.BatchNorm1d(in_features)

        self.conv_q1 = nn.Conv1d(in_features, in_features // 4, 1)
        self.conv_k1 = nn.Conv1d(in_features, in_features // 4, 1)
        self.alpha1 = nn.Parameter(torch.zeros(1))

        # 2
        self.pool2 = nn.Linear(joints[0], joints[1])

        A = torch.eye(joints[1]) / 32 + 1 / 32
        self.adj2 = nn.Parameter(copy.deepcopy(A))
        self.conv2 = nn.Conv1d(in_features, in_features, 1)
        self.batch_norm2 = nn.BatchNorm1d(in_features)

        self.conv_q2 = nn.Conv1d(in_features, in_features // 4, 1)
        self.conv_k2 = nn.Conv1d(in_features, in_features // 4, 1)
        self.alpha2 = nn.Parameter(torch.zeros(1))

        # 3
        self.pool3 = nn.Linear(joints[1], joints[2])

        A = torch.eye(joints[2]) / 32 + 1 / 32
        self.adj3 = nn.Parameter(copy.deepcopy(A))
        self.conv3 = nn.Conv1d(in_features, in_features, 1)
        self.batch_norm3 = nn.BatchNorm1d(in_features)

        self.conv_q3 = nn.Conv1d(in_features, in_features // 4, 1)
        self.conv_k3 = nn.Conv1d(in_features, in_features // 4, 1)
        self.alpha3 = nn.Parameter(torch.zeros(1))

        self.pool4 = nn.Linear(joints[2], 1)

        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(in_features, num_classes)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.pool1(x)
        q1 = self.conv_q1(x).mean(1)
        k1 = self.conv_k1(x).mean(1)
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))
        A1 = self.adj1 + A1 * self.alpha1
        x = self.conv1(x)
        x = torch.matmul(x, A1)
        x = self.batch_norm1(x)

        x = self.pool2(x)
        q2 = self.conv_q2(x).mean(1)
        k2 = self.conv_k2(x).mean(1)
        A2 = self.tanh(q2.unsqueeze(-1) - k2.unsqueeze(1))
        A2 = self.adj2 + A2 * self.alpha2
        x = self.conv2(x)
        x = torch.matmul(x, A2)
        x = self.batch_norm2(x)

        x = self.pool3(x)
        q3 = self.conv_q3(x).mean(1)
        k3 = self.conv_k3(x).mean(1)
        A3 = self.tanh(q3.unsqueeze(-1) - k3.unsqueeze(1))
        A3 = self.adj3 + A3 * self.alpha3
        x = self.conv2(x)
        x = torch.matmul(x, A3)
        x = self.batch_norm2(x)

        x = self.pool4(x)
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.classifier(x)

        return x
