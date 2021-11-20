import torch
import torch.nn as nn
import torch.optim as optim


class InitConvBlock(nn.Module):
    def __init__(self, seq_type, drop_rate, ):
        super(InitConvBlock, self).__init__()
        if seq_type == 'dna':
            self.block = nn.Sequential(
                nn.Conv2d(1, 128, (4, 9), 1, (0, 4)),
            )
        elif seq_type == 'dns':
            self.block = nn.Sequential(
                nn.Conv2d(1, 128, (1, 9), 1, (0, 4)),
            )
        else:  # service for match
            self.block = nn.Sequential(
                nn.Conv2d(1, 128, (1, 9), 1, (0, 4)),
            )
    
    def forward(self, x):
        return self.block(x)


class OneDimConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pool_size, pool_type, drop_rate, ):
        super(OneDimConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channel, (1, 1), 1),
        )
        self.pool = nn.MaxPool2d(pool_size) if pool_type == 'max' else nn.AvgPool2d(pool_size)
  
    def forward(self, x):
        out = self.block(x)
        out = self.pool(out)
        return out


class BasicConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernal, drop_rate, ):
        super(BasicConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channel, kernal, 1, (0, 4)),
        )

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)
    

class DenseConvBlock(nn.Module):
    def __init__(self, num_layers, in_channel, out_channel, drop_rate, ):
        super(DenseConvBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(BasicConvBlock(in_channel*(i+1), out_channel, (1, 9), drop_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CommonConvBlock(nn.Module):
    def __init__(self, seq_type, drop_rate, ):
        super(CommonConvBlock, self).__init__()
        self.seq_type = seq_type
        self.block = nn.Sequential(
            InitConvBlock(seq_type, drop_rate),
            DenseConvBlock(3, 128, 128, drop_rate),
            OneDimConvBlock(128 + 128 * 3, 256, (1, 4), 'max', drop_rate),
            DenseConvBlock(3, 256, 256, drop_rate),
            OneDimConvBlock(256 + 256 * 3, 512, (1, 4), 'max', drop_rate),
        )
        self.out_size = 1000 // 4 // 4 * 512
    
    def forward(self, x):
        b, _, _, w = x.size()
        x = x.view(b, 1, 4, w) if self.seq_type == 'dna' else x.view(b, 1, 1, w)
        out = self.block(x)
        out = out.view(b, -1)
        return out


class MatchConvBlock(nn.Module):
    def __init__(self, drop_rate, ):
        super(MatchConvBlock, self).__init__()
        self.init = InitConvBlock(None, drop_rate)  # channel = 128
        self.conv1_1 = BasicConvBlock(128, 128, (1, 9), drop_rate)  # channel = 256
        self.conv1_2 = BasicConvBlock(256, 256, (1, 9), drop_rate)  # channel = 512
        self.conv1_3 = OneDimConvBlock(2 * 256, 128, (1, 4), 'avg', drop_rate)  # channel = 128
        self.conv2_1 = BasicConvBlock(128, 128, (1, 9), drop_rate)  # channel = 256
        self.conv2_2 = BasicConvBlock(256, 256, (1, 9), drop_rate)  # channel = 512
        self.conv2_3 = OneDimConvBlock(2 * 256, 128, (1, 4), 'avg', drop_rate)  # channel = 128
        self.out_size = 1000 // 4 // 4 * 128

    def forward(self, x):
        b, _, _, _ = x.size()
        init = self.init(x)
        out1_1 = self.conv1_1(init)  # output:256
        out1_2 = self.conv1_2(out1_1)  # output:512
        out1_3 = self.conv1_3(out1_2)  # output:128
        out2_1 = self.conv2_1(out1_3)  # output:256
        out2_2 = self.conv2_2(out2_1)  # output:512
        out2_3 = self.conv2_3(out2_2)  # output:128
        out = out2_3.view(b, -1)
        return out


class GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size, ):
        super(GRUBlock, self).__init__()
        self.out_size = 100 * 2 * 4 * hidden_size
        self.gru1 = nn.GRU(input_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.ln1 = nn.LayerNorm((100, 64), 2*hidden_size)
        self.gru2 = nn.GRU(2*hidden_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.ln2 = nn.LayerNorm((100, 64), 2*hidden_size)
        self.gru3 = nn.GRU(4*hidden_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.ln3 = nn.LayerNorm((100, 64), 2*hidden_size)
        self.gru4 = nn.GRU(6*hidden_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.ln4 = nn.LayerNorm((100, 64), 2*hidden_size)
        
    def forward(self, x):
        # user warning: need to compact rnn module weights
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        self.gru3.flatten_parameters()
        self.gru4.flatten_parameters()
        # main block
        b, _, _ = x.size()
        out1, _ = self.gru1(x)
        out1 = self.ln1(out1)
        out2, _ = self.gru2(out1)
        out2 = self.ln2(out2)
        in3 = torch.cat([out1, out2], 2)
        out3, _ = self.gru3(in3)
        out3 = self.ln3(out3)
        in4 = torch.cat([in3, out3], 2)
        out4, _ = self.gru4(in4)
        out4 = self.ln4(out4)
        out = torch.cat([in4, out4], 2)
        out = out.contiguous().view(b, -1)
        return out  # b, seq_len, 8*hidden_size(256)


class MyModule(nn.Module):
    def __init__(self, drop_rate, ):
        super(MyModule, self).__init__()
        self.dna_module = CommonConvBlock(seq_type='dna', drop_rate=drop_rate)
        self.dna_len = self.dna_module.out_size
        self.dns_module = CommonConvBlock(seq_type='dns', drop_rate=drop_rate)
        self.dns_len = self.dns_module.out_size
        self.combined_len = self.dna_len + self.dns_len
        # convolution -> conv ffn -> match result
        self.conv = MatchConvBlock(drop_rate)
        self.conv_len = self.conv.out_size
        # gru -> gru ffn -> classification result
        self.gru = GRUBlock(1, 32)
        self.gru_len = self.gru.out_size
        # ffn
        self.ffn = nn.Sequential(  # feed the output of cnn to gru
            nn.Dropout(drop_rate),
            nn.Linear(int(self.combined_len), 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(2000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
        self.conv2out = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(int(self.conv_len)+int(self.gru_len), 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(2000, 1000),
            nn.Sigmoid(),
        )
        self.ffn2gru = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(1000, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU()
        )
        self.gru2out = nn.Sequential(  # use the output of lstm to classify
            nn.Dropout(drop_rate),
            nn.Linear(int(self.gru_len)+int(self.conv_len), 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(1000, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(200, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(50, 7),
            nn.Sigmoid()
        )


    def forward(self, dna, dns):
        flat_dna = self.dna_module(dna)
        flat_dns = self.dns_module(dns)
        combined = torch.cat([flat_dna, flat_dns], 1)
        out_ffn = self.ffn(combined)  # dim: b, 500
        # prepare for conv
        in_conv = out_ffn.unsqueeze(2)  # dim: b, 500, 1  (in_c)
        in_conv = in_conv.transpose(1, 2)  # dim: b, 1, 500
        in_conv = in_conv.unsqueeze(1)  # dim: b, 1, 1, 500  (channel=1)
        out_conv = self.conv(in_conv)  # dim: b, 500 // 4 // 4 * 128  (cat_m to concatenate)
        # prepare for rnn
        in_gru = self.ffn2gru(out_ffn)
        in_gru = in_gru.unsqueeze(2)
        out_gru = self.gru(in_gru)
        b, _ = out_gru.size()  # batch_size, seq_len(500), hidden_status(4*32)
        out_gru = out_gru.contiguous().view(b, -1)  # dim: 100 * 8 * 32 // 4 // 4 * 128  (cat_c to concatenate)
        # prepare for match
        out_gru_truncate = out_gru.clone().detach()
        in_match = torch.cat([out_conv, out_gru_truncate], 1)
        out_match = self.conv2out(in_match)
        out_match = out_match.unsqueeze(1)  # dim: b, 1, 1000
        # prepare for classification
        in_clf = torch.cat([out_conv, out_gru], 1)
        out_clf = self.gru2out(in_clf)
        return out_clf, out_match


class MyHistone():
    def __init__(self, drop_rate, learning_rate, device):
        self.module = torch.nn.DataParallel(MyModule(drop_rate).cuda(device[0]), device_ids=device)
        self.criterion_c = nn.BCELoss().cuda(device[0])
        self.criterion_m = nn.BCELoss().cuda(device[0])
        self.optimizer = optim.Adam(self.module.parameters(), lr=learning_rate)
        self.device = device

    def train_module(self, dna_batch, dns_batch, lab_batch, masked_batch,):
        self.module.train()
        dna_batch, dns_batch, lab_batch, masked_batch = map(lambda x: x.float(), (dna_batch, dns_batch, lab_batch, masked_batch))
        lab_batch = lab_batch.squeeze(1)
        if self.device:
            dna_batch, dns_batch, lab_batch, masked_batch = map(lambda x: x.cuda(self.device[0]), (dna_batch, dns_batch, lab_batch, masked_batch))
        ret_c, ret_m = self.module(dna_batch, dns_batch)
        if torch.any(torch.isnan(ret_c)):
            print('ret_c error')
            print(ret_c)
            return None
        if torch.any(torch.isnan(ret_m)):
            print('ret_m error')
            print(ret_m)
            return None
        try:
            loss_c = self.criterion_c(ret_c, lab_batch)
            loss_m = self.criterion_m(ret_m, masked_batch)
        except:
            print('loss fault.')
            return None
        else:
            loss = loss_c + loss_m
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(self.module.parameters(), 2)
            self.optimizer.step()
            return map(lambda x: x.cpu().data, (loss_c, loss_m))

    def valid_module(self, dna_batch, dns_batch, lab_batch, masked_batch, ):
        self.module.eval()
        dna_batch, dns_batch, lab_batch, masked_batch = map(lambda x: x.float(), (dna_batch, dns_batch, lab_batch, masked_batch))
        lab_batch = lab_batch.squeeze(1)
        if self.device:
            dna_batch, dns_batch, lab_batch, masked_batch = map(lambda x: x.cuda(self.device[0]), (dna_batch, dns_batch, lab_batch, masked_batch))
        ret_c, ret_m = self.module(dna_batch, dns_batch)
        if torch.any(torch.isnan(ret_c)):
            print('ret_c error')
            print(ret_c)
            return None
        if torch.any(torch.isnan(ret_m)):
            print('ret_m error')
            print(ret_m)
            return None
        try:
            loss_c = self.criterion_c(ret_c, lab_batch)
            loss_m = self.criterion_m(ret_m, masked_batch)
        except:
            print('-----> ', str(loss_c), '-----> ', str(loss_m))
            return None
        else:
            return map(lambda x: x.cpu().data, (loss_c, loss_m, ret_c, ret_m))

    def test_module(self, dna_batch, dns_batch, ):
        self.module.eval()
        dna_batch, dns_batch = map(lambda x: x.float(), (dna_batch, dns_batch))
        if self.device:
            dna_batch, dns_batch = map(lambda x: x.cuda(self.device[0]), (dna_batch, dns_batch))
        ret_c, ret_m = self.module(dna_batch, dns_batch)
        return map(lambda x: x.cpu().data, (ret_c, ret_m))

    def update_lr(self, fold):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= fold
        
    def save_module(self, path):
        torch.save({
            'module_state_dict': self.module.state_dict(),
        }, path)
    
    def load_module(self, path):
        state_dict = torch.load(path)['module_state_dict']
        self.module.load_state_dict(state_dict, strict=False)
