import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WeightPrediction(nn.Module):
    """
    自适应权重预测网络
    """
    def __init__(self, input_channels, num_layers):
        super(WeightPrediction, self).__init__()
        self.num_layers = num_layers
        self.fc_layers = nn.Sequential(
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Linear(64, num_layers),
            nn.Softmax(dim=1)  # 使用Softmax将输出转换为权重向量，使得权重之和为1
        )

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = x.mean(dim=1)  # 对时间步求均值，得到 [Batch, Channel] 的张量
        weights = self.fc_layers(x)
        return weights

class Bottleneck(nn.Module):
   

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

    

    def forward(self, x):
        

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out

class Bottleneck_2(nn.Module):
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_2, self).__init__()
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                                padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes , kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # 1x1 convolution layer to adjust dimensions
        self.adjust_channels = nn.Conv1d(inplanes, planes , kernel_size=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        elif x.shape[1] != out.shape[1]:
             residual = self.adjust_channels(x)

        out += residual
        out = self.relu(out)

        return out

class ConvolutionalModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvolutionalModule, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.ConvLayer1 = ConvolutionalModule(configs.enc_in, configs.enc_in)
        self.ConvLayer2 = ConvolutionalModule(configs.enc_in, configs.enc_in)
        self.ConvLayer3 = ConvolutionalModule(configs.enc_in, configs.enc_in)
        self.interactionmodule = Bottleneck(32, 32)
        self.interactionmodule_2 = Bottleneck_2(32, 32)
        self.weight_prediction =WeightPrediction(configs.enc_in,3)
        self.lstm1 = nn.LSTM(configs.enc_in, 32, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(configs.enc_in, 32, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(configs.enc_in, 32, num_layers=1, batch_first=True)
        self.interactionmodule_3 = Bottleneck(32, configs.enc_in)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = x.permute(0, 2, 1)
        conv1 = self.ConvLayer1(x)
        conv2 = self.ConvLayer2(conv1)
        conv3 = self.ConvLayer3(conv2)
        lstm_out1, _ = self.lstm1(conv1.permute(0, 2, 1))
        lstm_out2, _ = self.lstm2(conv2.permute(0, 2, 1))
        lstm_out3, _ = self.lstm3(conv3.permute(0, 2, 1))
        lstm_out1 = lstm_out1.permute(0, 2, 1)
        lstm_out2 = lstm_out2.permute(0, 2, 1)
        lstm_out3 = lstm_out3.permute(0, 2, 1)
        Interaction1_2 =  torch.cat([lstm_out1, lstm_out2], dim=2)
        Interaction2_3 =  torch.cat([lstm_out2, lstm_out3], dim=2)
        Interaction1_2init = self.interactionmodule(Interaction1_2)
        Interaction1_21 = torch.cat([lstm_out1, Interaction1_2init], dim=2)
        Interaction1_2init2 = self.interactionmodule_2(Interaction1_21)
        Interaction2_3init = self.interactionmodule(Interaction2_3)
        Interaction2_31 = torch.cat([lstm_out2, Interaction2_3init], dim=2)
        Interaction2_3init2 = self.interactionmodule_2(Interaction2_31)
        Interaction_Linear1 = nn.Linear(Interaction1_2init2.size(2), self.pred_len)
        Interaction_Linear2 = nn.Linear(Interaction2_3init2.size(2), self.pred_len)
        Interactionoutput= Interaction_Linear1(Interaction1_2init2)+Interaction_Linear2(Interaction2_3init2)
        Interactionoutput=self.interactionmodule_3(Interactionoutput)
        x = seasonal_output  +trend_output +Interactionoutput
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel] 
        

        
