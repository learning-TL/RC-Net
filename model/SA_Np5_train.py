import torch.nn as nn
from scipy.io import savemat
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
import os
import torch
import numpy as np

parser = ArgumentParser(description='ISTA-Net')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=100, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=5, help='phase number of ISTA-Net')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
gpu_list = args.gpu_list

N = 6859
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Phi_data_Name = 'data/shape_A_3D_19_1528.mat'
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['A']
Phi_input = np.transpose(Phi_input)
# 将复数数组转换为复数张量
real_tensor = torch.tensor(Phi_input.real)
imag_tensor = torch.tensor(Phi_input.imag)
Phi = torch.complex(real_tensor, imag_tensor)
Phi = Phi.to(torch.complex128).to(device)
energy = (1 / (torch.sqrt(torch.sum(abs(torch.mul(Phi, Phi)), dim=1)))).unsqueeze(1)  # torch.Size([1528, 1])
Phi = Phi * energy

Training_data_Name = 'data/19_3D_2000.mat'
Training_data = sio.loadmat(Training_data_Name)
Training_labels = Training_data['x']

n_output = 19
nrtrain = Training_labels.shape[0]
batch_size = 64

L_data_Name = 'data/L_gyh.mat'
Training_data = sio.loadmat(L_data_Name)
L = Training_data['L']
L = torch.tensor(L).to(torch.complex128).to(device)


def normalize_data(s):
    data_max = torch.max(s) + 1e-8
    normalized_data = s / data_max
    return normalized_data


def abs_complex(s):
    s_abs = torch.abs(s)
    abs_data = s / s_abs
    return abs_data


def dice_coefficient(prediction, target, epsilon=1e-8):
    dice = 0
    for i in range(batch_size):
        intersection = torch.sum(prediction[i, :] * target[i, :])  # ([64, 784])
        union = torch.sum(prediction[i, :]) + torch.sum(target[i, :])  # (542.0961)一个数

        dice1 = 2.0 * (intersection + epsilon) / (union + epsilon)
        dice = dice + dice1

    loss = 1 - (dice / batch_size)
    # loss = torch.abs(loss)
    return loss


# SSIM (结构相似性指数)求每一行的ssim,然后求均值
def ssim_loss(pred, target, C1=0.001 ** 2, C2=0.001 ** 2):
    ssim_value = 0
    for i in range(batch_size):
        mu_x = torch.mean(pred[i, :])
        mu_y = torch.mean(target[i, :])

        sigma_x = torch.var(pred[i, :])
        sigma_y = torch.var(target[i, :])

        sigma_xy = torch.mean((pred[i, :] - mu_x) * (target[i, :] - mu_y))

        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)

        denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        ssim_value1 = numerator / denominator
        ssim_value = ssim_value + ssim_value1

    ssim_loss = 1 - (ssim_value / batch_size)
    return ssim_loss


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=4, spatial_kernel=3):
        super(CBAMLayer, self).__init__()

        # spatial attention
        self.conv = nn.Conv3d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


# Define ISTA-Net Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([1e-5]))
        self.thf = nn.Parameter(torch.Tensor([1e-4]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.5]))

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(4, 2, 3, 3)))
        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(4, 4, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(8, 4, 3, 3)))
        self.conv3_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(8, 8, 3, 3)))
        self.conv4_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 8, 3, 3)))
        self.conv5_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 16, 3, 3)))

        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 16, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(8, 16, 3, 3)))
        self.conv3_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(8, 8, 3, 3)))
        self.conv4_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(4, 8, 3, 3)))
        self.conv5_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(4, 4, 3, 3)))
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 4, 3, 3)))
        self.bn = nn.BatchNorm3d(2)
        self.bn1 = nn.BatchNorm3d(1)
        self.SA = CBAMLayer(16)

    def forward(self, x0, PhiTPhi, PhiTb):
        x0 = x0.to(device)
        LTL = torch.mm(torch.transpose(L, 0, 1), L)
        lapl = torch.mm(LTL, torch.transpose(x0, 0, 1))
        x = x0 - self.lambda_step * (torch.transpose(torch.mm(PhiTPhi, torch.transpose(x0, 0, 1)), 0,
                                                     1) - PhiTb) - 2 * self.thf * torch.transpose(lapl, 0,
                                                                                                  1)

        x_input = x.view(-1, 1, n_output, n_output)
        real_part = torch.real(x_input)
        imaginary_part = torch.imag(x_input)

        x_input_2 = torch.cat([real_part, imaginary_part], dim=1)
        x_input_2 = x_input_2.type(self.conv_D.dtype)
        x_input_2 = self.bn(x_input_2)

        # 开始卷积
        x_D = F.relu(F.conv3d(x_input_2, self.conv_D, padding=1))
        x = F.relu(F.conv3d(x_D, self.conv1_forward, padding=1))
        x = F.relu(F.conv3d(x, self.conv2_forward, padding=1))
        x = F.relu(F.conv3d(x, self.conv3_forward, padding=1))
        x = F.relu(F.conv3d(x, self.conv4_forward, padding=1))
        x = F.relu(F.conv3d(x, self.conv5_forward, padding=1))

        x_forward = self.SA(x)

        x = F.relu(F.conv3d(x_forward, self.conv1_backward, padding=1))
        x = F.relu(F.conv3d(x, self.conv2_backward, padding=1))
        x = F.relu(F.conv3d(x, self.conv3_backward, padding=1))
        x = F.relu(F.conv3d(x, self.conv4_backward, padding=1))
        x = F.relu(F.conv3d(x, self.conv5_backward, padding=1))
        x_G = F.conv3d(x, self.conv_G, padding=1)
        x_temp = x_G
        x_pred = x_temp[:, 0, :, :, :] + 1j * x_temp[:, 1, :, :, :]
        x_pred = x_pred.view(-1, N)
        x_pred = x_pred + self.soft_thr * (x_pred - x0)
        return x_pred


class post_ca(nn.Module):
    def __init__(self):
        super(post_ca, self).__init__()
        self.conv_1 = nn.Parameter(init.xavier_normal_(torch.Tensor(4, 2, 3, 3)))
        self.conv_2 = nn.Parameter(init.xavier_normal_(torch.Tensor(8, 4, 3, 3)))
        self.conv_3 = nn.Parameter(init.xavier_normal_(torch.Tensor(8, 8, 3, 3)))
        self.conv_4 = nn.Parameter(init.xavier_normal_(torch.Tensor(4, 8, 3, 3)))
        self.conv_5 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 4, 3, 3)))
        self.bn = nn.BatchNorm3d(2)

    def forward(self, x0):
        x0 = x0.view(-1, 1, n_output, n_output)

        real_part = torch.real(x0)
        imaginary_part = torch.imag(x0)
        x = torch.cat([real_part, imaginary_part], dim=1)
        x = x.type(self.conv_1.dtype)
        x = self.bn(x)
        x = F.relu(F.conv3d(x, self.conv_1, padding=1))
        x = F.relu(F.conv3d(x, self.conv_2, padding=1))
        x = F.relu(F.conv3d(x, self.conv_3, padding=1))
        x = F.relu(F.conv3d(x, self.conv_4, padding=1))
        x = torch.relu(F.conv3d(x, self.conv_5, padding=1))
        x = x.view(-1, N)
        return x


class RNet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(RNet, self).__init__()
        onelayer = []

        self.LayerNo = LayerNo
        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)
        self.fcs1 = post_ca()

    def forward(self, Phix, Phi):
        Phi = torch.complex(Phi[:, :, :, 0], Phi[:, :, :, 1])
        Phix = torch.complex(Phix[:, :, :, 0], Phix[:, :, :, 1])

        PhiTPhi = torch.matmul(torch.transpose(Phi, 0, 1), Phi)

        PhiTb = torch.matmul(torch.transpose(Phi, 0, 1), torch.transpose(Phix, 0, 1))
        PhiTb = torch.transpose(PhiTb, 0, 1)

        x = (torch.zeros(batch_size, N)).to(torch.complex128).to(device)  # 初始化
        for i in range(self.LayerNo):
            x = self.fcs[i](x, PhiTPhi, PhiTb)
        x_final = x
        x_single = self.fcs1(x_final)

        return x_single


model = RNet(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

print_flag = 1  # print parameter number

if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        # print('Layer %d' % num_count)
        # print(para.size())


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :])  # 根据索引 `index` 返回对应的数据样本

    def __len__(self):
        return self.len


if (platform.system() == "Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True, drop_last=True)

else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=4,
                             shuffle=True, drop_last=True)


# Training loop
def train():
    losses = []
    dice_losses = []
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=learning_rate, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=100,
                                                              verbose=True, threshold=1e-5, threshold_mode='rel',
                                                              cooldown=0, min_lr=0, eps=1e-8)

    for epoch_i in range(start_epoch + 1, end_epoch + 1):
        index = 0
        for data in rand_loader:
            index += 1
            data = data.to(device)
            batch_x = data.to(torch.complex128)
            Phix = torch.mm(Phi.conj(), torch.transpose(batch_x, 0, 1)).to(device)
            Phix = torch.transpose(Phix, 0, 1)
            Phix = Phix * torch.transpose(energy, 0, 1)

            x_single = model(Phix, Phi)
            loss = F.mse_loss(x_single, data)

            losses.append(loss.item())

            dice_losses.append(loss2.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.zero_grad()
            lr_scheduler.step(loss)

            if epoch_i % 1 == 0:
                output_data = "[%02d/%02d]  total_loss: %.4f\n" % (
                    epoch_i, end_epoch, loss.item())
                print(output_data)
            if epoch_i == end_epoch:
                savemat(f'data/0dB/test_resul_{index}dB.mat', {"data": x_single.cpu().detach().numpy()})
                savemat(f'data/0dB/test_resul_{index}dB_label.mat', {"data": data.cpu().detach().numpy()})
    torch.save(model, "save_model/OurNetmodel_3D_fangfanduibi.pt")
if __name__ == '__main__':
    train()
