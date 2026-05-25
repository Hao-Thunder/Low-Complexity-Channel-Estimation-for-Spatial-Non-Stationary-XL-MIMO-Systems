import numpy as np
from scipy.special import gamma, psi
from scipy.stats import norm
from function import complex_to_real_4x4, complex_to_real_stack, real_to_complex_np, real_to_complex_stack, \
                    real_to_complex_4x4, genarate_sub_channel, genarate_sub_channel11, \
                    array_response_g12_torch, array_response_g12_torch11, function_grad_preparation, \
                    MP_in_2D_Markov_torch, MP_in_2D_Markov_torch11
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from torch.distributions import Normal
from modelss.convnexts.convnext import ConvNeXt, ConvNeXt1
from modelss.mrdn.DenoisingModels import MRDN
from torch.utils.data import Dataset
from thop import profile
import timm


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, X_data, Y_data):
        """
        :param X_data: 输入数据，形状为 (256, 10000)，每列一个样本
        :param Y_data: 目标数据，形状为 (16, 2, 4, 10000)，每列一个目标
        """
        # 将输入数据转置为 (10000, 256) 以便每行是一个样本
        self.X_data = X_data.T  # 转置，使得每行是一个样本
        # self.Y_data = Y_data.T  # 转置，使得每行是一个样本
        self.Y_data = np.transpose(Y_data, (3, 0, 1, 2))

    def __len__(self):
        # 返回数据集的大小，这里是样本的数量，即 X_data 或 Y_data 的行数
        return len(self.X_data)

    def __getitem__(self, idx):
        """
        :param idx: 索引
        :return: 一个包含输入和目标数据的元组
        """
        # 返回当前索引对应的输入和目标数据
        return self.X_data[idx], self.Y_data[idx]


class VBInet(nn.Module):
    def __init__(self, Iter_VBI, in_chans_VBI, depths, dims, num_classes_L, num_classes_rou):
        super(VBInet, self).__init__()
        self.L = Iter_VBI
        self.in_chans = in_chans_VBI
        self.depths = depths
        self.dims = dims
        self.num_classes_L = num_classes_L
        self.num_classes_rou = num_classes_rou

        self.layers_L = ConvNeXt(in_chans=64, depths=self.depths, dims=self.dims, num_classes=self.num_classes_L)
        self.layers_conv_layer = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=8, padding=1)

        self.layers_rou = nn.ModuleList()
        for i in range(self.L):
            # 每个 ConvNeXt 层使用相同的参数
            convnext_layer1 = ConvNeXt(in_chans=1, depths=self.depths, dims=self.dims, num_classes=self.num_classes_rou)
            self.layers_rou.append(convnext_layer1)

    def forward(self, y_torch, sigma2_torch, PI, F_torch, z_torch, h_torch_real):
        BATCH_SIZE = y_torch.shape[0]
        myloss = torch.tensor(0.0, dtype=torch.float32)

        with torch.no_grad():
            F_conj_T = F_torch.conj().transpose(1, 2)  # F_torch[batch, :, :].conj().T
            noise_precision = (1 / sigma2_torch)
            FTY = torch.bmm(F_conj_T, y_torch)  # 批量计算 F_conj_T @ y_torch
            Fz = torch.bmm(F_torch, z_torch)  # 批量计算 F_torch @ z_torch
            FHF = torch.bmm(F_conj_T, F_torch)

        input1 = self.layers_conv_layer(FHF.unsqueeze(1))
        output_L = self.layers_L(input1)
        output_rou = PI

        # 向量化循环操作，直接对整个batch进行计算
        for i in range(self.L):
            SigmaX = 1 / ((noise_precision * output_L).unsqueeze(1) + output_rou)  # 向量化SigmaX计算
            MuX = SigmaX * (z_torch * output_L.unsqueeze(1) + (FTY - torch.bmm(F_conj_T, Fz))) * noise_precision.unsqueeze(1)

            # aaaa = torch.cat((MuX, SigmaX), dim=2).unsqueeze(1)

            input_rou = torch.cat((MuX, SigmaX), dim=2).unsqueeze(1).view(BATCH_SIZE, 1, 64, 32)
            a = self.layers_rou[i](input_rou).unsqueeze(2)
            output_rou = a  # 转置以适应批次更新

            z_torch = MuX
            Fz = torch.bmm(F_torch, z_torch)

        SigmaX = 1 / ((noise_precision * output_L).unsqueeze(1) + output_rou)  # 向量化SigmaX计算
        MuX = SigmaX * (z_torch * output_L.unsqueeze(1) + (FTY - torch.bmm(F_conj_T, Fz))) * noise_precision.unsqueeze(
            1)
        z_torch = MuX
        # h_est = Fz
        # loss = torch.mean((h_torch_real - h_est) ** 2, dim=1)
        # myloss1 = loss / sigma2_torch
        # final_loss = torch.sum(myloss1)
        # myloss = myloss + final_loss

        return z_torch, F_torch, myloss


class Gradnet31(nn.Module):
    def __init__(self, Iter_grad, depths, dims):
        super(Gradnet31, self).__init__()
        self.L = Iter_grad
        self.depths = depths
        self.dims = dims

        self.layers_conv_layer0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=8, padding=1)
        self.upconv = nn.ConvTranspose2d(2, 4, kernel_size=4, stride=2, padding=1)
        self.layer1 = ConvNeXt(in_chans=65, depths=self.depths, dims=self.dims, num_classes=8)
        self.layer2 = ConvNeXt(in_chans=65, depths=self.depths, dims=self.dims, num_classes=8)
        # self.relu = nn.ReLU()
        # self.tanh = torch.tanh

    def forward(self, y_torch, F_torch1, z_torch1, polar_grid_torch, Mx, Mz, A_torch, lambda_, d, M, nGrid, sigma2,
                nPath, grad_Q_to_g11, grad_Q_to_g22, index_amp):
        # global sqrt_Mx
        BATCH_SIZE = y_torch.shape[0]

        A_complex = real_to_complex_4x4(A_torch.cuda())

        F_downsample = self.layers_conv_layer0(F_torch1.unsqueeze(1))

        # aaa = torch.cat((y_torch, z_torch1), dim=2)
        x_y_cat = torch.cat((y_torch, z_torch1), dim=2).view(-1, 2, 32, 32)
        x_y_cat2 = self.upconv(x_y_cat).view(-1, 1, 128, 128)

        input00 = torch.cat((F_downsample, x_y_cat2), dim=1)

        out1 = (self.layer1(input00))
        out2 = (self.layer2(input00))

        grid_update = polar_grid_torch.clone()
        grid_update[torch.arange(BATCH_SIZE).unsqueeze(1), index_amp, 0] = (
                grid_update[torch.arange(BATCH_SIZE).unsqueeze(1), index_amp, 0]
                + (out1 * 2 / Mx / 40 * torch.sign(grad_Q_to_g11)))
        grid_update[torch.arange(BATCH_SIZE).unsqueeze(1), index_amp, 1] = (
                grid_update[torch.arange(BATCH_SIZE).unsqueeze(1), index_amp, 1]
                + (out2 * 2 / Mz / 40 * torch.sign(grad_Q_to_g22)))

        A_complex_update1 = A_complex.clone()

        delta_x = (
            torch.arange(-(Mx - 1) / 2, (Mx - 1) / 2 + 1, dtype=torch.float64, device=A_complex.device)).unsqueeze(
            1).expand(BATCH_SIZE, -1, -1)  # X轴位置
        delta_z = (
            torch.arange(-(Mz - 1) / 2, (Mz - 1) / 2 + 1, dtype=torch.float64, device=A_complex.device)).unsqueeze(
            1).expand(BATCH_SIZE, -1, -1)  # Z轴位置
        k_lambda = -1j * 2 * torch.pi / lambda_
        sqrt_Mx = torch.sqrt(torch.tensor(1 / Mx, dtype=torch.float64, device=A_complex.device))
        sqrt_Mz = torch.sqrt(torch.tensor(1 / Mz, dtype=torch.float64, device=A_complex.device))

        xx = torch.zeros((BATCH_SIZE, M, 8), dtype=torch.complex64, device=A_complex.device)
        for q in range(index_amp.shape[1]):
            # g1_batch = torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float32, device=grid_update.device)
            # g2_batch = torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float32, device=grid_update.device)
            # r_batch = torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float32, device=grid_update.device)
            #
            # for batch_idx in range(BATCH_SIZE):
            #     grid_index = index_amp[batch_idx, q]
            #     g1_batch[batch_idx, 0, 0] = grid_update[batch_idx, grid_index, 0]
            #     g2_batch[batch_idx, 0, 0] = grid_update[batch_idx, grid_index, 1]
            #     r_batch[batch_idx, 0, 0] = grid_update[batch_idx, grid_index, 2]

            indices = index_amp[:, q]  # 这是一个形状为 (BATCH_SIZE,) 的张量，表示每个批次对应的索引

            # 使用批量索引提取相应的 g1, g2 和 r 值
            g1_batch = grid_update[torch.arange(BATCH_SIZE), indices, 0].unsqueeze(1).unsqueeze(2)
            g2_batch = grid_update[torch.arange(BATCH_SIZE), indices, 1].unsqueeze(1).unsqueeze(2)
            r_batch = grid_update[torch.arange(BATCH_SIZE), indices, 2].unsqueeze(1).unsqueeze(2)

            ax000 = (sqrt_Mx * torch.exp(
                    k_lambda * (-delta_x * d * g1_batch + delta_x ** 2 * d ** 2 * (1 - g1_batch ** 2) / (2 * r_batch)))).squeeze(2)
            az000 = (sqrt_Mz * torch.exp(
                k_lambda * (-delta_z * d * g2_batch + delta_z ** 2 * d ** 2 * (1 - g2_batch ** 2) / (2 * r_batch)))).squeeze(2)

            a_cross_term111 = torch.exp(
                (k_lambda * (-1 / r_batch)) *
                ((delta_x * d * g1_batch) @ ((delta_z * d * g2_batch).transpose(1, 2)))
            )
            a_cross_term000 = a_cross_term111.transpose(1, 2).reshape(BATCH_SIZE, -1)

            result = []
            for i in range(BATCH_SIZE):
                # 对每个 batch 的 az[i] 和 ax[i] 进行 Kronecker 积
                kronecker_product = torch.kron(az000[i].squeeze(), ax000[i].squeeze())  # (4, 1) 和 (64, 1) 的 Kronecker 积
                result.append(kronecker_product)
            # 将每个结果堆叠到一起，得到 (BATCH_SIZE, 256)
            result_tensor000 = torch.stack(result, dim=0)

            xx[:, :, q] = result_tensor000 * a_cross_term000
            for batch_idx in range(BATCH_SIZE):
                grid_index = index_amp[batch_idx, q]
                A_complex_update1[batch_idx, :, grid_index] = xx[batch_idx, :, q]

        # print(np.max((A_complex_update1 - A_complex_update).detach().cpu().numpy()))

        F_update_complex10 = A_complex_update1
        F_update_grad0000 = complex_to_real_4x4(F_update_complex10)

        return F_update_grad0000, grid_update


class Gradnet33(nn.Module):
    def __init__(self, Iter_grad, depths, dims):
        super(Gradnet33, self).__init__()
        self.L = Iter_grad
        self.depths = depths
        self.dims = dims

        self.layers_conv_layer0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=8, padding=1)
        self.upconv = nn.ConvTranspose2d(2, 4, kernel_size=4, stride=2, padding=1)
        self.layer1 = ConvNeXt(in_chans=65, depths=self.depths, dims=self.dims, num_classes=8)
        self.layer2 = ConvNeXt(in_chans=65, depths=self.depths, dims=self.dims, num_classes=8)
        # self.relu = nn.ReLU()
        # self.tanh = torch.tanh

        self.layer3 = nn.Sequential(
            nn.Linear(1026, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1024)
        )

    def forward(self, y_torch, F_torch1, z_torch1, polar_grid_torch, Mx, Mz, A_torch, lambda_, d, M, nGrid, sigma2,
                nPath, grad_Q_to_g11, grad_Q_to_g22, index_amp):
        # global sqrt_Mx
        BATCH_SIZE = y_torch.shape[0]

        input_for_grad_00 = torch.cat((y_torch, z_torch1), dim=-1).repeat(1, 1, 8)
        input_for_grad = torch.cat((input_for_grad_00, F_torch1), dim=-1).view(BATCH_SIZE, -1, 128, 128)

        out1 = (self.layer1(input_for_grad))
        out2 = (self.layer2(input_for_grad))

        grid_update = polar_grid_torch.clone()
        grid_update[torch.arange(BATCH_SIZE).unsqueeze(1), index_amp, 0] = grid_update[torch.arange(BATCH_SIZE).unsqueeze(1), index_amp, 0] + out1
        grid_update[torch.arange(BATCH_SIZE).unsqueeze(1), index_amp, 1] = grid_update[torch.arange(BATCH_SIZE).unsqueeze(1), index_amp, 1] + out2

        A_complex = real_to_complex_4x4(A_torch.cuda())
        A_complex_update1 = A_complex.clone()

        # index = index_amp.unsqueeze(1).expand(-1, M, -1)
        # A_selected = torch.gather(A_complex, dim=2, index=index)

        delta_x = (
            torch.arange(-(Mx - 1) / 2, (Mx - 1) / 2 + 1, dtype=torch.float64, device=A_complex.device)).unsqueeze(
            1).expand(BATCH_SIZE, -1, -1)  # X轴位置
        delta_z = (
            torch.arange(-(Mz - 1) / 2, (Mz - 1) / 2 + 1, dtype=torch.float64, device=A_complex.device)).unsqueeze(
            1).expand(BATCH_SIZE, -1, -1)  # Z轴位置
        k_lambda = -1j * 2 * torch.pi / lambda_
        sqrt_Mx = torch.sqrt(torch.tensor(1 / Mx, dtype=torch.float64, device=A_complex.device))
        sqrt_Mz = torch.sqrt(torch.tensor(1 / Mz, dtype=torch.float64, device=A_complex.device))

        xx = torch.zeros((BATCH_SIZE, M, 8), dtype=torch.complex64, device=A_complex.device)
        for q in range(index_amp.shape[1]):

            indices = index_amp[:, q]  # 这是一个形状为 (BATCH_SIZE,) 的张量，表示每个批次对应的索引

            # 使用批量索引提取相应的 g1, g2 和 r 值
            g1_batch = grid_update[torch.arange(BATCH_SIZE), indices, 0].unsqueeze(1).unsqueeze(2)
            g2_batch = grid_update[torch.arange(BATCH_SIZE), indices, 1].unsqueeze(1).unsqueeze(2)
            r_batch = grid_update[torch.arange(BATCH_SIZE), indices, 2].unsqueeze(1).unsqueeze(2)

            ax000 = (sqrt_Mx * torch.exp(
                    k_lambda * (-delta_x * d * g1_batch + delta_x ** 2 * d ** 2 * (1 - g1_batch ** 2) / (2 * r_batch)))).squeeze(2)
            az000 = (sqrt_Mz * torch.exp(
                k_lambda * (-delta_z * d * g2_batch + delta_z ** 2 * d ** 2 * (1 - g2_batch ** 2) / (2 * r_batch)))).squeeze(2)

            a_cross_term111 = torch.exp(
                (k_lambda * (-1 / r_batch)) *
                ((delta_x * d * g1_batch) @ ((delta_z * d * g2_batch).transpose(1, 2)))
            )
            a_cross_term000 = a_cross_term111.transpose(1, 2).reshape(BATCH_SIZE, -1)

            result = []
            for i in range(BATCH_SIZE):
                # 对每个 batch 的 az[i] 和 ax[i] 进行 Kronecker 积
                kronecker_product = torch.kron(az000[i].squeeze(), ax000[i].squeeze())  # (4, 1) 和 (64, 1) 的 Kronecker 积
                result.append(kronecker_product)
            # 将每个结果堆叠到一起，得到 (BATCH_SIZE, 256)
            result_tensor000 = torch.stack(result, dim=0)

            xx[:, :, q] = result_tensor000 * a_cross_term000
            for batch_idx in range(BATCH_SIZE):
                grid_index = index_amp[batch_idx, q]
                A_complex_update1[batch_idx, :, grid_index] = xx[batch_idx, :, q]

        # print(np.max((A_complex_update1 - A_complex_update).detach().cpu().numpy()))

        F_update_complex10 = A_complex_update1
        F_update_grad0000 = complex_to_real_4x4(F_update_complex10)

        input_for_vr = torch.cat((F_update_grad0000, y_torch, z_torch1), dim=-1)
        output_for_vr = self.layer3(input_for_vr)
        prob_for_vr = torch.sigmoid(output_for_vr)

        F_torch_VR = F_update_grad0000 * prob_for_vr

        return F_torch_VR


class Gradnet4(nn.Module):
    def __init__(self):
        super(Gradnet4, self).__init__()

        self.layer3 = nn.Sequential(
            nn.Linear(1026, 256),
            nn.ReLU(),
            nn.Linear(256, 1024)
        )

    def forward(self, y_torch, F_torch1, z_torch1):

        input_for_vr = torch.cat((F_torch1, y_torch, z_torch1), dim=-1)
        output_for_vr = self.layer3(input_for_vr)
        prob_for_vr = torch.sigmoid(output_for_vr)

        F_torch_VR = F_torch1 * prob_for_vr

        return F_torch_VR


class PGDnet(nn.Module):
    def __init__(self, Iter_PGD):
        super(PGDnet, self).__init__()
        self.L = Iter_PGD

        self.layers1 = nn.ModuleList()
        for i in range(self.L):
            # 每个 ConvNeXt 层使用相同的参数
            layer1 = nn.Linear(1024, 1024)
            self.layers1.append(layer1)

        self.layers2 = nn.ModuleList()
        for i in range(self.L):
            # 每个 ConvNeXt 层使用相同的参数
            layer2 = nn.Linear(1024, 1024)
            self.layers2.append(layer2)

    def forward(self, y, F, x):
        F_T = F.transpose(1, 2)
        F_T_F = F_T @ F
        F_T_y = F_T @ y
        F_T_y1 = F_T_y.squeeze(-1)
        output = x
        for i in range(self.L):
            FTFx = F_T_F @ output
            FTFx1 = FTFx.squeeze(-1)
            out1 = self.layers1[i](FTFx1)
            out2 = self.layers2[i](F_T_y1)
            output = output - out1.unsqueeze(-1) + out2.unsqueeze(-1)

        return output



