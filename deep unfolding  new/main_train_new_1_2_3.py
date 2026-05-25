import numpy as np
import os
import matplotlib.pyplot as plt
from function import VR_generate_2D_Markov, array_response, array_response_g12
from function import complex_to_real_4x4, complex_to_real_stack, real_to_complex_np, real_to_complex_stack, \
    real_to_complex_4x4, genarate_sub_channel, genarate_sub_channel11, \
    array_response_g12_torch, array_response_g12_torch11, function_grad_preparation11
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from mymodel1 import CustomDataset, VBInet, Gradnet31, PGDnet, Gradnet4
from thop import profile
import pandas as pd
from datetime import datetime


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDE_VISIBLE_DIVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_DIVICE_ORDER"] = "PCI_BUS_ID"

# torch.autograd.set_detect_anomaly(True)

EPOCH = 10
BATCH_SIZE = 8
LR_VBI = 0.000005
LR_Grad = 0.00001
LR_PGD = 0.0001

# System Parameters
Mx = 128  # Number of UPA antennas in x direction
Mz = 4  # Number of UPA antennas in z direction
M = Mx * Mz  # Total number of UPA antennas

frequency = 30 * 10 ** 9  # Frequency is 30 GHz
c = 3e8  # Speed of light in m/s
lambda_ = c / frequency  # Wavelength
d = lambda_ / 2  # Antenna spacing (lambda / 2)
D = np.sqrt(Mx ** 2 + Mz ** 2) * d  # Aperture of UPA
Rayleigh_distance = 2 * D ** 2 / lambda_  # Rayleigh distance

Kx = 16  # Number of subarrays in x direction
Kz = 2  # Number of subarrays in z direction
K = Kx * Kz  # Total number of subarrays

Nx = Mx / Kx  # Antenna number in x direction for each subarray (integer division)
Nz = Mz / Kz  # Antenna number in z direction for each subarray
N = Nx * Nz  # Total number of antennas per subarray

nPath = 4  # Number of paths
p01_x = 1 / 4  # Probability for transition from 0 to 1 in x direction
p10_x = 1 / 4  # Probability for transition from 1 to 0 in x direction
p01_z = 1 / 4  # Probability for transition from 0 to 1 in z direction
p10_z = 1 / 4  # Probability for transition from 1 to 0 in z direction
kapa = 0.5  # Sparsity level for 2D Markov model

# Load the datasets (replace 'dataset/kapa=...' with your directory structure)
VR_set = np.load(f'dataset/kapa={kapa}_M={M}_nSample=20000_L={nPath}/VR_set.npy')
h_set = np.load(f'dataset/kapa={kapa}_M={M}_nSample=20000_L={nPath}/h_set.npy')
grid_active_index = np.load(f'dataset/kapa={kapa}_M={M}_nSample=20000_L={nPath}/grid_active_index.npy')
print(h_set.shape)

Phi = np.load(f'dataset/Phi.npy')

mydatasets = CustomDataset(h_set, VR_set)
train_loader = data.DataLoader(dataset=mydatasets, batch_size=BATCH_SIZE, shuffle=True)
torch.set_printoptions(precision=8)

# 极坐标网格
g1_grid = np.linspace(-1 + 1 / Mx, 1 - 1 / Mx, Mx)  # g1_grid: 从 -1+1/Mx 到 1-1/Mx，Mx个点
g2_grid = np.linspace(-1 + 1 / Mz, 1 - 1 / Mz, Mz)  # g2_grid: 从 -1+1/Mz 到 1-1/Mz，Mz个点
r_grid = np.arange(10, 11, 5)
nGrid = len(g1_grid) * len(g2_grid) * len(r_grid)  # 网格总数
polar_grid = np.zeros((nGrid, 3))  # 用于存储组合的Theta、Phi和Distance
grid_index = 0
# 三重循环遍历所有可能的g1, g2和r值
for m in range(len(g1_grid)):
    for n in range(len(g2_grid)):
        for i in range(len(r_grid)):
            grid_index += 1
            polar_grid[grid_index - 1, :] = [g1_grid[m], g2_grid[n], r_grid[i]]  # 存储每个组合的值

polar_grid_batch = np.tile(polar_grid, (BATCH_SIZE, 1, 1))
polar_grid_torch = torch.from_numpy(polar_grid_batch).float().cuda()

# Array_response 基础
A = np.zeros((M, nGrid), dtype=complex)  # 创建一个大小为 (M, nGrid) 的复数零矩阵

for grid_index in range(nGrid):
    g1 = polar_grid[grid_index, 0]  # 获取 polar_grid 中的 g1
    g2 = polar_grid[grid_index, 1]  # 获取 polar_grid 中的 g2
    r = polar_grid[grid_index, 2]  # 获取 polar_grid 中的 r
    A[:, grid_index] = array_response_g12(r, g1, g2, Mx, Mz, lambda_, d)  # 调用 array_response_g12 函数

A_batch = np.tile(A, (BATCH_SIZE, 1, 1))
Phi_batch = np.tile(Phi, (BATCH_SIZE, 1, 1))
Phi_torch = torch.from_numpy(Phi_batch).to(torch.complex128).cuda()
# VR_matrix
U = np.ones((BATCH_SIZE, M, nGrid))

A_torch = torch.from_numpy(A_batch)
U_torch = torch.from_numpy(U)
A_U_product = A_torch * U_torch
F = A_U_product

F_torch = complex_to_real_4x4(F)
A_torch1 = complex_to_real_4x4(A_torch).cuda()
A_torch_real = complex_to_real_4x4(A_torch.cuda())
pinv_F_torch = torch.linalg.pinv(F_torch)

# v = 0.5 * np.ones((Kx, Kz, nGrid))
# v_batch = np.tile(v, (BATCH_SIZE, 1, 1, 1))
# V_torch = torch.from_numpy(v_batch).float()

PI = 2 * (nPath / nGrid) * np.ones(nGrid * 2)
PI_torch = torch.from_numpy(PI)
PI_torch = PI_torch.unsqueeze(0).unsqueeze(-1).expand(BATCH_SIZE, 2 * M, 1)
PI_torch = PI_torch.to(torch.float32)

# 初始化 NMSE 和 Error 相关数组
# NMSE_proposed = np.zeros(len(SNR_set))

# Simulations 初始化
# NMSE_H = np.zeros((outside_loop, nSample, len(SNR_set)))
# VR_EST = np.zeros((Kx, Kz, nGrid, outside_loop, len(SNR_set), nSample), dtype=bool)
# Error_rate = np.zeros((outside_loop, nSample, len(SNR_set)))

device_ids = [0]
VBI_net3 = VBInet(Iter_VBI=1, in_chans_VBI=1, depths=[1, 1, 1, 1], dims=[64, 128, 256, 512], num_classes_L=1,
                    num_classes_rou=nGrid * 2)
VBI3 = nn.DataParallel(VBI_net3, device_ids=device_ids).cuda()
VBI3_optimizer = torch.optim.Adam(VBI3.parameters(), lr=LR_VBI)
VBI3.load_state_dict(torch.load('./model_pt/VBI3_hybrid_kapa=0_5_L4_Q32_i1_framework34.pt', weights_only=True))


Grad_net = Gradnet4()
Grad4 = nn.DataParallel(Grad_net, device_ids=device_ids).cuda()
Grad4_optimizer = torch.optim.Adam(Grad4.parameters(), lr=LR_Grad)
Grad4.load_state_dict(torch.load('./model_pt/Grad4_hybrid_kapa=0_5_L4_Q32_i1_framework34.pt', weights_only=True))


PGD_net = PGDnet(Iter_PGD=1)
PGD = nn.DataParallel(PGD_net, device_ids=device_ids).cuda()
PGD_optimizer = torch.optim.Adam(PGD.parameters(), lr=LR_PGD)
# PGD_scheduler = optim.lr_scheduler.StepLR(PGD_optimizer, step_size=2, gamma=0.1)

Loss = nn.MSELoss()
Loss.cuda()

Phi_torch_pinv = torch.linalg.pinv(Phi_torch)

# F_white = Phi_torch_pinv @ Phi_torch @ (A_U_product.cuda())
# F_torch_white = complex_to_real_4x4(F_white).to(torch.float32)
# F_white_pinv = torch.linalg.pinv(F_white)

# F_white_pinv_one = np.load(f'dataset/F_white_pinv.npy')
# F_white_pinv = torch.from_numpy(F_white_pinv_one).repeat(BATCH_SIZE, 1, 1).cuda()

# torch.max(torch.abs(F_white_pinv - F_white_pinv_one_batch))

A_FPN_OAMP = Phi_torch @ F.cuda()

real_A_FPN_OAMP = torch.real(A_FPN_OAMP)
imag_A_FPN_OAMP = torch.imag(A_FPN_OAMP)

M_FPN_OAMP = torch.zeros((BATCH_SIZE, 512, 1024), dtype=torch.float32, device=A_FPN_OAMP.device)

# 按照给定公式填充 result 矩阵
M_FPN_OAMP[:, 0:256, 0:512] = real_A_FPN_OAMP  # 第一部分: \Re(F)
M_FPN_OAMP[:, 0:256, 512:2 * 512] = -imag_A_FPN_OAMP  # 第二部分: -\Im(F)
M_FPN_OAMP[:, 256:2 * 256, 0:512] = imag_A_FPN_OAMP  # 第三部分: \Im(F)
M_FPN_OAMP[:, 256:2 * 256, 512:2 * 512] = real_A_FPN_OAMP  # 第四部分: \Re(F)

M_FPN_OAMP_pinv = torch.linalg.pinv(M_FPN_OAMP)

# 记录 batch 日志
batch_logs = []

# 记录 epoch 日志
epoch_logs = []

T1 = time.perf_counter()
Step = 0
print('---------------------------------------Start Train-------------------------------------------')
for epoch in range(EPOCH):
    epoch_mse_sum = 0
    epoch_nmse_sum = 0
    epoch_batch_count = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        h = inputs.reshape(BATCH_SIZE, M, 1)

        # VR_real = targets.cuda()

        points = torch.tensor([-5, 0, 5, 10, 15])
        snr = points[torch.randint(0, 5, (BATCH_SIZE, 1))]  # BATCH_SIZE 是你需要的批次大小

        # snr = 10 + (torch.linspace(0, 1, BATCH_SIZE).view(-1, 1) * (
        #         10.000001 - 10))  # 生成固定间隔的 SNR 值，范围在 [-10, 10] 之间，形状为 (BATCH_SIZE, 1)

        norm_h = torch.norm(h, p=2, dim=1, keepdim=True).squeeze(2)  # 计算每个样本的 L2 范数，维度 (batch_size, 1)
        sigma2 = (norm_h ** 2) / M / (10 ** (snr / 10))  # 按批次计算噪声方差，sigma2 形状为 (batch_size, 1)

        # 生成实部和虚部噪声，形状为 (batch_size, M, 1)
        real_noise = torch.randn(BATCH_SIZE, M, 1)  # 生成实部噪声
        imag_noise = torch.randn(BATCH_SIZE, M, 1)  # 生成虚部噪声

        # 归一化并调整噪声幅度
        noise = (real_noise + 1j * imag_noise) / torch.sqrt(torch.tensor(2.0))
        noise = torch.sqrt(sigma2.unsqueeze(1)) * noise

        y = (h + noise)
        y_complex = y.cuda()
        y_hybrid = torch.bmm(Phi_torch, y_complex)
        y_white = Phi_torch_pinv @ y_hybrid
        # z_white = F_white_pinv @ y_white

        y_torch_hybrid = complex_to_real_stack(y_hybrid).to(torch.float32)
        y_torch = complex_to_real_stack(y_white).to(torch.float32)
        # z_torch = complex_to_real_stack(z_white).to(torch.float32)
        # z_torch = torch.matmul(pinv_F_torch, y_torch)
        h_torch_real = complex_to_real_stack(h).to(torch.float32)
        # noise_torch = complex_to_real_stack(noise).to(torch.float32)
        sigma2 = sigma2.to(torch.float32).cuda()

        z_torch1 = M_FPN_OAMP_pinv @ y_torch_hybrid

        #############################################################################
        F = M_FPN_OAMP
        z = z_torch1
        z_torch_VBI, F_torch_VBI, loss_VBI_net = VBI3(y_torch_hybrid, sigma2, PI_torch, F, z, h_torch_real)

        x_est = z_torch_VBI
        h_est = torch.bmm(F_torch.cuda(), x_est.cuda())

        error = h_torch_real.cuda() - h_est
        mse = torch.mean(error ** 2, dim=1)  # 计算每个样本的均方误差，形状为 (batch_size, 1)
        energy_real = torch.mean(h_torch_real.cuda() ** 2, dim=1)  # 计算每个样本的能量，形状为 (batch_size, 1)
        nmse = mse / energy_real
        nmse_db = 10 * torch.log10(nmse)  # 为防止 log(0) 加上小常数 1e-10
        final_nmse_VBI_1 = torch.mean(nmse_db)  # 如果需要，可以对所有样本的 NMSE 取平均值

        #################################################################################
        F_torch2 = F_torch.cuda()
        z_torch2 = z_torch_VBI

        F_update_grad = Grad4(y_torch, F_torch, z_torch_VBI)

        F_est1 = F_update_grad
        h_est1 = torch.bmm(F_est1.cuda(), z_torch2)
        # MSE_Grad_1 = Loss(h_torch_real.cuda(), h_est1)

        error = h_torch_real.cuda() - h_est1
        mse = torch.mean(error ** 2, dim=1)  # 计算每个样本的均方误差，形状为 (batch_size, 1)
        energy_real = torch.mean(h_torch_real.cuda() ** 2, dim=1)  # 计算每个样本的能量，形状为 (batch_size, 1)
        nmse = mse / energy_real
        nmse_db = 10 * torch.log10(nmse)  # 为防止 log(0) 加上小常数 1e-10
        final_nmse_Grad_1 = torch.mean(nmse_db)  # 如果需要，可以对所有样本的 NMSE 取平均值
        #################################################################################
        # F_PGD = F_update_grad.clone().detach().cuda()
        F_PGD = F_update_grad
        z_PGD = z_torch_VBI
        z_torch_PGD = PGD(y_torch, F_PGD, z_PGD)

        # PGD_net = PGDnet(Iter_PGD=1)
        # PGD = nn.DataParallel(PGD_net, device_ids=device_ids).cuda()
        #
        # macs, params = profile(PGD_net, inputs=(y_torch, F.cuda(), z.cuda()))
        # print(f"MACs: {macs * 2 / 1e9:.8f} G, Params: {params / 1e6:.8f} M")

        x_est = z_torch_PGD
        h_est_PGD = torch.bmm(F_PGD, x_est)
        MSE_PGD_1 = Loss(h_torch_real.cuda(), h_est_PGD)

        if epoch < 10:
            PGD_optimizer.zero_grad()
            MSE_PGD_1.backward()
            PGD_optimizer.step()
        else:
            PGD_optimizer.zero_grad()
            Grad4_optimizer.zero_grad()
            VBI3_optimizer.zero_grad()
            MSE_PGD_1.backward()
            PGD_optimizer.step()
            Grad4_optimizer.step()
            VBI3_optimizer.step()

        error = h_torch_real.cuda() - h_est_PGD
        mse = torch.mean(error ** 2, dim=1)  # 计算每个样本的均方误差，形状为 (batch_size, 1)
        energy_real = torch.mean(h_torch_real.cuda() ** 2, dim=1)  # 计算每个样本的能量，形状为 (batch_size, 1)
        nmse = mse / energy_real
        nmse_db = 10 * torch.log10(nmse)  # 为防止 log(0) 加上小常数 1e-10
        final_nmse_PGD_1 = torch.mean(nmse_db)  # 如果需要，可以对所有样本的 NMSE 取平均值

        #################################################################################

        if Step % 10 == 0:
            T2 = time.perf_counter()
            current_lr = PGD_optimizer.param_groups[0]['lr']
            print(f"[epoch {epoch + 1}][{batch_idx + 1}/{len(train_loader)}]    "
                    f"Time: {(T2 - T1):.4f}    "
                    # f"loss_VBI: {MSE_VBI_1:.8f}    "
                    # f"loss_Grad: {MSE_Grad_1:.8f}    "
                    f"loss_PGD: {MSE_PGD_1:.8f}    "
                    f"NMSE: {final_nmse_VBI_1:.8f}    "
                    f"NMSE2: {final_nmse_Grad_1:.8f}    "
                    f"NMSE: {final_nmse_PGD_1:.8f}    "
                    f"Learning Rate: {current_lr}"
                    )

        if batch_idx == 0:
            print('---------------------------------------Test Start-------------------------------------------')
            print("NMSE", final_nmse_PGD_1)
            print('---------------------------------------Test end-------------------------------------------')

        mse_value = MSE_PGD_1.item()
        nmse_value = final_nmse_PGD_1.item()

        # 保存 batch 数据
        batch_logs.append({
            "epoch": epoch + 1,
            "batch": batch_idx + 1,
            "step": Step,
            "MSE_VBI": mse_value,
            "NMSE_dB": nmse_value
        })

        # epoch统计
        epoch_mse_sum += mse_value
        epoch_nmse_sum += nmse_value
        epoch_batch_count += 1

        Step += 1

    # VBI_scheduler.step()
    # PGD_scheduler.step()
    # Grad_scheduler.step()

    epoch_mse_avg = epoch_mse_sum / epoch_batch_count
    epoch_nmse_avg = epoch_nmse_sum / epoch_batch_count

    epoch_logs.append({
        "epoch": epoch + 1,
        "MSE_VBI_avg": epoch_mse_avg,
        "NMSE_dB_avg": epoch_nmse_avg
    })

    print(f"Epoch {epoch + 1} Avg MSE: {epoch_mse_avg:.8f}, Avg NMSE: {epoch_nmse_avg:.8f}")

torch.save(VBI3.state_dict(), './model_pt/VBI3_hybrid_kapa=0_5_L4_Q32.pt')
torch.save(Grad4.state_dict(), './model_pt/Grad4_hybrid_kapa=0_5_L4_Q32.pt')
torch.save(PGD.state_dict(), './model_pt/PGD_hybrid_kapa=0_5_L4_Q32.pt')


run_name = "PGD_hybrid_kapa=0_5_L4_Q32"

base_out_dir = "./train_logs"
os.makedirs(base_out_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
out_dir = os.path.join(base_out_dir, f"{run_name}_{timestamp}")
os.makedirs(out_dir, exist_ok=True)

print("Logs will be saved to:", out_dir)


batch_df = pd.DataFrame(batch_logs)
epoch_df = pd.DataFrame(epoch_logs)

batch_csv_path = os.path.join(out_dir, "batch_log.csv")
epoch_csv_path = os.path.join(out_dir, "epoch_log.csv")

batch_df.to_csv(batch_csv_path, index=False)
epoch_df.to_csv(epoch_csv_path, index=False)

print("CSV logs saved.")

plt.figure(figsize=(12,5))

# =========================
# Batch 曲线
# =========================
ax1 = plt.subplot(1,2,1)

# 左轴：MSE
line1 = ax1.plot(batch_df["step"], batch_df["MSE_VBI"],
                 color="tab:blue", linewidth=2, label="MSE")
ax1.set_xlabel("Step")
ax1.set_ylabel("MSE", color="tab:blue")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax1.grid()

# 右轴：NMSE
ax1_r = ax1.twinx()
line2 = ax1_r.plot(batch_df["step"], batch_df["NMSE_dB"],
                   color="tab:red", linestyle="--", linewidth=2, label="NMSE(dB)")
ax1_r.set_ylabel("NMSE (dB)", color="tab:red")
ax1_r.tick_params(axis='y', labelcolor="tab:red")

# 图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels)

ax1.set_title("Batch Training Curve")

# =========================
# Epoch 曲线
# =========================
ax2 = plt.subplot(1,2,2)

# 左轴：MSE
line1 = ax2.plot(epoch_df["epoch"], epoch_df["MSE_VBI_avg"],
                 color="tab:blue", linewidth=2, label="MSE")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MSE", color="tab:blue")
ax2.tick_params(axis='y', labelcolor="tab:blue")
ax2.grid()

# 右轴：NMSE
ax2_r = ax2.twinx()
line2 = ax2_r.plot(epoch_df["epoch"], epoch_df["NMSE_dB_avg"],
                   color="tab:red", linestyle="--", linewidth=2, label="NMSE(dB)")
ax2_r.set_ylabel("NMSE (dB)", color="tab:red")
ax2_r.tick_params(axis='y', labelcolor="tab:red")

# 图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels)

ax2.set_title("Epoch Average Curve")

plt.tight_layout()
plt.show()

fig_path = os.path.join(out_dir, "training_curve.png")
plt.savefig(fig_path, dpi=300)

print("Training curves saved.")

