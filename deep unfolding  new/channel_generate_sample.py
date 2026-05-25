import numpy as np
import os
import matplotlib.pyplot as plt
from function import VR_generate_2D_Markov, array_response
from function import array_response_g12, complex_to_real_4x4
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDE_VISIBLE_DIVICES"] = "1"
os.environ["CUDA_DIVICE_ORDER"] = "PCI_BUS_ID"

# System Parameters
Mx = 128
Mz = 4
M = Mx * Mz  # Total number of UPA antennas
frequency = 30 * 10**9  # Frequency is 30 GHz
c = 3e8  # Speed of light
wavelength = c / frequency  # Wavelength
d = wavelength / 2  # Antenna spacing is lambda/2
D = np.sqrt(Mx**2 + Mz**2) * d  # Aperture of UPA
Rayleigh_distance = 2 * D**2 / wavelength  # Rayleigh distance

Kx = 16
Kz = 2
K = Kx * Kz  # Number of subarrays
Nx = Mx / Kx
Nz = Mz / Kz
N = Nx * Nz  # Antenna number of subarrays
# BS Center Position
Position_BS_center = np.array([0, 0, 0])
# BS Antenna Positions
Position_BS_antennas = np.zeros((Mx, Mz, 3))  # 3 for x, y, z coordinates
for m in range(Mx):
    delta_x = m - (Mx - 1) / 2
    for n in range(Mz):
        delta_z = n - (Mz - 1) / 2
        Position_BS_antennas[m, n, :] = [delta_x * d, 0, delta_z * d]

# Channel Parameters
nPath = 4
p01_x = 1 / 4
p10_x = 1 / 4
p01_z = 1 / 4
p10_z = 1 / 4
kapa = 0.9  # 2D Markov model

# Grids
g1_grid = np.linspace(-1 + 1 / Mx, 1 - 1 / Mx, Mx)  # Grid in the x-direction
g2_grid = np.linspace(-1 + 1 / Mz, 1 - 1 / Mz, Mz)  # Grid in the z-direction
# Active Index Initialization
grid_active_index = np.zeros((nPath, 2), dtype=int)

# Dataset Parameters
nSample = 1000
# Initialize Datasets
VR_set = np.zeros((Kx, Kz, nPath, nSample), dtype=int)  # Index of VR
Channel_coeff_set = np.zeros((nPath, nSample), dtype=complex)  # Channel coefficients
Theta_set = np.zeros((nPath, nSample))  # Theta angles
Phi_set = np.zeros((nPath, nSample))  # Phi angles
Distance_set = np.zeros((nPath, nSample))  # Distances
Position_scatter_set = np.zeros((nPath, 3, nSample))  # Scatterer positions (3 for x, y, z)
h_set = np.zeros((M, nSample), dtype=complex)  # Channel responses
g1_set = np.zeros((nPath, nSample))
g2_set = np.zeros((nPath, nSample))

for sample_index in range(nSample):
    # 生成 2D_Markov VR
    VR_set[:, :, :, sample_index] = VR_generate_2D_Markov(p01_x, p10_x, p01_z, p10_z, kapa, Kx, Kz, nPath)

    # 确定信道系数
    for path_index in range(nPath):
        while ( abs(Channel_coeff_set[path_index, sample_index]) < 0.8 or abs(Channel_coeff_set[path_index, sample_index]) > 1.2):
            Channel_coeff_set[path_index, sample_index] = (np.sqrt(1 / 2) * (np.random.randn() + 1j * np.random.randn()))

    # 计算散射点参数
    for path_index in range(nPath):
        Theta_set[path_index, sample_index] = 0.5 * (np.pi) * np.random.rand()
        Phi_set[path_index, sample_index] = 0.5 * (np.pi) * np.random.rand()
        g1_set[path_index, sample_index] = np.cos(Theta_set[path_index, sample_index]) * np.sin(Phi_set[path_index, sample_index])
        g2_set[path_index, sample_index] = np.cos(Phi_set[path_index, sample_index])
        Distance_set[path_index, sample_index] = 10 + (50 - 10) * np.random.rand()

    # 计算散射点位置
    Position_scatter_set[:, :, sample_index] = np.column_stack([
        Distance_set[:, sample_index] * np.cos(Theta_set[:, sample_index]) * np.sin(Phi_set[:, sample_index]),
        Distance_set[:, sample_index] * np.sin(Theta_set[:, sample_index]) * np.sin(Phi_set[:, sample_index]),
        Distance_set[:, sample_index] * np.cos(Phi_set[:, sample_index]),])

    # 计算信道响应
    for path_index in range(nPath):
        r = Distance_set[path_index, sample_index]
        theta = Theta_set[path_index, sample_index]
        phi = Phi_set[path_index, sample_index]

        # 阵列响应
        aR = array_response(r, theta, phi, Mx, Mz, wavelength, d)
        # VR 矩阵转换为向量
        VR_matrix = VR_set[:, :, path_index, sample_index]
        VR_vector = np.reshape(np.kron(VR_matrix, np.ones((int(Nx), int(Nz)))), (M,), order='F')
        # 累加信道响应
        h_set[:, sample_index] += (Channel_coeff_set[path_index, sample_index] * (aR * VR_vector))


# 定义保存路径
save_dir = f"dataset/kapa={kapa}_M={M}_nSample={nSample}_L={nPath}/"

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 保存数据集
np.save(os.path.join(save_dir, "VR_set.npy"), VR_set)
np.save(os.path.join(save_dir, "Channel_coeff_set.npy"), Channel_coeff_set)
np.save(os.path.join(save_dir, "Theta_set.npy"), Theta_set)
np.save(os.path.join(save_dir, "Phi_set.npy"), Phi_set)
np.save(os.path.join(save_dir, "Distance_set.npy"), Distance_set)
np.save(os.path.join(save_dir, "Position_scatter_set.npy"), Position_scatter_set)
np.save(os.path.join(save_dir, "g1_set.npy"), g1_set)
np.save(os.path.join(save_dir, "g2_set.npy"), g2_set)
np.save(os.path.join(save_dir, "h_set.npy"), h_set)
np.save(os.path.join(save_dir, "grid_active_index.npy"), grid_active_index)

print(f"Data saved in directory: {save_dir}")
