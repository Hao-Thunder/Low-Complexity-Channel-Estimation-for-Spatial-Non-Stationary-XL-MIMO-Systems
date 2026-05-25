import numpy as np
import os
import torch
import torch.cuda.amp as amp
from torch.distributions import Normal
from scipy.special import gammaln, psi
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import matplotlib.pyplot as plt


def array_response(r, theta, phi, Mx, Mz, wavelength, d):
    # near field array response vector
    delta_x = np.arange(-(Mx - 1) / 2, (Mx - 1) / 2 + 1)
    delta_z = np.arange(-(Mz - 1) / 2, (Mz - 1) / 2 + 1)

    g1 = np.cos(theta) * np.sin(phi)
    g2 = np.cos(phi)

    ax = np.sqrt(1 / Mx) * np.exp(
        (-1j * 2 * np.pi / wavelength) * (-delta_x * d * g1 + delta_x ** 2 * d ** 2 * (1 - g1 ** 2) / (2 * r)))
    az = np.sqrt(1 / Mz) * np.exp(
        (-1j * 2 * np.pi / wavelength) * (-delta_z * d * g2 + delta_z ** 2 * d ** 2 * (1 - g2 ** 2) / (2 * r)))

    # Cross term calculation
    a_cross_term = np.reshape(
        np.exp((-1j * 2 * np.pi / wavelength) * (-1 / r) * (delta_x * d * g1)[:, np.newaxis] * (delta_z * d * g2)),
        (-1,), order='F')

    # Final array response
    a = np.kron(az, ax) * a_cross_term

    return a


def Grad_update111(y, A, U, F, Mx, Mz, x_post, sigma2, grid, lambda_, d, nPath, active_index ):
    # Extract parameters from sigPar dictionary
    # y = sigPar['y']
    # A = sigPar['A']
    # U = sigPar['U']
    # F = sigPar['F']
    M, nGrid = F.shape
    # Mx = sigPar['Mx']
    # Mz = sigPar['Mz']
    # x_post = sigPar['x_post']
    # sigma2 = max(sigPar['sigma2'], 0.005)
    # grid = sigPar['grid']
    # lambda_ = sigPar['lambda']  # wave length
    # d = sigPar['d']  # antenna space
    # nPath = sigPar['nPath']
    # active_index = sigPar['active_index']

    # Sort and index selection
    sort_index = np.argsort(np.abs(x_post) ** 2)[::-1]
    index_amp = sort_index[:2 * nPath]
    # index_dif = active_index[-1]
    # index_amp = np.setdiff1d(index_amp, index_dif)

    # Gradient ascent method w.r.t theta
    grid_update = np.copy(grid)
    grad_F_to_g1 = np.zeros((M, nGrid), dtype=complex)
    grad_F_to_g2 = np.zeros((M, nGrid), dtype=complex)
    delta_x = np.arange(-(Mx - 1) / 2, (Mx - 1) / 2 + 1).reshape(-1, 1, order='F')
    delta_z = np.arange(-(Mz - 1) / 2, (Mz - 1) / 2 + 1).reshape(-1, 1, order='F')

    # Loop over each index in index_amp
    for q in range(len(index_amp)):
        grid_index = index_amp[q]
        g1_q = grid[grid_index, 0]
        g2_q = grid[grid_index, 1]
        r_q = grid[grid_index, 2]

        # Gradient calculation for g1 and g2
        grad_F_to_g1[:, grid_index] = (np.reshape(F[:, grid_index],(-1, 1), order='F') *
            (np.kron(np.ones((Mz, 1)), 1j * 2 * np.pi / lambda_ * (delta_x * d + delta_x ** 2 * d ** 2 * g1_q / r_q)) +
            np.reshape((1j * 2 * np.pi / lambda_ / r_q) * ((delta_x * d) @ (np.dot(delta_z.T, d) * g2_q)), (-1, 1), order='F') ) ).flatten()

        grad_F_to_g2[:, grid_index] = (np.reshape(F[:, grid_index],(-1, 1), order='F') * (
                    np.kron(  np.reshape((1j * 2 * np.pi / lambda_ * (delta_z * d + delta_z ** 2 * d ** 2 * g2_q / r_q)),(-1, 1), order='F' ) , np.ones((Mx, 1))) +
                    np.reshape((1j * 2 * np.pi / lambda_ / r_q) * (delta_x * d * g1_q) @ (np.dot(delta_z.T, d)), (-1, 1), order='F'))).flatten()

        # Gradient of Q w.r.t g1 and g2
        grad_Q_to_g1 = -2 / sigma2 * np.real( -np.conj(x_post[grid_index].T) * grad_F_to_g1[:, grid_index].conj().T @ (y - F @ x_post))

        grad_Q_to_g2 = -2 / sigma2 * np.real( -np.conj(x_post[grid_index].T) * grad_F_to_g2[:, grid_index].conj().T @ (y - F @ x_post))

        # Update grid positions based on the gradient
        grid_update[grid_index, 0] = g1_q + 2 / Mx / 40 * np.sign(grad_Q_to_g1)
        grid_update[grid_index, 1] = g2_q + 2 / Mz / 40 * np.sign(grad_Q_to_g2)

    # Update sensing matrix A
    A_update = np.copy(A)
    for q in range(len(index_amp)):
        grid_index = index_amp[q]
        A_update[:, grid_index] = array_response_g12(grid_update[grid_index, 2], grid_update[grid_index, 0],grid_update[grid_index, 1], Mx, Mz, lambda_, d)

    # Update F matrix
    F_update = A_update * U

    # # Return updated sigPar
    # sigPar['grid'] = grid_update
    # sigPar['A'] = A_update
    # sigPar['F'] = F_update

    return F_update, grid_update, grad_F_to_g1, grad_F_to_g2


def VR_generate_2D_Markov(p01_x, p10_x, p01_z, p10_z, kapa, Kx, Kz, L):
    """
    Generate a 2D Markov VR matrix with specified transition probabilities and sparsity level.

    Parameters:
    - p01_x: Transition probability for x-direction (from 0 to 1)
    - p10_x: Transition probability for x-direction (from 1 to 0)
    - p01_z: Transition probability for z-direction (from 0 to 1)
    - p10_z: Transition probability for z-direction (from 1 to 0)
    - kapa: Sparsity level
    - Kx: Number of subarrays in the x-direction
    - Kz: Number of subarrays in the z-direction
    - L: Number of paths

    Returns:
    - VR: A 3D array representing the generated 2D Markov VR matrix
    """

    # Initialize VR array with zeros
    VR = np.zeros((Kx, Kz, L))

    for i in range(L):
        # Repeat until the sum of VR elements is within the desired sparsity range
        while np.sum(VR[:, :, i]) < (kapa - 0.1) * Kx * Kz or np.sum(VR[:, :, i]) > (kapa + 0.1) * Kx * Kz:

            # Initialize the first element
            VR[0, 0, i] = np.random.rand() <= kapa

            # Generate first row
            for kz in range(1, Kz):  # first row
                if VR[0, kz - 1, i] == 1:
                    VR[0, kz, i] = np.random.rand() <= (1 - p10_z)
                else:
                    VR[0, kz, i] = np.random.rand() <= p01_z

            # Generate first column
            for kx in range(1, Kx):  # first column
                if VR[kx - 1, 0, i] == 1:
                    VR[kx, 0, i] = np.random.rand() <= (1 - p10_x)
                else:
                    VR[kx, 0, i] = np.random.rand() <= p01_x

            # Generate other elements
            for kz in range(1, Kz):
                for kx in range(1, Kx):
                    # Calculate transition probabilities
                    if VR[kx, kz - 1, i] == 1:
                        p1_z = 1 - p10_z
                    else:
                        p1_z = p01_z

                    if VR[kx - 1, kz, i] == 1:
                        p1_x = 1 - p10_x
                    else:
                        p1_x = p01_x

                    p1 = p1_x * p1_z / (p1_x * p1_z + (1 - p1_x) * (1 - p1_z))
                    VR[kx, kz, i] = np.random.rand() <= p1

    return VR


def array_response_g12(r, g1, g2, Mx, Mz, lambda_, d):
    # Near field array response vector
    delta_x = np.arange(-(Mx - 1) / 2, (Mx - 1) / 2 + 1)  # X-axis positions
    delta_z = np.arange(-(Mz - 1) / 2, (Mz - 1) / 2 + 1)  # Z-axis positions

    # Ax and Az calculations
    ax = np.sqrt(1 / Mx) * np.exp(
        (-1j * 2 * np.pi / lambda_) * (-delta_x * d * g1 + delta_x ** 2 * d ** 2 * (1 - g1 ** 2) / (2 * r)))
    az = np.sqrt(1 / Mz) * np.exp(
        (-1j * 2 * np.pi / lambda_) * (-delta_z * d * g2 + delta_z ** 2 * d ** 2 * (1 - g2 ** 2) / (2 * r)))

    # Cross term calculation
    a_cross_term1 = np.exp(
        (-1j * 2 * np.pi / lambda_) * (-1 / r) * (delta_x * d * g1).reshape(-1, 1, order='F') * (delta_z * d * g2).reshape(1, -1, order='F'))
    a_cross_term = a_cross_term1.reshape(-1, order='F')

    # Compute final response
    a = np.kron(az, ax) * a_cross_term

    return a


def array_response_g12_torch(r, g1, g2, Mx, Mz, lambda_, d):
    # BATCH_SIZE = r.shape[0]
    # 使用torch创建X轴和Z轴的位置数组
    delta_x = torch.arange(-(Mx - 1) / 2, (Mx - 1) / 2 + 1, dtype=torch.float64, device=r.device)  # X轴位置
    delta_z = torch.arange(-(Mz - 1) / 2, (Mz - 1) / 2 + 1, dtype=torch.float64, device=r.device)  # Z轴位置
    # a = torch.zeros((Mx * Mz), dtype=torch.complex64, device=r.device)

    # 预计算常见的数值
    k_lambda = -1j * 2 * torch.pi / lambda_
    sqrt_Mx = torch.sqrt(torch.tensor(1 / Mx, dtype=torch.float64, device=r.device))
    sqrt_Mz = torch.sqrt(torch.tensor(1 / Mz, dtype=torch.float64, device=r.device))

    g1_batch = g1
    g2_batch = g2
    r_batch = r

    # 计算 ax 和 az
    ax = sqrt_Mx * torch.exp(
            k_lambda * (-delta_x * d * g1_batch + delta_x ** 2 * d ** 2 * (1 - g1_batch ** 2) / (2 * r_batch)))
    az = sqrt_Mz * torch.exp(
            k_lambda * (-delta_z * d * g2_batch + delta_z ** 2 * d ** 2 * (1 - g2_batch ** 2) / (2 * r_batch)))

    a_cross_term1 = torch.exp(
            (k_lambda * (-1 / r_batch)) *
            ((delta_x * d * g1_batch).unsqueeze(1) @ (delta_z * d * g2_batch).unsqueeze(1).t())
        )
    a_cross_term = a_cross_term1.t().reshape(-1, )
    a = torch.kron(az, ax) * a_cross_term

    return a


def array_response_g12_torch11(r, g1, g2, Mx, Mz, lambda_, d):
    BATCH_SIZE = r.shape[0]
    # 使用torch创建X轴和Z轴的位置数组
    delta_x = torch.arange(-(Mx - 1) / 2, (Mx - 1) / 2 + 1, device=r.device)  # X轴位置
    delta_z = torch.arange(-(Mz - 1) / 2, (Mz - 1) / 2 + 1, device=r.device)  # Z轴位置
    a = torch.zeros((BATCH_SIZE, Mx * Mz), dtype=torch.complex64, device=r.device)

    # 预计算常见的数值
    k_lambda = -1j * 2 * torch.pi / lambda_
    sqrt_Mx = torch.sqrt(torch.tensor(1 / Mx, dtype=torch.cfloat, device=r.device))
    sqrt_Mz = torch.sqrt(torch.tensor(1 / Mz, dtype=torch.cfloat, device=r.device))

    for batch_idx in range(BATCH_SIZE):
        g1_batch = g1[batch_idx]
        g2_batch = g2[batch_idx]
        r_batch = r[batch_idx]

        # 计算 ax 和 az
        ax = sqrt_Mx * torch.exp(
            k_lambda * (-delta_x * d * g1_batch + delta_x ** 2 * d ** 2 * (1 - g1_batch ** 2) / (2 * r_batch)))
        az = sqrt_Mz * torch.exp(
            k_lambda * (-delta_z * d * g2_batch + delta_z ** 2 * d ** 2 * (1 - g2_batch ** 2) / (2 * r_batch)))

        a_cross_term1 = torch.exp(
            (k_lambda * (-1 / r_batch)) *
            ((delta_x * d * g1_batch).unsqueeze(1) @ (delta_z * d * g2_batch).unsqueeze(1).t())
        )
        a_cross_term = a_cross_term1.t().reshape(-1, )
        a[batch_idx, :] = torch.kron(az, ax) * a_cross_term

    return a


def MP_in_2D_Markov(lamb_in, sigPar):
    # Model parameters
    kapa = sigPar['kapa']
    p01_x = sigPar['p01_x']
    p10_x = sigPar['p10_x']
    p01_z = sigPar['p01_z']
    p10_z = sigPar['p10_z']
    Kx = sigPar['Kx']
    Kz = sigPar['Kz']

    # Message initialization
    gamma_l = 0.5 * np.ones((Kx, Kz))
    gamma_r = 0.5 * np.ones((Kx, Kz))
    gamma_t = 0.5 * np.ones((Kx, Kz))
    gamma_b = 0.5 * np.ones((Kx, Kz))
    gamma_t[0, 0] = kapa

    # MP in 2D-Markov
    for iter in range(2):
        # Calculate gamma_l
        for kz in range(1, Kz):
            for kx in range(Kx):
                gamma_l[kx, kz] = ((1 - p10_z) * lamb_in[kx, kz - 1] * gamma_l[kx, kz - 1] * gamma_t[kx, kz - 1] * gamma_b[kx, kz - 1] +
                                    p01_z * (1 - lamb_in[kx, kz - 1]) * (1 - gamma_l[kx, kz - 1]) * (1 - gamma_t[kx, kz - 1]) * (1 - gamma_b[kx, kz - 1])) / \
                                    (lamb_in[kx, kz - 1] * gamma_l[kx, kz - 1] * gamma_t[kx, kz - 1] * gamma_b[kx, kz - 1] +
                                    (1 - lamb_in[kx, kz - 1]) * (1 - gamma_l[kx, kz - 1]) * (1 - gamma_t[kx, kz - 1]) * (1 - gamma_b[kx, kz - 1]))

        # Calculate gamma_r
        for kz in range(Kz - 1):
            for kx in range(Kx):
                gamma_r[kx, kz] = ((1 - p10_z) * lamb_in[kx, kz + 1] * gamma_r[kx, kz + 1] * gamma_t[kx, kz + 1] * gamma_b[kx, kz + 1] +
                                    p10_z * (1 - lamb_in[kx, kz + 1]) * (1 - gamma_r[kx, kz + 1]) * (1 - gamma_t[kx, kz + 1]) * (1 - gamma_b[kx, kz + 1])) / \
                                    ((1 - p10_z + p01_z) * lamb_in[kx, kz + 1] * gamma_r[kx, kz + 1] * gamma_t[kx, kz + 1] * gamma_b[kx, kz + 1] +
                                    (1 - p01_z + p10_z) * (1 - lamb_in[kx, kz + 1]) * (1 - gamma_r[kx, kz + 1]) * (1 - gamma_t[kx, kz + 1]) * (1 - gamma_b[kx, kz + 1]))

        # Calculate gamma_t
        for kx in range(1, Kx):
            for kz in range(Kz):
                gamma_t[kx, kz] = ((1 - p10_x) * lamb_in[kx - 1, kz] * gamma_l[kx - 1, kz] * gamma_r[kx - 1, kz] * gamma_t[kx - 1, kz] +
                                    p01_x * (1 - lamb_in[kx - 1, kz]) * (1 - gamma_l[kx - 1, kz]) * (1 - gamma_r[kx - 1, kz]) * (1 - gamma_t[kx - 1, kz])) / \
                                    (lamb_in[kx - 1, kz] * gamma_l[kx - 1, kz] * gamma_r[kx - 1, kz] * gamma_t[kx - 1, kz] +
                                    (1 - lamb_in[kx - 1, kz]) * (1 - gamma_l[kx - 1, kz]) * (1 - gamma_r[kx - 1, kz]) * (1 - gamma_t[kx - 1, kz]))

        # Calculate gamma_b
        for kx in range(Kx - 1):
            for kz in range(Kz):
                gamma_b[kx, kz] = ((1 - p10_x) * lamb_in[kx + 1, kz] * gamma_l[kx + 1, kz] * gamma_r[kx + 1, kz] * gamma_b[kx + 1, kz] +
                                    p10_x * (1 - lamb_in[kx + 1, kz]) * (1 - gamma_l[kx + 1, kz]) * (1 - gamma_r[kx + 1, kz]) * (1 - gamma_b[kx + 1, kz])) / \
                                    ((1 - p10_x + p01_x) * lamb_in[kx + 1, kz] * gamma_l[kx + 1, kz] * gamma_r[kx + 1, kz] * gamma_b[kx + 1, kz] +
                                    (1 - p01_x + p10_x) * (1 - lamb_in[kx + 1, kz]) * (1 - gamma_l[kx + 1, kz]) * (1 - gamma_r[kx + 1, kz]) * (1 - gamma_b[kx + 1, kz]))

    # Calculate lamb_out
    lamb_out = (gamma_l * gamma_r * gamma_t * gamma_b) / \
                (gamma_l * gamma_r * gamma_t * gamma_b + (1 - gamma_l) * (1 - gamma_r) * (1 - gamma_t) * (1 - gamma_b))

    return lamb_out


def MP_in_2D_Markov11(lamb_in, Kx, Kz):
    # Model parameters
    kapa = 0.5
    p01_x = 1/4
    p10_x = 1/4
    p01_z = 1/4
    p10_z = 1/4

    # Message initialization
    gamma_l = 0.5 * np.ones((Kx, Kz))
    gamma_r = 0.5 * np.ones((Kx, Kz))
    gamma_t = 0.5 * np.ones((Kx, Kz))
    gamma_b = 0.5 * np.ones((Kx, Kz))
    gamma_t[0, 0] = kapa

    # MP in 2D-Markov
    for iter in range(2):
        # Calculate gamma_l
        for kz in range(1, Kz):
            for kx in range(Kx):
                gamma_l[kx, kz] = ((1 - p10_z) * lamb_in[kx, kz - 1] * gamma_l[kx, kz - 1] * gamma_t[kx, kz - 1] * gamma_b[kx, kz - 1] +
                                    p01_z * (1 - lamb_in[kx, kz - 1]) * (1 - gamma_l[kx, kz - 1]) * (1 - gamma_t[kx, kz - 1]) * (1 - gamma_b[kx, kz - 1])) / \
                                    (lamb_in[kx, kz - 1] * gamma_l[kx, kz - 1] * gamma_t[kx, kz - 1] * gamma_b[kx, kz - 1] +
                                    (1 - lamb_in[kx, kz - 1]) * (1 - gamma_l[kx, kz - 1]) * (1 - gamma_t[kx, kz - 1]) * (1 - gamma_b[kx, kz - 1]))

        # Calculate gamma_r
        for kz in range(Kz - 1):
            for kx in range(Kx):
                gamma_r[kx, kz] = ((1 - p10_z) * lamb_in[kx, kz + 1] * gamma_r[kx, kz + 1] * gamma_t[kx, kz + 1] * gamma_b[kx, kz + 1] +
                                    p10_z * (1 - lamb_in[kx, kz + 1]) * (1 - gamma_r[kx, kz + 1]) * (1 - gamma_t[kx, kz + 1]) * (1 - gamma_b[kx, kz + 1])) / \
                                    ((1 - p10_z + p01_z) * lamb_in[kx, kz + 1] * gamma_r[kx, kz + 1] * gamma_t[kx, kz + 1] * gamma_b[kx, kz + 1] +
                                    (1 - p01_z + p10_z) * (1 - lamb_in[kx, kz + 1]) * (1 - gamma_r[kx, kz + 1]) * (1 - gamma_t[kx, kz + 1]) * (1 - gamma_b[kx, kz + 1]))

        # Calculate gamma_t
        for kx in range(1, Kx):
            for kz in range(Kz):
                gamma_t[kx, kz] = ((1 - p10_x) * lamb_in[kx - 1, kz] * gamma_l[kx - 1, kz] * gamma_r[kx - 1, kz] * gamma_t[kx - 1, kz] +
                                    p01_x * (1 - lamb_in[kx - 1, kz]) * (1 - gamma_l[kx - 1, kz]) * (1 - gamma_r[kx - 1, kz]) * (1 - gamma_t[kx - 1, kz])) / \
                                    (lamb_in[kx - 1, kz] * gamma_l[kx - 1, kz] * gamma_r[kx - 1, kz] * gamma_t[kx - 1, kz] +
                                    (1 - lamb_in[kx - 1, kz]) * (1 - gamma_l[kx - 1, kz]) * (1 - gamma_r[kx - 1, kz]) * (1 - gamma_t[kx - 1, kz]))

        # Calculate gamma_b
        for kx in range(Kx - 1):
            for kz in range(Kz):
                gamma_b[kx, kz] = ((1 - p10_x) * lamb_in[kx + 1, kz] * gamma_l[kx + 1, kz] * gamma_r[kx + 1, kz] * gamma_b[kx + 1, kz] +
                                    p10_x * (1 - lamb_in[kx + 1, kz]) * (1 - gamma_l[kx + 1, kz]) * (1 - gamma_r[kx + 1, kz]) * (1 - gamma_b[kx + 1, kz])) / \
                                    ((1 - p10_x + p01_x) * lamb_in[kx + 1, kz] * gamma_l[kx + 1, kz] * gamma_r[kx + 1, kz] * gamma_b[kx + 1, kz] +
                                    (1 - p01_x + p10_x) * (1 - lamb_in[kx + 1, kz]) * (1 - gamma_l[kx + 1, kz]) * (1 - gamma_r[kx + 1, kz]) * (1 - gamma_b[kx + 1, kz]))

    # Calculate lamb_out
    lamb_out = (gamma_l * gamma_r * gamma_t * gamma_b) / \
                (gamma_l * gamma_r * gamma_t * gamma_b + (1 - gamma_l) * (1 - gamma_r) * (1 - gamma_t) * (1 - gamma_b))

    return lamb_out


def MP_in_2D_Markov_torch(lamb_in, Kx, Kz):
    # Model parameters
    kapa = 0.5
    p01_x = 1/4
    p10_x = 1/4
    p01_z = 1/4
    p10_z = 1/4

    # Message initialization using torch
    gamma_l = 0.5 * torch.ones((Kx, Kz), dtype=torch.float32)
    gamma_r = 0.5 * torch.ones((Kx, Kz), dtype=torch.float32)
    gamma_t = 0.5 * torch.ones((Kx, Kz), dtype=torch.float32)
    gamma_b = 0.5 * torch.ones((Kx, Kz), dtype=torch.float32)
    gamma_t[0, 0] = kapa

    # MP in 2D-Markov
    for iter in range(2):
        # Calculate gamma_l
        for kz in range(1, Kz):
            for kx in range(Kx):
                gamma_l[kx, kz] = ((1 - p10_z) * lamb_in[kx, kz - 1] * gamma_l[kx, kz - 1] * gamma_t[kx, kz - 1] * gamma_b[kx, kz - 1] +
                                    p01_z * (1 - lamb_in[kx, kz - 1]) * (1 - gamma_l[kx, kz - 1]) * (1 - gamma_t[kx, kz - 1]) * (1 - gamma_b[kx, kz - 1])) / \
                                    (lamb_in[kx, kz - 1] * gamma_l[kx, kz - 1] * gamma_t[kx, kz - 1] * gamma_b[kx, kz - 1] +
                                     (1 - lamb_in[kx, kz - 1]) * (1 - gamma_l[kx, kz - 1]) * (1 - gamma_t[kx, kz - 1]) * (1 - gamma_b[kx, kz - 1]))

        # Calculate gamma_r
        for kz in range(Kz - 1):
            for kx in range(Kx):
                gamma_r[kx, kz] = ((1 - p10_z) * lamb_in[kx, kz + 1] * gamma_r[kx, kz + 1] * gamma_t[kx, kz + 1] * gamma_b[kx, kz + 1] +
                                    p10_z * (1 - lamb_in[kx, kz + 1]) * (1 - gamma_r[kx, kz + 1]) * (1 - gamma_t[kx, kz + 1]) * (1 - gamma_b[kx, kz + 1])) / \
                                    ((1 - p10_z + p01_z) * lamb_in[kx, kz + 1] * gamma_r[kx, kz + 1] * gamma_t[kx, kz + 1] * gamma_b[kx, kz + 1] +
                                     (1 - p01_z + p10_z) * (1 - lamb_in[kx, kz + 1]) * (1 - gamma_r[kx, kz + 1]) * (1 - gamma_t[kx, kz + 1]) * (1 - gamma_b[kx, kz + 1]))

        # Calculate gamma_t
        for kx in range(1, Kx):
            for kz in range(Kz):
                gamma_t[kx, kz] = ((1 - p10_x) * lamb_in[kx - 1, kz] * gamma_l[kx - 1, kz] * gamma_r[kx - 1, kz] * gamma_t[kx - 1, kz] +
                                    p01_x * (1 - lamb_in[kx - 1, kz]) * (1 - gamma_l[kx - 1, kz]) * (1 - gamma_r[kx - 1, kz]) * (1 - gamma_t[kx - 1, kz])) / \
                                    (lamb_in[kx - 1, kz] * gamma_l[kx - 1, kz] * gamma_r[kx - 1, kz] * gamma_t[kx - 1, kz] +
                                     (1 - lamb_in[kx - 1, kz]) * (1 - gamma_l[kx - 1, kz]) * (1 - gamma_r[kx - 1, kz]) * (1 - gamma_t[kx - 1, kz]))

        # Calculate gamma_b
        for kx in range(Kx - 1):
            for kz in range(Kz):
                gamma_b[kx, kz] = ((1 - p10_x) * lamb_in[kx + 1, kz] * gamma_l[kx + 1, kz] * gamma_r[kx + 1, kz] * gamma_b[kx + 1, kz] +
                                    p10_x * (1 - lamb_in[kx + 1, kz]) * (1 - gamma_l[kx + 1, kz]) * (1 - gamma_r[kx + 1, kz]) * (1 - gamma_b[kx + 1, kz])) / \
                                    ((1 - p10_x + p01_x) * lamb_in[kx + 1, kz] * gamma_l[kx + 1, kz] * gamma_r[kx + 1, kz] * gamma_b[kx + 1, kz] +
                                     (1 - p01_x + p10_x) * (1 - lamb_in[kx + 1, kz]) * (1 - gamma_l[kx + 1, kz]) * (1 - gamma_r[kx + 1, kz]) * (1 - gamma_b[kx + 1, kz]))

    # Calculate lamb_out
    lamb_out = (gamma_l * gamma_r * gamma_t * gamma_b) / \
               (gamma_l * gamma_r * gamma_t * gamma_b + (1 - gamma_l) * (1 - gamma_r) * (1 - gamma_t) * (1 - gamma_b))

    return lamb_out


def MP_in_2D_Markov_torch11(lamb_in, Kx, Kz, batch_size):
    # Model parameters
    kapa = 0.5
    p01_x = 1/4
    p10_x = 1/4
    p01_z = 1/4
    p10_z = 1/4

    # Message initialization using torch
    gamma_l = 0.5 * torch.ones((batch_size, Kx, Kz), dtype=torch.float32, device=lamb_in.device)
    gamma_r = 0.5 * torch.ones((batch_size, Kx, Kz), dtype=torch.float32, device=lamb_in.device)
    gamma_t = 0.5 * torch.ones((batch_size, Kx, Kz), dtype=torch.float32, device=lamb_in.device)
    gamma_b = 0.5 * torch.ones((batch_size, Kx, Kz), dtype=torch.float32, device=lamb_in.device)
    gamma_t[:, 0, 0] = kapa

    # MP in 2D-Markov (iterations over the batch and across Kx, Kz)
    for iter in range(2):
        # Calculate gamma_l (left message)
        for kz in range(1, Kz):
            for kx in range(Kx):
                gamma_l[:, kx, kz] = ((1 - p10_z) * lamb_in[:, kx, kz - 1] * gamma_l[:, kx, kz - 1] * gamma_t[:, kx, kz - 1] * gamma_b[:, kx, kz - 1] +
                                       p01_z * (1 - lamb_in[:, kx, kz - 1]) * (1 - gamma_l[:, kx, kz - 1]) * (1 - gamma_t[:, kx, kz - 1]) * (1 - gamma_b[:, kx, kz - 1])) / \
                                      (lamb_in[:, kx, kz - 1] * gamma_l[:, kx, kz - 1] * gamma_t[:, kx, kz - 1] * gamma_b[:, kx, kz - 1] +
                                       (1 - lamb_in[:, kx, kz - 1]) * (1 - gamma_l[:, kx, kz - 1]) * (1 - gamma_t[:, kx, kz - 1]) * (1 - gamma_b[:, kx, kz - 1]))

        # Calculate gamma_r (right message)
        for kz in range(Kz - 1):
            for kx in range(Kx):
                gamma_r[:, kx, kz] = ((1 - p10_z) * lamb_in[:, kx, kz + 1] * gamma_r[:, kx, kz + 1] * gamma_t[:, kx, kz + 1] * gamma_b[:, kx, kz + 1] +
                                       p10_z * (1 - lamb_in[:, kx, kz + 1]) * (1 - gamma_r[:, kx, kz + 1]) * (1 - gamma_t[:, kx, kz + 1]) * (1 - gamma_b[:, kx, kz + 1])) / \
                                      ((1 - p10_z + p01_z) * lamb_in[:, kx, kz + 1] * gamma_r[:, kx, kz + 1] * gamma_t[:, kx, kz + 1] * gamma_b[:, kx, kz + 1] +
                                       (1 - p01_z + p10_z) * (1 - lamb_in[:, kx, kz + 1]) * (1 - gamma_r[:, kx, kz + 1]) * (1 - gamma_t[:, kx, kz + 1]) * (1 - gamma_b[:, kx, kz + 1]))

        # Calculate gamma_t (top message)
        for kx in range(1, Kx):
            for kz in range(Kz):
                gamma_t[:, kx, kz] = ((1 - p10_x) * lamb_in[:, kx - 1, kz] * gamma_l[:, kx - 1, kz] * gamma_r[:, kx - 1, kz] * gamma_t[:, kx - 1, kz] +
                                       p01_x * (1 - lamb_in[:, kx - 1, kz]) * (1 - gamma_l[:, kx - 1, kz]) * (1 - gamma_r[:, kx - 1, kz]) * (1 - gamma_t[:, kx - 1, kz])) / \
                                      (lamb_in[:, kx - 1, kz] * gamma_l[:, kx - 1, kz] * gamma_r[:, kx - 1, kz] * gamma_t[:, kx - 1, kz] +
                                       (1 - lamb_in[:, kx - 1, kz]) * (1 - gamma_l[:, kx - 1, kz]) * (1 - gamma_r[:, kx - 1, kz]) * (1 - gamma_t[:, kx - 1, kz]))

        # Calculate gamma_b (bottom message)
        for kx in range(Kx - 1):
            for kz in range(Kz):
                gamma_b[:, kx, kz] = ((1 - p10_x) * lamb_in[:, kx + 1, kz] * gamma_l[:, kx + 1, kz] * gamma_r[:, kx + 1, kz] * gamma_b[:, kx + 1, kz] +
                                       p10_x * (1 - lamb_in[:, kx + 1, kz]) * (1 - gamma_l[:, kx + 1, kz]) * (1 - gamma_r[:, kx + 1, kz]) * (1 - gamma_b[:, kx + 1, kz])) / \
                                      ((1 - p10_x + p01_x) * lamb_in[:, kx + 1, kz] * gamma_l[:, kx + 1, kz] * gamma_r[:, kx + 1, kz] * gamma_b[:, kx + 1, kz] +
                                       (1 - p01_x + p10_x) * (1 - lamb_in[:, kx + 1, kz]) * (1 - gamma_l[:, kx + 1, kz]) * (1 - gamma_r[:, kx + 1, kz]) * (1 - gamma_b[:, kx + 1, kz]))

    # Calculate lamb_out
    lamb_out = (gamma_l * gamma_r * gamma_t * gamma_b) / \
               (gamma_l * gamma_r * gamma_t * gamma_b + (1 - gamma_l) * (1 - gamma_r) * (1 - gamma_t) * (1 - gamma_b))

    return lamb_out


def complex_to_real_4x4(F):
    """
    将复数矩阵 F 转换为实数矩阵（按照公式构造）
    参数:
    F -- 形状为 (BATCH_SIZE, M, M) 的复数矩阵

    返回:
    result -- 形状为 (BATCH_SIZE, 2 * M, 2 * M) 的实数矩阵
    """
    # 提取实部和虚部
    real_F = torch.real(F)  # 获取 F 的实部
    imag_F = torch.imag(F)  # 获取 F 的虚部

    # # 将实部和虚部转换为 torch 张量
    # real_F_torch = torch.tensor(real_F, dtype=torch.float32)
    # imag_F_torch = torch.tensor(imag_F, dtype=torch.float32)

    # 构造矩阵 \mathbf{\ddot{F}}
    batch_size, M, _ = real_F.shape
    result = torch.zeros((batch_size, 2 * M, 2 * M), dtype=torch.float32, device=F.device)
    # result = torch.zeros((batch_size, 2 * M, 2 * M), dtype=torch.float64, device=F.device)

    # 按照给定公式填充 result 矩阵
    result[:, 0:M, 0:M] = real_F  # 第一部分: \Re(F)
    result[:, 0:M, M:2 * M] = -imag_F  # 第二部分: -\Im(F)
    result[:, M:2 * M, 0:M] = imag_F  # 第三部分: \Im(F)
    result[:, M:2 * M, M:2 * M] = real_F  # 第四部分: \Re(F)

    return result


def real_to_complex_4x4(F):
    """
    将实数矩阵 F 转换为复数矩阵（反变换）。
    参数:
    F -- 形状为 (BATCH_SIZE, 2 * M, 2 * M) 的实数矩阵

    返回:
    result -- 形状为 (BATCH_SIZE, M, M) 的复数矩阵
    """
    # 获取 batch_size 和 M
    batch_size, size_2M, _ = F.shape
    M = size_2M // 2

    # 提取实部和虚部
    real_F = F[:, 0:M, 0:M]  # 第一部分: \Re(F)
    imag_F = -F[:, 0:M, M:2 * M]  # 第二部分: -\Im(F)

    # 恢复复数矩阵
    result = real_F + 1j * imag_F  # 复数矩阵 = 实部 + 虚部 * i

    return result


def complex_to_real_stack(y):
    """
    将复数向量 y 转换为实数向量（按照公式构造）
    参数:
    y -- 形状为 (BATCH_SIZE, M, 1) 的复数向量

    返回:
    result -- 形状为 (BATCH_SIZE, 2 * M, 1) 的实数向量
    """
    # 提取实部和虚部
    real_y = torch.real(y)  # 获取 y 的实部
    imag_y = torch.imag(y)  # 获取 y 的虚部

    output_real = torch.cat((real_y, imag_y), dim=1)

    return output_real


def real_to_complex_stack(y_real):
    """
    将实数向量 y_real 转换为复数向量（逆变换）
    参数:
    y_real -- 形状为 (BATCH_SIZE, 2 * M, 1) 的实数向量

    返回:
    result -- 形状为 (BATCH_SIZE, M, 1) 的复数向量
    """
    # 分离实部和虚部
    real_y = y_real[:, :y_real.shape[1] // 2, :]  # 获取前半部分作为实部
    imag_y = y_real[:, y_real.shape[1] // 2:, :]  # 获取后半部分作为虚部

    # 构造复数向量
    result = torch.complex(real_y, imag_y)

    return result


def compute_NMSE(x, y):
    """
    计算两个复数向量之间的归一化均方误差 (NMSE)
    参数:
    x -- 形状为 (BATCH_SIZE, M, 1) 的复数向量
    y -- 形状为 (BATCH_SIZE, M, 1) 的复数向量

    返回:
    nmse_value -- 归一化均方误差
    """
    # 计算均方误差 (MSE)
    mse = torch.norm(x - y, dim=1) ** 2  # Frobenius范数的平方

    # 计算 x 的能量 (L2范数的平方)
    energy_x = torch.norm(x, dim=1) ** 2

    # 计算 NMSE
    nmse_value = mse / energy_x

    return nmse_value  # 平均 NMSE 跨所有样本


def real_to_complex_np(y_double):
    """
    从实数向量 \mathbf{\ddot{y}} 恢复复数向量 \mathbf{y}，并返回 NumPy 数组

    参数:
    y_double -- 形状为 (BATCH_SIZE, 2 * M, 1) 的实数向量

    返回:
    y -- 形状为 (BATCH_SIZE, M, 1) 的复数 NumPy 数组
    """
    # 提取实部和虚部
    real_part = y_double[:, :y_double.shape[1] // 2, 0].numpy()  # 提取前 M 行作为实部
    imag_part = y_double[:, y_double.shape[1] // 2:, 0].numpy()  # 提取后 M 行作为虚部

    # 使用 NumPy 合成复数
    y_np = real_part + 1j * imag_part  # 恢复复数向量

    # 调整形状为 (BATCH_SIZE, M, 1)
    y_np = y_np[:, :, np.newaxis]

    return y_np


def genarate_sub_channel(y_complex, z_torch, F_torch, A_torch, nPath, N, Kx, Kz, Nx, Nz, Mx):
    BATCH_SIZE = z_torch.shape[0]
    z_complex = real_to_complex_stack(z_torch)
    # F_complex = real_to_complex_4x4(F_torch)
    A_complex = A_torch
    x_post = z_complex
    sorted_index = torch.argsort(torch.abs(x_post.squeeze(-1)), descending=True)
    index_amp = sorted_index[:, :4 * nPath].type(torch.int32)  # Take the first 2*nPath indices


    H_tilde_set_torch = torch.zeros((BATCH_SIZE, int(2 * N), index_amp.shape[1], Kx, Kz), dtype=torch.float32,
                                    device=z_torch.device)
    # H_tilde_set_torch11 = torch.zeros((BATCH_SIZE, int(2 * N), index_amp.shape[1], Kx, Kz), dtype=torch.float32,
    #                                 device=z_torch.device)
    y_tilde_set_torch = torch.zeros((BATCH_SIZE, int(2 * N), Kx, Kz), dtype=torch.float32, device=z_torch.device)
    # v_tilde_est_torch = torch.zeros((BATCH_SIZE, int(2 * N), Kx, Kz), dtype=torch.float32).cuda()

    # oness = torch.ones((N, 1), dtype=torch.complex64).cuda()
    oness = torch.ones((N, 1), dtype=torch.complex64, device=z_torch.device)

    for kx in range(Kx):
        for kz in range(Kz):
            Phik = np.zeros((Nx, Nz), dtype=int)
            for nz in range(Nz):
                start_index = (kz * Nz + nz) * Mx
                Phik[:, nz] = np.arange(start_index + (kx * Nx), start_index + (kx + 1) * Nx)
            Phik = np.reshape(Phik, (-1,), order='F')
            yk = y_complex[:, Phik, :]

            for ii in range(BATCH_SIZE):
                a = index_amp[ii, :].cpu()
                A_batch = A_complex[ii, :]
                aa = A_batch[np.ix_(Phik, a)]
                Hk = aa * (oness @ x_post[ii, index_amp[ii, :], 0].view(1, index_amp.shape[1]))
                H_tilde_set_torch[ii, :, :, kx, kz] = torch.cat([torch.real(Hk), torch.imag(Hk)], dim=0)

            y_tilde_set_torch[:, :, kx, kz] = torch.cat([torch.real(yk), torch.imag(yk)], dim=1).squeeze(-1)

    return y_tilde_set_torch, H_tilde_set_torch, index_amp, A_complex


def genarate_sub_channel11(y_complex, z_torch, F_torch, A_torch, nPath, N, Kx, Kz, Nx, Nz, Mx, V, sigma2):
    BATCH_SIZE = z_torch.shape[0]
    z_complex = real_to_complex_stack(z_torch)
    # F_complex = real_to_complex_4x4(F_torch)
    A_complex = A_torch
    x_post = z_complex
    sorted_index = torch.argsort(torch.abs(x_post.squeeze(-1)), descending=True)
    index_amp = sorted_index[:, :2 * nPath].type(torch.int32)  # Take the first 2*nPath indices

    H_tilde_set_torch = torch.zeros((BATCH_SIZE, int(2 * N), index_amp.shape[1], Kx, Kz), dtype=torch.float32,
                                    device=z_torch.device)
    y_tilde_set_torch = torch.zeros((BATCH_SIZE, int(2 * N), Kx, Kz), dtype=torch.float32, device=z_torch.device)
    oness = torch.ones((N, 1), dtype=torch.complex64, device=z_torch.device)

    for kx in range(Kx):
        for kz in range(Kz):
            Phik = np.zeros((Nx, Nz), dtype=int)
            for nz in range(Nz):
                start_index = (kz * Nz + nz) * Mx
                Phik[:, nz] = np.arange(start_index + (kx * Nx), start_index + (kx + 1) * Nx)
            Phik = np.reshape(Phik, (-1,), order='F')
            yk = y_complex[:, Phik, :]

            for ii in range(BATCH_SIZE):
                a = index_amp[ii, :].cpu()
                A_batch = A_complex[ii, :]
                aa = A_batch[np.ix_(Phik, a)]
                Hk = aa * (oness @ x_post[ii, index_amp[ii, :], 0].view(1, index_amp.shape[1]))
                H_tilde_set_torch[ii, :, :, kx, kz] = torch.cat([torch.real(Hk), torch.imag(Hk)], dim=0)

            y_tilde_set_torch[:, :, kx, kz] = torch.cat([torch.real(yk), torch.imag(yk)], dim=1).squeeze(-1)

    index_amp_batch = index_amp.to(torch.long)
    V_selected = V[torch.arange(BATCH_SIZE).unsqueeze(-1), :, :, index_amp_batch]
    mean_A_pri_set = V_selected.clone().permute(0, 2, 3, 1)
    var_A_pri_set = mean_A_pri_set * (1 - mean_A_pri_set)

    mean_B_pri_set = mean_A_pri_set.clone()
    var_B_pri_set = var_A_pri_set.clone()

    # 计算 V_tilde_update_A 和 V_tilde_update_B
    V_tilde_update_A = mean_A_pri_set.clone()  # 直接使用 clone 来避免多次创建

    # Module A: LMMSE estimator
    for kx in range(Kx):
        for kz in range(Kz):
            H_sub = H_tilde_set_torch[:, :, :, kx, kz]
            y_sub = y_tilde_set_torch[:, :, kx, kz]

            # prior information
            mean_A_pri = mean_A_pri_set[:, kx, kz, :]
            var_A_pri = var_A_pri_set[:, kx, kz, :]
            # Posterior information: LMMSE
            H_sub_t = H_sub.transpose(1, 2)  # 计算 H 的转置
            var_A_post = torch.linalg.inv(
                    1 / sigma2.unsqueeze(1) * torch.bmm(H_sub_t, H_sub) + torch.diag_embed(
                        1 / var_A_pri))

            mean_A_post = (torch.bmm(var_A_post, (
                        1 / sigma2.unsqueeze(1) * torch.bmm(H_sub_t, y_sub.unsqueeze(2)) + (
                        mean_A_pri.clone() / var_A_pri.clone()).unsqueeze(2)))).squeeze(2)
            var_A_post1 = var_A_post.diagonal(dim1=-2, dim2=-1)
            # Extrinsic information
            var_A_ext = 1 / (1 / var_A_post1 - 1 / var_A_pri)
            mean_A_ext = var_A_ext * (mean_A_post / var_A_post1 - mean_A_pri / var_A_pri)
            # Passed to Module B
            mean_B_pri_set[:, kx, kz, :] = mean_A_ext
            var_B_pri_set[:, kx, kz, :] = var_A_ext
            V_tilde_update_A[:, kx, kz, :] = mean_A_post

    return index_amp, mean_B_pri_set, var_B_pri_set, V_tilde_update_A


def generate_sub_channel_for_kx_kz(kx, kz, y_complex, A_complex, index_amp, N, x_post, oness, BATCH_SIZE, Phik, Kx, Kz, H_tilde_set_torch, y_tilde_set_torch):
    """
    这个函数处理每个 kx, kz 的计算，单独分配到一个线程。
    """
    for ii in range(BATCH_SIZE):
        a = index_amp[ii, :].cpu()
        A_batch = A_complex[ii, :]
        aa = A_batch[np.ix_(Phik, a)]
        Hk = aa * (oness @ x_post[ii, index_amp[ii, :], 0].view(1, N))
        H_tilde_set_torch[ii, :, :, kx, kz] = torch.cat([torch.real(Hk), torch.imag(Hk)], dim=0)

    yk = y_complex[:, Phik, :]
    y_tilde_set_torch[:, :, kx, kz] = torch.cat([torch.real(yk), torch.imag(yk)], dim=1).squeeze(-1)


def generate_sub_channel_parallel(y_complex, z_torch, F_torch, A_torch, nPath, N, Kx, Kz, Nx, Nz, Mx):
    """
    多线程计算生成子信道的主函数。
    """
    BATCH_SIZE = z_torch.shape[0]
    z_complex = real_to_complex_stack(z_torch)
    A_complex = A_torch
    x_post = z_complex
    sorted_index = torch.argsort(torch.abs(x_post.squeeze(-1)), descending=True)
    index_amp = sorted_index[:, :2 * nPath].type(torch.int32)  # Take the first 2*nPath indices

    H_tilde_set_torch = torch.zeros((BATCH_SIZE, int(2 * N), index_amp.shape[1], Kx, Kz), dtype=torch.float32).cuda()
    y_tilde_set_torch = torch.zeros((BATCH_SIZE, int(2 * N), Kx, Kz), dtype=torch.float32).cuda()

    oness = torch.ones((N, 1), dtype=torch.complex64).cuda()

    # 在这里初始化线程池，最多使用 8 个线程，你可以根据需要调整
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        # 使用 kx 和 kz 的组合来分配任务
        for kx in range(Kx):
            for kz in range(Kz):
                Phik = np.zeros((Nx, Nz), dtype=int)
                for nz in range(Nz):
                    start_index = (kz * Nz + nz) * Mx
                    Phik[:, nz] = np.arange(start_index + (kx * Nx), start_index + (kx + 1) * Nx)
                Phik = np.reshape(Phik, (-1,), order='F')

                # 提交任务到线程池
                futures.append(executor.submit(generate_sub_channel_for_kx_kz, kx, kz, y_complex, A_complex, index_amp, N, x_post, oness, BATCH_SIZE, Phik, Kx, Kz, H_tilde_set_torch, y_tilde_set_torch))

        # 等待所有线程完成
        for future in futures:
            future.result()

    return y_tilde_set_torch, H_tilde_set_torch, index_amp, A_complex


def EP111(H_tilde_set, y_tilde_set, index_amp, V, Kx, Kz, sigma2, BATCH_SIZE, U_torch, Nx, Nz, M):
    index_amp_batch = index_amp.to(torch.long)
    V_selected = V[torch.arange(BATCH_SIZE).unsqueeze(-1), :, :, index_amp_batch]
    mean_A_pri_set = V_selected.clone().permute(0, 2, 3, 1)
    var_A_pri_set = mean_A_pri_set * (1 - mean_A_pri_set)

    mean_B_pri_set = mean_A_pri_set.clone()
    var_B_pri_set = var_A_pri_set.clone()

    # 计算 V_tilde_update_A 和 V_tilde_update_B
    V_tilde_update_A = mean_A_pri_set.clone()  # 直接使用 clone 来避免多次创建
    # V_tilde_update_B = mean_A_pri_set.clone()  # 假设这里是直接赋值，你可以根据需求调整

    # Module A: LMMSE estimator
    for kx in range(Kx):
        for kz in range(Kz):
            H_sub = H_tilde_set[:, :, :, kx, kz]
            y_sub = y_tilde_set[:, :, kx, kz]

            # prior information
            mean_A_pri = mean_A_pri_set[:, kx, kz, :]
            var_A_pri = var_A_pri_set[:, kx, kz, :]
            # Posterior information: LMMSE
            H_sub_t = H_sub.transpose(1, 2)  # 计算 H 的转置
            var_A_post = torch.linalg.inv(
                        1 / sigma2.unsqueeze(1) * torch.bmm(H_sub_t, H_sub) + torch.diag_embed(
                            1 / var_A_pri))

            b = torch.bmm(H_sub_t, y_sub.unsqueeze(2))
            aaa = (mean_A_pri.clone() / var_A_pri.clone())
            c = aaa.unsqueeze(2)
            aa = (1 / sigma2.unsqueeze(1) * b + c)
            a = (torch.bmm(var_A_post, aa))
            mean_A_post = a.squeeze(2)

            # mean_A_post = (torch.bmm(var_A_post, (
            #             1 / sigma2.unsqueeze(1) * torch.bmm(H_sub.transpose(1, 2), y_sub.unsqueeze(2)) + (
            #                 mean_A_pri / var_A_pri).unsqueeze(2)))).squeeze(2)
            var_A_post1 = var_A_post.diagonal(dim1=-2, dim2=-1)

            # Passed to Module B
            mean_B_pri_set[:, kx, kz, :] = mean_A_post
            var_B_pri_set[:, kx, kz, :] = var_A_post1
            V_tilde_update_A[:, kx, kz, :] = mean_A_post

        # Update V
    V_update = V
    for batch_idx in range(BATCH_SIZE):
        V_update[batch_idx, :, :, index_amp[batch_idx, :]] = V_tilde_update_A[batch_idx, :, :, :]
    V_update = torch.clamp(V_update, min=0, max=1)
    VR_est = torch.floor(V_update + 0.5)

    # Update sensing matrix
    U_update = U_torch
    ones_tensor = torch.ones((BATCH_SIZE, Nx, Nz), device=V_update.device, dtype=V_update.dtype)
    for batch_idx in range(BATCH_SIZE):
        for q in range((index_amp.shape[1])):
            grid_index = index_amp[batch_idx, q]
            kron_result = torch.matmul(V_update[batch_idx, :, :, grid_index].view(-1, 1), ones_tensor[batch_idx, :, :].view(1, -1))
            U_update[batch_idx, :, grid_index] = kron_result.t().reshape(-1, )  # 变形并按列优先顺序排列

    return VR_est, U_update


def Grad111(y, F_complex, x, index_amp, grid, Mx, Mz, A_torch, lambda_, d, M, nGrid, sigma2):
        BATCH_SIZE = y.shape[0]
        z_complex = x
        y_complex = real_to_complex_stack(y.cuda())
        # BATCH_SIZE = z_complex.shape[0]
        # nGrid = grid.shape[1]
        A_complex = A_torch
        # A_complex = real_to_complex_4x4(A_torch)
        # F_complex = real_to_complex_4x4(F)

        # Gradient ascent method w.r.t theta
        grid_update = grid.clone()
        grad_F_to_g1 = torch.zeros((BATCH_SIZE, M, nGrid), dtype=torch.complex64, device=x.device)
        grad_F_to_g2 = torch.zeros((BATCH_SIZE, M, nGrid), dtype=torch.complex64, device=x.device)

        # 创建 delta_x 和 delta_z 张量，使用torch.arange和reshape
        delta_x = torch.arange(-(Mx - 1) / 2, (Mx - 1) / 2 + 1, dtype=torch.float64, device=x.device).view(-1, 1)  # .view替代reshape
        delta_z = torch.arange(-(Mz - 1) / 2, (Mz - 1) / 2 + 1, dtype=torch.float64, device=x.device).view(-1, 1)

        ones_Mz = torch.ones(Mz, 1, dtype=torch.float64, device=z_complex.device)
        ones_Mx = torch.ones(Mx, 1, dtype=torch.float64, device=z_complex.device)
        pi_lambda = 1j * 2 * torch.pi / lambda_

        error_term = (y_complex - torch.matmul(F_complex, z_complex))

        # 批量计算误差项
        delta_x_d = (delta_x * d)
        delta_z_d = (delta_z * d)
        delta_zT_d = delta_z.T * d
        delta_x2_d2 = (delta_x ** 2 * d ** 2)
        delta_z2_d2 = (delta_z ** 2 * d ** 2)

        for q in range(index_amp.shape[1]):
            for batch_idx in range(BATCH_SIZE):
                grid_index = index_amp[batch_idx, q]
                g1_q = grid_update[batch_idx, grid_index, 0]
                g2_q = grid_update[batch_idx, grid_index, 1]
                r_q = grid_update[batch_idx, grid_index, 2]

                # a = F_complex[batch_idx, :, grid_index].view(-1, 1)
                F_grid = F_complex[batch_idx, :, grid_index].view(-1, 1)
                # 变为列向量
                term1 = torch.kron(ones_Mz, pi_lambda * (delta_x_d + delta_x2_d2 * g1_q / r_q))
                term2 = ((pi_lambda / r_q) * (delta_x_d @ (delta_zT_d * g2_q)) ).t().reshape(-1, 1)

                grad_F_to_g1[batch_idx, :, grid_index] = (F_grid * (term1 + term2)).flatten()

                a0 = (delta_z_d + delta_z2_d2 * g2_q / r_q)
                a00 = (pi_lambda * a0)

                term11 = torch.kron(a00, ones_Mx)

                term22 = (pi_lambda / r_q) * ((delta_x_d * g1_q) @ delta_zT_d)

                # 计算最终的 grad_F_to_g2
                grad_F_to_g2[batch_idx, :, grid_index] = (F_grid * (term11 + term22.t().reshape(-1, 1))).flatten()

                term1_g1 = torch.matmul(grad_F_to_g1[batch_idx, :, grid_index].conj().T, (error_term[batch_idx, :]))
                grad_Q_to_g1 = -2 / sigma2[batch_idx, 0] * torch.real(-torch.conj(z_complex[batch_idx, grid_index, :].T)
                                                                      * term1_g1)

                # 计算 grad_Q_to_g2
                term1_g2 = torch.matmul(grad_F_to_g2[batch_idx, :, grid_index].conj().T, (error_term[batch_idx, :]))
                grad_Q_to_g2 = -2 / sigma2[batch_idx, 0] * torch.real(-torch.conj(z_complex[batch_idx, grid_index, :].T)
                                                                      * term1_g2)

                grid_update[batch_idx, grid_index, 0] = g1_q + 2 / Mx / 40 * torch.sign(grad_Q_to_g1)
                grid_update[batch_idx, grid_index, 1] = g2_q + 2 / Mz / 40 * torch.sign(grad_Q_to_g2)

        A_complex_update = A_complex
        for q in range(index_amp.shape[1]):
            for batch_idx in range(BATCH_SIZE):
                grid_index = index_amp[batch_idx, q]
                A_complex_update[batch_idx, :, grid_index] = array_response_g12_torch(grid_update[batch_idx, grid_index, 2], grid_update[batch_idx, grid_index, 0],
                                                                grid_update[batch_idx, grid_index, 1], Mx, Mz, lambda_, d)

        return A_complex_update, grid_update


def function_grad(y_torch, F_torch1, z_torch1, polar_grid_torch, Mx, Mz, A_torch, lambda_, d, M, nGrid, sigma2, nPath):
    BATCH_SIZE = y_torch.shape[0]
    z_complex = real_to_complex_stack(z_torch1)
    F_complex = real_to_complex_4x4(F_torch1)
    sorted_index = torch.argsort(torch.abs(z_complex.squeeze(-1)), descending=True)
    index_amp = sorted_index[:, :2 * nPath].type(torch.int64)
    A_complex_update, grid_update = Grad111(y_torch, F_complex, z_complex, index_amp, polar_grid_torch, Mx, Mz,
                                            A_torch, lambda_, d, M, nGrid, sigma2)

    F_update_complex10 = A_complex_update
    F_update_grad0000 = complex_to_real_4x4(F_update_complex10)

    return F_update_grad0000


def function_grad_preparation(y_torch, F_torch1, z_torch1, polar_grid_torch, Mx, Mz, A_torch, lambda_, d, M, nGrid, sigma2, nPath):
    BATCH_SIZE = y_torch.shape[0]
    z_complex = real_to_complex_stack(z_torch1)
    F_complex = real_to_complex_4x4(F_torch1)
    sorted_index = torch.argsort(torch.abs(z_complex.squeeze(-1)), descending=True)
    index_amp = sorted_index[:, :2 * nPath].type(torch.int64)

    y_complex = real_to_complex_stack(y_torch.cuda())
    grid_update = polar_grid_torch.clone()

    grad_F_to_g1 = torch.zeros((BATCH_SIZE, M, index_amp.shape[1]), dtype=torch.complex64, device=z_complex.device)
    grad_F_to_g2 = torch.zeros((BATCH_SIZE, M, index_amp.shape[1]), dtype=torch.complex64, device=z_complex.device)
    grad_Q_to_g1 = torch.zeros((BATCH_SIZE, index_amp.shape[1]), dtype=torch.float32, device=z_complex.device)
    grad_Q_to_g2 = torch.zeros((BATCH_SIZE, index_amp.shape[1]), dtype=torch.float32, device=z_complex.device)

    # 创建 delta_x 和 delta_z 张量，使用torch.arange和reshape
    delta_x = torch.arange(-(Mx - 1) / 2, (Mx - 1) / 2 + 1, dtype=torch.float64, device=z_complex.device).view(-1, 1)
    delta_z = torch.arange(-(Mz - 1) / 2, (Mz - 1) / 2 + 1, dtype=torch.float64, device=z_complex.device).view(-1, 1)

    ones_Mz = torch.ones(Mz, 1, dtype=torch.float64, device=z_complex.device)
    ones_Mx = torch.ones(Mx, 1, dtype=torch.float64, device=z_complex.device)
    pi_lambda = 1j * 2 * torch.pi / lambda_

    error_term = (y_complex - torch.matmul(F_complex, z_complex))

    # 批量计算误差项
    delta_x_d = (delta_x * d)
    delta_z_d = (delta_z * d)
    delta_zT_d = delta_z.T * d
    delta_x2_d2 = (delta_x ** 2 * d ** 2)
    delta_z2_d2 = (delta_z ** 2 * d ** 2)

    for q in range(index_amp.shape[1]):
        for batch_idx in range(BATCH_SIZE):
            grid_index = index_amp[batch_idx, q]
            g1_q = grid_update[batch_idx, grid_index, 0]
            g2_q = grid_update[batch_idx, grid_index, 1]
            r_q = grid_update[batch_idx, grid_index, 2]

            # a = F_complex[batch_idx, :, grid_index].view(-1, 1)
            F_grid = F_complex[batch_idx, :, grid_index].view(-1, 1)
            # 变为列向量
            term1 = torch.kron(ones_Mz, pi_lambda * (delta_x_d + delta_x2_d2 * g1_q.clone() / r_q.clone()))

            term2 = ((pi_lambda / r_q.clone()) * (delta_x_d @ (delta_zT_d * g2_q.clone()))).t().reshape(-1, 1)

            grad_F_to_g1[batch_idx, :, q] = (F_grid * (term1 + term2)).flatten()

            a0 = (delta_z_d + delta_z2_d2 * g2_q.clone() / r_q.clone())
            a00 = (pi_lambda * a0)

            term11 = torch.kron(a00, ones_Mx)

            term22 = (pi_lambda / r_q) * ((delta_x_d * g1_q) @ delta_zT_d)

            # 计算最终的 grad_F_to_g2
            grad_F_to_g2[batch_idx, :, q] = (F_grid * (term11 + term22.t().reshape(-1, 1))).flatten()

            term1_g1 = torch.matmul((grad_F_to_g1[batch_idx, :, q].conj().T).unsqueeze(0), (error_term[batch_idx, :]))

            grad_Q_to_g1[batch_idx, q] = (-2 / sigma2[batch_idx, 0]
                                          * torch.real(-torch.conj(z_complex[batch_idx, grid_index, :].T) * term1_g1))

            # 计算 grad_Q_to_g2
            term1_g2 = torch.matmul(grad_F_to_g2[batch_idx, :, q].conj().T, (error_term[batch_idx, :]))
            grad_Q_to_g2[batch_idx, q] = (-2 / sigma2[batch_idx, 0]
                                          * torch.real(-torch.conj(z_complex[batch_idx, grid_index, :].T) * term1_g2))

            # grid_update[batch_idx, grid_index, 0] = (g1_q + 2 / Mx / 40 * torch.sign(grad_Q_to_g1[batch_idx, q]))
            # grid_update[batch_idx, grid_index, 1] = (g2_q + 2 / Mz / 40 * torch.sign(grad_Q_to_g2[batch_idx, q]))

    return grad_Q_to_g1, grad_Q_to_g2, index_amp


def function_grad_preparation11(y_torch, F_torch1, z_torch1, polar_grid_torch, Mx, Mz, A_torch, lambda_, d, M, nGrid, sigma2, nPath):
    BATCH_SIZE = y_torch.shape[0]
    z_complex = real_to_complex_stack(z_torch1)
    F_complex = real_to_complex_4x4(F_torch1)
    sorted_index = torch.argsort(torch.abs(z_complex.squeeze(-1)), descending=True)
    index_amp = sorted_index[:, :2 * nPath].type(torch.int64)

    y_complex = real_to_complex_stack(y_torch.cuda())
    grid_update = polar_grid_torch.clone()

    F_grid_batch00 = torch.zeros((BATCH_SIZE, M, index_amp.shape[1]), dtype=torch.complex64, device=grid_update.device)
    for q in range(index_amp.shape[1]):
        for batch_idx in range(BATCH_SIZE):
            grid_index = index_amp[batch_idx, q]
            F_grid_batch00[batch_idx, :, q] = F_complex[batch_idx, :, grid_index].view(-1, )

    ones_Mz_batch = torch.ones(BATCH_SIZE, Mz, 1, dtype=torch.float64, device=z_complex.device)
    ones_Mx_batch = torch.ones(BATCH_SIZE, Mx, 1, dtype=torch.float64, device=z_complex.device)
    delta_x_batch = (
        torch.arange(-(Mx - 1) / 2, (Mx - 1) / 2 + 1, dtype=torch.float64, device=z_complex.device)).unsqueeze(
        1).expand(BATCH_SIZE, -1, -1)  # X轴位置
    delta_z_batch = (
        torch.arange(-(Mz - 1) / 2, (Mz - 1) / 2 + 1, dtype=torch.float64, device=z_complex.device)).unsqueeze(
        1).expand(BATCH_SIZE, -1, -1)  # Z轴位置
    pi_lambda = 1j * 2 * torch.pi / lambda_
    error_term1 = (y_complex - torch.matmul(F_complex, z_complex))

    # 批量计算误差项
    delta_x_d_batch = (delta_x_batch * d)
    delta_z_d_batch = (delta_z_batch * d)
    delta_zT_d_batch = delta_z_batch.transpose(1, 2) * d
    delta_x2_d2_batch = (delta_x_batch ** 2 * d ** 2)
    delta_z2_d2_batch = (delta_z_batch ** 2 * d ** 2)

    grad_F_to_g100 = torch.zeros((BATCH_SIZE, M, index_amp.shape[1]), dtype=torch.complex64, device=z_complex.device)
    grad_F_to_g200 = torch.zeros((BATCH_SIZE, M, index_amp.shape[1]), dtype=torch.complex64, device=z_complex.device)
    grad_Q_to_g100 = torch.zeros((BATCH_SIZE, index_amp.shape[1]), dtype=torch.float32, device=z_complex.device)
    grad_Q_to_g200 = torch.zeros((BATCH_SIZE, index_amp.shape[1]), dtype=torch.float32, device=z_complex.device)

    for q in range(index_amp.shape[1]):
        indices = index_amp[:, q]  # 这是一个形状为 (BATCH_SIZE,) 的张量，表示每个批次对应的索引

        # 使用批量索引提取相应的 g1, g2 和 r 值
        g1_batch = grid_update[torch.arange(BATCH_SIZE), indices, 0].unsqueeze(1).unsqueeze(2)
        g2_batch = grid_update[torch.arange(BATCH_SIZE), indices, 1].unsqueeze(1).unsqueeze(2)
        r_batch = grid_update[torch.arange(BATCH_SIZE), indices, 2].unsqueeze(1).unsqueeze(2)

        F_grid_batch_item = F_grid_batch00[:, :, q].unsqueeze(2)

        aa00 = pi_lambda * (delta_x_d_batch + delta_x2_d2_batch * g1_batch / r_batch)
        result = []
        for i in range(BATCH_SIZE):
            kronecker_product = torch.kron(ones_Mz_batch[i].squeeze(), aa00[i].squeeze())  # (4, 1) 和 (64, 1) 的 Kronecker 积
            result.append(kronecker_product)
        term1_batch = torch.stack(result, dim=0).unsqueeze(2)

        term2_batch = ((pi_lambda / r_batch) * (delta_x_d_batch @ (delta_zT_d_batch * g2_batch))).transpose(1, 2).reshape(BATCH_SIZE, -1, 1)
        grad_F_to_g100[:, :, q] = (F_grid_batch_item * (term1_batch + term2_batch)).squeeze(2)

        bb0 = pi_lambda * (delta_z_d_batch + delta_z2_d2_batch * g2_batch / r_batch)
        result1 = []
        for i in range(BATCH_SIZE):
            kronecker_product1 = torch.kron(bb0[i].squeeze(), ones_Mx_batch[i].squeeze())  # (4, 1) 和 (64, 1) 的 Kronecker 积
            result1.append(kronecker_product1)
        term11_batch = torch.stack(result1, dim=0).unsqueeze(2)
        term22_batch = (pi_lambda / r_batch) * ((delta_x_d_batch * g1_batch) @ delta_zT_d_batch).transpose(1, 2).reshape(BATCH_SIZE, -1, 1)

        grad_F_to_g200[:, :, q] = (F_grid_batch_item * (term11_batch + term22_batch)).squeeze(2)

        term1_g1_batch = torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.complex64, device=z_complex.device)
        for batch_idx in range(BATCH_SIZE):
            term1_g1_batch[batch_idx, 0, 0] = grad_F_to_g100[batch_idx, :, q].conj().T @ error_term1[batch_idx, :, :]

        # indices = index_amp[:, q]  # 这是形状为 (BATCH_SIZE,) 的张量，表示每个批次对应的索引
        z_selected = z_complex[torch.arange(BATCH_SIZE), indices, :]  # 这是形状为 (BATCH_SIZE, N) 的张量

        grad_Q_to_g100[:, q] = (-2 / sigma2[:, 0] *
                                torch.real(-torch.conj(z_selected).T * term1_g1_batch.squeeze(1).squeeze(1)))

        # term1_g2_batch = grad_F_to_g200[:, :, q].unsqueeze(2).conj().transpose(1, 2) @ error_term
        term1_g2_batch = torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.complex64, device=z_complex.device)
        for batch_idx in range(BATCH_SIZE):
            term1_g2_batch[batch_idx, 0, 0] = grad_F_to_g200[batch_idx, :, q].conj().T @ error_term1[batch_idx, :]

        grad_Q_to_g200[:, q] = (-2 / sigma2[:, 0] *
                                torch.real(-torch.conj(z_selected).T * term1_g2_batch.squeeze(1).squeeze(1)))

        # print(np.max((grad_Q_to_g2001 - grad_Q_to_g200).detach().cpu().numpy()))

    return grad_Q_to_g100, grad_Q_to_g200, index_amp


def function_grad_preparation10(y_torch, F_torch1, z_torch1, polar_grid_torch, Mx, Mz, A_torch, lambda_, d, M, nGrid, sigma2, nPath):
    BATCH_SIZE = y_torch.shape[0]
    z_complex = real_to_complex_stack(z_torch1)
    F_complex = real_to_complex_4x4(F_torch1)
    sorted_index = torch.argsort(torch.abs(z_complex.squeeze(-1)), descending=True)
    index_amp = sorted_index[:, :4 * nPath].type(torch.int64)

    y_complex = real_to_complex_stack(y_torch.cuda())
    grid_update = polar_grid_torch.clone()

    F_grid_batch00 = torch.zeros((BATCH_SIZE, M, index_amp.shape[1]), dtype=torch.complex64, device=grid_update.device)
    for q in range(index_amp.shape[1]):
        for batch_idx in range(BATCH_SIZE):
            grid_index = index_amp[batch_idx, q]
            F_grid_batch00[batch_idx, :, q] = F_complex[batch_idx, :, grid_index].view(-1, )

    ones_Mz_batch = torch.ones(BATCH_SIZE, Mz, 1, dtype=torch.float64, device=z_complex.device)
    ones_Mx_batch = torch.ones(BATCH_SIZE, Mx, 1, dtype=torch.float64, device=z_complex.device)
    delta_x_batch = (
        torch.arange(-(Mx - 1) / 2, (Mx - 1) / 2 + 1, dtype=torch.float64, device=z_complex.device)).unsqueeze(
        1).expand(BATCH_SIZE, -1, -1)  # X轴位置
    delta_z_batch = (
        torch.arange(-(Mz - 1) / 2, (Mz - 1) / 2 + 1, dtype=torch.float64, device=z_complex.device)).unsqueeze(
        1).expand(BATCH_SIZE, -1, -1)  # Z轴位置
    pi_lambda = 1j * 2 * torch.pi / lambda_
    error_term1 = (y_complex - torch.matmul(F_complex, z_complex))

    # 批量计算误差项
    delta_x_d_batch = (delta_x_batch * d)
    delta_z_d_batch = (delta_z_batch * d)
    delta_zT_d_batch = delta_z_batch.transpose(1, 2) * d
    delta_x2_d2_batch = (delta_x_batch ** 2 * d ** 2)
    delta_z2_d2_batch = (delta_z_batch ** 2 * d ** 2)

    grad_F_to_g100 = torch.zeros((BATCH_SIZE, M, index_amp.shape[1]), dtype=torch.complex64, device=z_complex.device)
    grad_F_to_g200 = torch.zeros((BATCH_SIZE, M, index_amp.shape[1]), dtype=torch.complex64, device=z_complex.device)
    grad_Q_to_g100 = torch.zeros((BATCH_SIZE, index_amp.shape[1]), dtype=torch.float32, device=z_complex.device)
    grad_Q_to_g200 = torch.zeros((BATCH_SIZE, index_amp.shape[1]), dtype=torch.float32, device=z_complex.device)
    grad_Q_to_g1001 = torch.zeros((BATCH_SIZE, index_amp.shape[1]), dtype=torch.float32, device=z_complex.device)
    grad_Q_to_g2001 = torch.zeros((BATCH_SIZE, index_amp.shape[1]), dtype=torch.float32, device=z_complex.device)

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

        # print(np.max((g1_batch1 - g1_batch).detach().cpu().numpy()))
        # print(np.max((g2_batch1 - g2_batch).detach().cpu().numpy()))
        # print(np.max((r_batch1 - r_batch).detach().cpu().numpy()))

        F_grid_batch_item = F_grid_batch00[:, :, q].unsqueeze(2)

        aa00 = pi_lambda * (delta_x_d_batch + delta_x2_d2_batch * g1_batch / r_batch)
        result = []
        for i in range(BATCH_SIZE):
            kronecker_product = torch.kron(ones_Mz_batch[i].squeeze(), aa00[i].squeeze())  # (4, 1) 和 (64, 1) 的 Kronecker 积
            result.append(kronecker_product)
        term1_batch = torch.stack(result, dim=0).unsqueeze(2)

        term2_batch = ((pi_lambda / r_batch) * (delta_x_d_batch @ (delta_zT_d_batch * g2_batch))).transpose(1, 2).reshape(BATCH_SIZE, -1, 1)
        grad_F_to_g100[:, :, q] = (F_grid_batch_item * (term1_batch + term2_batch)).squeeze(2)

        bb0 = pi_lambda * (delta_z_d_batch + delta_z2_d2_batch * g2_batch / r_batch)
        result1 = []
        for i in range(BATCH_SIZE):
            kronecker_product1 = torch.kron(bb0[i].squeeze(), ones_Mx_batch[i].squeeze())  # (4, 1) 和 (64, 1) 的 Kronecker 积
            result1.append(kronecker_product1)
        term11_batch = torch.stack(result1, dim=0).unsqueeze(2)
        term22_batch = (pi_lambda / r_batch) * ((delta_x_d_batch * g1_batch) @ delta_zT_d_batch).transpose(1, 2).reshape(BATCH_SIZE, -1, 1)

        grad_F_to_g200[:, :, q] = (F_grid_batch_item * (term11_batch + term22_batch)).squeeze(2)

        # term1_g1_batch = torch.matmul(grad_F_to_g100[:, :, q].unsqueeze(2).conj().transpose(1, 2), error_term1)
        term1_g1_batch = torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.complex64, device=z_complex.device)
        for batch_idx in range(BATCH_SIZE):
            term1_g1_batch[batch_idx, 0, 0] = grad_F_to_g100[batch_idx, :, q].conj().T @ error_term1[batch_idx, :, :]

        # for batch_idx in range(BATCH_SIZE):
        #     grid_index = index_amp[batch_idx, q]
        #     grad_Q_to_g100[batch_idx, q] = (-2 / sigma2[batch_idx, 0]
        #                               * torch.real(-torch.conj(z_complex[batch_idx, grid_index, :].T) * term1_g1_batch[batch_idx, 0, 0]))

        # indices = index_amp[:, q]  # 这是形状为 (BATCH_SIZE,) 的张量，表示每个批次对应的索引
        z_selected = z_complex[torch.arange(BATCH_SIZE), indices, :]  # 这是形状为 (BATCH_SIZE, N) 的张量

        # 计算共轭转置的部分 (-torch.conj(z_selected).T)，并将其与 term1_g1_batch 进行逐批矩阵乘法
        grad_Q_to_g100[:, q] = (-2 / sigma2[:, 0] *
                                torch.real(-torch.conj(z_selected).T * term1_g1_batch.squeeze(1).squeeze(1)))

        # term1_g2_batch = grad_F_to_g200[:, :, q].unsqueeze(2).conj().transpose(1, 2) @ error_term
        term1_g2_batch = torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.complex64, device=z_complex.device)
        for batch_idx in range(BATCH_SIZE):
            term1_g2_batch[batch_idx, 0, 0] = grad_F_to_g200[batch_idx, :, q].conj().T @ error_term1[batch_idx, :]

        # for batch_idx in range(BATCH_SIZE):
        #     grid_index = index_amp[batch_idx, q]
        #     grad_Q_to_g200[batch_idx, q] = (-2 / sigma2[batch_idx, 0]
        #                               * torch.real(-torch.conj(z_complex[batch_idx, grid_index, :].T) * term1_g2_batch[batch_idx, 0, 0]))

        grad_Q_to_g200[:, q] = (-2 / sigma2[:, 0] *
                                torch.real(-torch.conj(z_selected).T * term1_g2_batch.squeeze(1).squeeze(1)))

        # print(np.max((grad_Q_to_g2001 - grad_Q_to_g200).detach().cpu().numpy()))

    return grad_Q_to_g100, grad_Q_to_g200, index_amp


def function_compute_EP(H_tilde_set, y_tilde_set, index_amp, V, Kx, Kz, sigma2, BATCH_SIZE, U_torch, Nx, Nz, M):
    index_amp_batch = index_amp.to(torch.long)
    V_selected = V[torch.arange(BATCH_SIZE).unsqueeze(-1), :, :, index_amp_batch]
    mean_A_pri_set = V_selected.clone().permute(0, 2, 3, 1)
    var_A_pri_set = mean_A_pri_set * (1 - mean_A_pri_set)

    mean_B_pri_set = mean_A_pri_set.clone()
    var_B_pri_set = var_A_pri_set.clone()

    # 计算 V_tilde_update_A 和 V_tilde_update_B
    V_tilde_update_A = mean_A_pri_set.clone()  # 直接使用 clone 来避免多次创建
    V_tilde_update_B = mean_A_pri_set.clone()  # 假设这里是直接赋值，你可以根据需求调整

    clamp_min = 1e-3
    clamp_max = 1 - 1e-3

    # Module A: LMMSE estimator
    for kx in range(Kx):
        for kz in range(Kz):
            H_sub = H_tilde_set[:, :, :, kx, kz]
            y_sub = y_tilde_set[:, :, kx, kz]

            # prior information
            mean_A_pri = mean_A_pri_set[:, kx, kz, :]
            var_A_pri = var_A_pri_set[:, kx, kz, :]
            # Posterior information: LMMSE
            H_sub_t = H_sub.transpose(1, 2)  # 计算 H 的转置
            var_A_post = torch.linalg.inv(
                1 / sigma2.unsqueeze(1) * torch.bmm(H_sub_t, H_sub) + torch.diag_embed(
                    1 / var_A_pri))

            b = torch.bmm(H_sub_t, y_sub.unsqueeze(2))
            aaa = (mean_A_pri.clone() / var_A_pri.clone())
            c = aaa.unsqueeze(2)
            aa = (1 / sigma2.unsqueeze(1) * b + c)
            a = (torch.bmm(var_A_post, aa))
            mean_A_post = a.squeeze(2)

            # mean_A_post = (torch.bmm(var_A_post, (
            #             1 / sigma2.unsqueeze(1) * torch.bmm(H_sub.transpose(1, 2), y_sub.unsqueeze(2)) + (
            #                 mean_A_pri / var_A_pri).unsqueeze(2)))).squeeze(2)
            var_A_post1 = var_A_post.diagonal(dim1=-2, dim2=-1)

            # Passed to Module B
            mean_B_pri_set[:, kx, kz, :] = mean_A_post
            var_B_pri_set[:, kx, kz, :] = var_A_post1
            V_tilde_update_A[:, kx, kz, :] = mean_A_post

    # Module B: Massage passing
    for n in range(index_amp.shape[1]):
        # prior information
        mean_B_pri = mean_B_pri_set[:, :, :, n]
        var_B_pri = var_B_pri_set[:, :, :, n]

        # Message passing
        # std_B_pri = torch.sqrt(var_B_pri)
        # 创建正态分布对象
        normal_dist = Normal(mean_B_pri, var_B_pri)
        # 计算 PDF 值
        pdf_1 = normal_dist.log_prob(torch.tensor(1.0)).exp()
        pdf_0 = normal_dist.log_prob(torch.tensor(0.0)).exp()
        lamb_in = pdf_1 / (pdf_1 + pdf_0)

        lamb_out = MP_in_2D_Markov_torch11(lamb_in, Kx, Kz, BATCH_SIZE)

        # Posterior information
        mean_B_post = lamb_out * lamb_in / ((1 - lamb_out) * (1 - lamb_in) + lamb_out * lamb_in)
        mean_B_post = torch.clamp(mean_B_post, min=clamp_min, max=clamp_max)
        # var_B_post = mean_B_post * (1 - mean_B_post)

        # Extrinsic information
        # var_B_ext = 1 / (1 / var_B_post - 1 / var_B_pri)
        # mean_B_ext = var_B_ext * (mean_B_post / var_B_post - mean_B_pri / var_B_pri)

        # Passed to Module A
        # var_A_pri_set[:, :, :, n] = var_B_ext
        # mean_A_pri_set[:, :, :, n] = mean_B_ext
        V_tilde_update_B[:, :, :, n] = mean_B_post

    V_update = V.clone()
    for batch_idx in range(BATCH_SIZE):
        V_update[batch_idx, :, :, index_amp[batch_idx, :]] = V_tilde_update_A[batch_idx, :, :, :]
    V_update = torch.clamp(V_update, min=0, max=1)
    VR_est = torch.floor(V_update + 0.5)

    U_update = U_torch
    ones_tensor = torch.ones((BATCH_SIZE, Nx, Nz), device=V_update.device, dtype=V_update.dtype)
    for batch_idx in range(BATCH_SIZE):
        for q in range((index_amp.shape[1])):
            grid_index = index_amp[batch_idx, q]
            kron_result = torch.kron(V_update[batch_idx, :, :, grid_index], ones_tensor[batch_idx, :, :])
            U_update[batch_idx, :, grid_index] = kron_result.t().reshape(-1, )  # 变形并按列优先顺序排列

    return VR_est, U_update


def function_compute_Grad(y_torch, F_torch1, z_torch1, polar_grid_torch, Mx, Mz, A_torch, lambda_, d, M, nGrid, sigma2,
                nPath, grad_Q_to_g11, grad_Q_to_g22, index_amp):
    BATCH_SIZE = y_torch.shape[0]
    A_complex = real_to_complex_4x4(A_torch.cuda())
    grid_update = polar_grid_torch.clone()
    grid_update[torch.arange(BATCH_SIZE).unsqueeze(1), index_amp, 0] = (
            grid_update[torch.arange(BATCH_SIZE).unsqueeze(1), index_amp, 0]
            + (2 / Mx / 40 * torch.sign(grad_Q_to_g11)))
    grid_update[torch.arange(BATCH_SIZE).unsqueeze(1), index_amp, 1] = (
            grid_update[torch.arange(BATCH_SIZE).unsqueeze(1), index_amp, 1]
            + (2 / Mz / 40 * torch.sign(grad_Q_to_g22)))

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
            k_lambda * (-delta_x * d * g1_batch + delta_x ** 2 * d ** 2 * (1 - g1_batch ** 2) / (
                        2 * r_batch)))).squeeze(2)
        az000 = (sqrt_Mz * torch.exp(
            k_lambda * (-delta_z * d * g2_batch + delta_z ** 2 * d ** 2 * (1 - g2_batch ** 2) / (
                        2 * r_batch)))).squeeze(2)

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

    F_update_complex10 = A_complex_update1
    F_update_grad0000 = complex_to_real_4x4(F_update_complex10)

    return F_update_grad0000, grid_update


def plot_complex_channel_heatmap(H, title='Channel Heatmap (Real | Imag)'):
    """
    将 512x1 复信道向量可视化为 32x32 热图：
    左 32x16 为实部，右 32x16 为虚部

    参数：
    H : numpy.ndarray 或 torch.Tensor
        形状为 (512,) 或 (512,1)，复数类型
    title : str
        图标题
    """

    # -------- 1. 转为 numpy --------
    if 'torch' in str(type(H)):
        H = H.detach().cpu().numpy()

    # -------- 2. reshape --------
    H = H.reshape(32, 16)

    # -------- 3. 分离实部和虚部 --------
    H_real = np.real(H)
    H_imag = np.imag(H)

    # -------- 4. 拼接 --------
    H_vis = np.concatenate([H_real, H_imag], axis=1)  # (32, 32)

    # -------- 5. 画图 --------
    plt.figure(figsize=(6, 5))
    im = plt.imshow(H_vis, cmap='seismic', aspect='auto')
    plt.colorbar(im, label='Value')

    # 分割线（实部 / 虚部）
    plt.axvline(x=15.5, color='black', linewidth=2)

    # 标注
    plt.text(8, -2, 'Real', ha='center', fontsize=12)
    plt.text(24, -2, 'Imag', ha='center', fontsize=12)

    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    plt.tight_layout()
    plt.show()



def plot_complex_channel_comparison(
    h,
    h_est=None,
    use_db=False,
    normalize=False,
    normalize_mode='separate',  # 'separate' or 'global'
    eps=1e-12,
    title1='Ground Truth',
    title2='Estimated Channel'
):
    """
    复信道热图（支持对比 + abs + dB + 归一化）

    参数：
    ----------
    h : numpy / torch
        (512,) 或 (512,1) 复数
    h_est : numpy / torch 或 None
        估计信道
    use_db : bool
        是否转为 dB: 20log10(|h|)
    normalize : bool
        是否归一化
    normalize_mode : str
        'separate'：各自归一化（默认）
        'global'  ：统一归一化（更适合公平对比 ⭐）
    eps : float
        防止 log(0)
    """

    # -------- 工具函数 --------
    def to_numpy(x):
        if x is None:
            return None
        if 'torch' in str(type(x)):
            return x.detach().cpu().numpy()
        return x

    def process(H):
        H = H.reshape(32, 16)
        H = np.abs(H)  # ⭐ 幅度
        return H

    # -------- 数据准备 --------
    h = to_numpy(h)
    h_est = to_numpy(h_est)

    H1 = process(h)
    H2 = process(h_est) if h_est is not None else None

    # -------- 归一化 --------
    if normalize:
        if h_est is None or normalize_mode == 'separate':
            H1 = H1 / (np.max(H1) + eps)
            if H2 is not None:
                H2 = H2 / (np.max(H2) + eps)
        elif normalize_mode == 'global':
            global_max = max(np.max(H1), np.max(H2))
            H1 = H1 / (global_max + eps)
            H2 = H2 / (global_max + eps)

    # -------- dB --------
    if use_db:
        H1 = 20 * np.log10(H1 + eps)
        if H2 is not None:
            H2 = 20 * np.log10(H2 + eps)

    # -------- 拼接（32×32）--------
    # H1_vis = np.concatenate([H1, H1], axis=1)
    H1_vis = H1
    if H2 is not None:
        # H2_vis = np.concatenate([H2, H2], axis=1)
        H2_vis = H2

    # -------- 画图 --------
    if H2 is None:
        plt.figure(figsize=(5, 4))
        im = plt.imshow(H1_vis, cmap='jet', aspect='auto')
        plt.title(title1)
        plt.colorbar(im)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        im1 = axes[0].imshow(H1_vis, cmap='jet', aspect='auto')
        axes[0].set_title(title1)

        im2 = axes[1].imshow(H2_vis, cmap='jet', aspect='auto')
        axes[1].set_title(title2)

        # ⭐ 统一 colorbar（关键）
        vmin = min(H1_vis.min(), H2_vis.min())
        vmax = max(H1_vis.max(), H2_vis.max())

        im1.set_clim(vmin, vmax)
        im2.set_clim(vmin, vmax)

        fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.9)

    plt.tight_layout()
    plt.show()