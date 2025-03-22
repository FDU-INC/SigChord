import torch
import numpy as np
from torch.distributions import Normal

from models.benchmark.dcs.dcs_torch import get_data_s

import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

print(torch.cuda.device_count())

def make_prior(num_latents):
    # Zero mean, unit variance prior.
    prior_mean = torch.zeros(num_latents)
    prior_scale = torch.ones(num_latents)

    return Normal(loc=prior_mean, scale=prior_scale)

def complex_to_real(input_tensor):
    """将复数张量的实部和虚部分离并交替拼接。"""
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    real_imag_tensor = torch.stack([real_part, imag_part], dim=-1)
    real_imag_tensor = real_imag_tensor.view(-1, 2 * input_tensor.shape[-1])
    return real_imag_tensor

def real_to_complex(input_tensor):
    """将实数张量还原为复数张量。"""
    real_part = input_tensor[..., ::2]
    imag_part = input_tensor[..., 1::2]
    return torch.complex(real_part, imag_part)

def project_z(z, project_method='clip'):
    """用于对 z 进行投影梯度下降的辅助函数。"""
    if project_method == 'norm':
        z_p = torch.nn.functional.normalize(z, p=2, dim=-1)  # L2 正则化
    elif project_method == 'clip':
        z_p = torch.clamp(z, -1, 1)  # 将值裁剪到 [-1, 1]
    else:
        raise ValueError(f'Unknown project_method: {project_method}')
    return z_p

def measure(A,X):
    """计算 X 和 A 的共轭转置的乘积。"""
    A = torch.tensor(A).to(device)
    A = A.to(dtype=torch.complex64)
    Y = torch.matmul(A, X.T)  # 矩阵乘法，A 的转置共轭
    return Y

def get_rip_loss(sig1, sig2,A):

    m1 = measure(A,real_to_complex(sig1)).T
    m2 = measure(A,real_to_complex(sig2)).T
    m1 = complex_to_real(m1)
    m2 = complex_to_real(m2)
    img_diff_norm = torch.norm(sig1 - sig2,p=2, dim=-1)
    m_diff_norm = torch.norm(m1 - m2,p=2, dim=-1)

    #return torch.square(img_diff_norm - m_diff_norm)
    return torch.square(img_diff_norm-m_diff_norm)

def gen_loss_fn(data,samples,A):
    data = real_to_complex(data)
    samples = real_to_complex(samples)
    #print("jh", data, samples)
    m_data = measure(A,data).T
    m_samples = measure(A,samples).T
    #print("diao", m_data, m_samples)
    m_data = complex_to_real(m_data)
    m_samples = complex_to_real(m_samples)
    #return torch.mean(torch.mean(torch.square(m_data - m_samples), dim=-1))
    return torch.mean(torch.mean(torch.square(m_data - m_samples), dim=-1))

def infer_gen_loss_fn(Y,samples):
    print("jg",Y)
    Y = complex_to_real(Y)
    samples = real_to_complex(samples)
    m_samples = measure(get_data_s.A,samples).T
    print("djg", m_samples)
    m_samples = complex_to_real(m_samples)
    #return torch.mean(torch.mean(torch.square(Y - m_samples), dim=-1))
    return torch.mean(torch.mean(torch.square(Y - m_samples), dim=-1))

def get_opt_cost(init_z,opt_z):
    optimisation_cost = torch.mean(
        torch.sum((opt_z - init_z) ** 2, -1))
    return optimisation_cost