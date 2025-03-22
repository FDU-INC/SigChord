import torch
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
import time

n_avail_bands = 16
n_total_bands = 40
from models.benchmark.dcs.dcs_torch import get_data_s
from models.benchmark.dcs.dcs_torch import utils

torch.autograd.set_detect_anomaly(True)

device = utils.device

def train_model(model, num_latents = 20, num_epochs = 2500,num_z_iters = 10,batch_size = 8,
                learning_rate = 1e-4,z_step_size = 0.01,n_Channels = 8,item_index = 10 , output_dir = "./output"):
    z_step_size = torch.tensor(z_step_size).to(device)
    model = model.to(device)
    prior = utils.make_prior(num_latents)
    train_loader,scale = get_data_s.get_single_loader(batch_size,item_index,0,"usrp")
    # test_loader,test_signal,indices = get_data.get_test_loader(2400)
    running_loss = 0
    A = get_data_s.getA(n_Channels)
    print("start")
    start_time = time.time()
    for epoch in range(num_epochs):
        es = time.time()
        print(epoch)
        model.train()
        gt_signals = np.zeros((2400, n_avail_bands), dtype=np.complex64)
        recon_signals = np.zeros((2400, n_avail_bands), dtype=np.complex64)
        for i,signals in enumerate(train_loader):
            signals = signals[0].to(device)
            signals = utils.complex_to_real(signals)
            z = prior.sample((n_Channels,))
            z.requires_grad_(True)
            init_z = z.clone()
            init_z = init_z.to(device)
            z = z.to(device)
            initial_samples = model(z)
            samples = model(z)
            #print(z)

            for _ in range(num_z_iters):
                loss = utils.gen_loss_fn(signals,samples,A)+torch.mean(torch.norm(samples,p=1,dim=-1))
                loss = loss.to(device)
                z_grad = torch.autograd.grad(loss, z, create_graph=True)[0].to(device)
                z = z - z_step_size * z_grad  # Gradient descent step
                z = z.to(device)
                samples = model(z)
            samples = model(z)

            # r1 = utils.get_rip_loss(samples, initial_samples,A)
            # r2 = utils.get_rip_loss(samples, signals,A)
            # r3 = utils.get_rip_loss(initial_samples, signals,A)
            # rip_loss = torch.mean((r1 + r2 + r3) / 3.0)
            generator_loss = utils.gen_loss_fn(signals, samples,A)
            sparsity_loss = torch.mean(torch.norm(samples,p=1,dim=-1))
            model_loss = generator_loss+sparsity_loss
            model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            model_optimizer.zero_grad()  # 清空梯度
            model_loss.backward()  # 计算梯度
            model_optimizer.step()  # 更新参数
            running_loss += model_loss.item()
            signals = utils.real_to_complex(signals)
            samples = utils.real_to_complex(samples)
            gt_signals[i * batch_size:(i + 1) * batch_size, :] = signals.detach().cpu().numpy()
            recon_signals[i * batch_size:(i + 1) * batch_size, :] = samples.detach().cpu().numpy()

        r = recon_signals
        r = r.T
        r = r.reshape([-1]) * scale * (n_total_bands*2400)
        r = np.pad(r, (0, (n_total_bands-n_avail_bands)*2400), mode="constant")
        r_temp = np.fft.ifft(r)

        s = gt_signals
        s = s.T
        s = s.reshape([-1]) * scale * (n_total_bands*2400)
        s = np.pad(s, (0, (n_total_bands-n_avail_bands)*2400), mode="constant")
        s_temp = np.fft.ifft(s)

        # if epoch%2 ==0:
        #   plt.subplot(2, 2, 1)
        #   plt.plot(s_temp.real)
        #   plt.plot(s_temp.imag)
        #   plt.subplot(2, 2, 2)
        #   plt.plot(r_temp.real)
        #   plt.plot(r_temp.imag)
        #   plt.subplot(2, 2, 3)
        #   plt.plot(np.abs(s))
        #   plt.subplot(2, 2, 4)
        #   plt.plot(np.abs(r))
        #   plt.savefig(f"{output_dir}/rimage_epoch_{epoch + 1}.png")
        #   plt.clf()

        s_temp /= np.sqrt(np.mean(np.abs(s_temp) ** 2))
        r_temp /= np.sqrt(np.mean(np.abs(r_temp) ** 2))
        mse = np.mean(np.abs(s_temp - r_temp) ** 2)
        print("mse:",mse)

        ee = time.time()
        print("epoch time:",ee-es)
    end_time = time.time()
    print(end_time-start_time)
