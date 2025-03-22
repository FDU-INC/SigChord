import torch
import h5py
import tqdm
from sigproc.cs import multicoset as M
from sigproc.prelude import awgn_torch

def solve(Y: torch.Tensor, A: torch.Tensor, e: float, sigma2: float, C: float, sparsity: int = None):
    R = Y.clone()
    S = set()
    ER = torch.inf
    Theta = {}
    
    p, L = A.shape
    N = Y.shape[-1]
    
    t = 1
    max_iter = min(sparsity, p) if sparsity is not None else p
    
    while t <= max_iter:
        alpha_t = torch.argmax(torch.norm(A.T.conj() @ R, dim=-1)).item()
        S.add(alpha_t)
        At = A[:, alpha_t]
        Theta_t = torch.linalg.lstsq(At.unsqueeze(1), R, rcond=None).solution
        Theta[alpha_t] = Theta_t
        R -= At.unsqueeze(1) @ Theta_t
        ER = torch.norm(R)
        t += 1
    
    S_res = S  # No filtering step
    Theta_res = torch.zeros((L, N), dtype=torch.cdouble, device=Y.device)
    for s in S_res:
        Theta_res[s] = Theta[s]
    
    return Theta_res, S_res

def main():
    snr = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with h5py.File("./data/multi_usrp_sub.h5", "r") as file:
        waveforms = torch.tensor(file["waveforms"][:], dtype=torch.float32)
        bands = torch.tensor(file["bands"][:] >= 0, dtype=torch.uint8)
        
        # data_len = bands.shape[0]
        data_len = 1024
        # for snr in [-10, -6, -2, 2, 6]:
        # for cosets in [5, 6, 8, 10, 11]:
        for cosets in [8]:
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            fp, fn, positives, negatives = 0, 0, 0, 0
            mses = []
            
            import time
            start_time = time.time()
            with tqdm.tqdm(range(data_len)) as pbar:
                for i in pbar:
                    waveform_r = waveforms[i].T
                    waveform = waveform_r[::2] + 1j * waveform_r[1::2]
                    waveform = waveform.to(torch.cdouble).to(device)
                    
                    sig_gt = waveform.T[:, M.DEFAULT_OFFSETS_SORTED_IDS].flatten()
                    sig_gt /= torch.sqrt(torch.mean(torch.abs(sig_gt) ** 2))
                    waveform = awgn_torch(waveform, snr)
                    
                    Y, A = M.dft_torch(waveform[:cosets], M.DEFAULT_OFFSETS[:cosets], 40)
                    Y, A = Y.to(device), A.to(device)
                    
                    X_rec, sup = solve(Y, A, 0.1, 16.8, 1, 6)
                    sig_rec = torch.fft.ifft(X_rec.flatten())
                    sig_rec /= torch.sqrt(torch.mean(torch.abs(sig_rec) ** 2))
                    
                    mses.append(torch.mean((sig_gt - sig_rec).abs() ** 2).item())
                    
                    pred = torch.zeros(40, dtype=torch.uint8, device=device)
                    pred[list(sup)] = 1
                    pred = pred[:16]
                    
                    band_i = bands[i].to(device)
                    fp += torch.sum((pred == 1) & (band_i == 0)).item()
                    fn += torch.sum((pred == 0) & (band_i == 1)).item()
                    positives += torch.sum(band_i == 1).item()
                    negatives += torch.sum(band_i == 0).item()
                    
                    pbar.set_postfix(FA=fp / negatives, MD=fn / positives, MSE=sum(mses) / (len(mses) + 1e-6))

            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"usage: {(total_mem - free_mem) / 1024 / 1024} MB", flush=True)

            end_time = time.time()
            print(f"Time elapsed: {end_time - start_time} s")

if __name__ == "__main__":
    main()
