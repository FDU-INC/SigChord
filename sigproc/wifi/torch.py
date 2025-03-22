import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import commpy.channelcoding.convcode as convcode

class Config:
    def __init__(self):
        conv_a = np.array([1,0,1,1,0,1,1])[::-1].copy()
        conv_b = np.array([1,1,1,1,0,0,1])[::-1].copy()
        conv_a = torch.from_numpy(conv_a).unsqueeze(0).unsqueeze(0)  # shape (1, 1, 7)
        conv_b = torch.from_numpy(conv_b).unsqueeze(0).unsqueeze(0)  # shape (1, 1, 7)
        self.conv_a = conv_a.float()
        self.conv_b = conv_b.float()

__config = Config()

def conv_encode(input: Tensor) -> Tensor:
    """
    PyTorch implementation of the my_encode function with batched input.
    
    Args:
        input (torch.Tensor): A 2D tensor of shape (batch_size, seq_len) containing binary inputs (0s and 1s).
    
    Returns:
        torch.Tensor: A 2D tensor of shape (batch_size, 2*seq_len) with the encoded sequence.
    """
    device = input.device
    # Pad the input bits with zeros at the beginning (6 zeros)
    padded_bits = F.pad(input.unsqueeze(1), (6, 0), mode='constant', value=0)  # shape (batch_size, 1, seq_len + 6)
    padded_bits = padded_bits.float()

    # Apply the convolutions (equivalent to convolve with valid mode)
    a_out = F.conv1d(padded_bits, __config.conv_a.to(device)) % 2  # shape (batch_size, 1, seq_len)
    b_out = F.conv1d(padded_bits, __config.conv_b.to(device)) % 2  # shape (batch_size, 1, seq_len)
    
    # Stack and flatten the output (equivalent to vstack and flatten in the original)
    res = torch.cat((a_out, b_out), dim=1)  # shape (batch_size, 2, seq_len)
    res = res.permute(0, 2, 1).flatten(-2, -1)  # shape (batch_size, 2*seq_len)

    return res.int()

def interleave(raw_bits: Tensor, n_bpsc: int) -> Tensor:
    """
    PyTorch version of the interleave function supporting batched input.
    
    Args:
        raw_bits (torch.Tensor): A 2D tensor of shape (batch_size, n_cbps) containing binary input bits.
        n_bpsc (int): Number of bits per symbol.
        
    Returns:
        torch.Tensor: A 2D tensor of shape (batch_size, n_cbps) with the interleaved bits.
    """
    batch_size, n_cbps = raw_bits.shape
    s = max(n_bpsc // 2, 1)
    
    k_ids = torch.arange(n_cbps, dtype=torch.long, device=raw_bits.device).unsqueeze(0)  # Shape (1, n_cbps)
    i_ids = (n_cbps // 16) * (k_ids % 16) + k_ids // 16  # Shape (1, n_cbps)
    j_ids = s * (i_ids // s) + (i_ids + n_cbps - 16 * i_ids // n_cbps) % s  # Shape (1, n_cbps)
    
    # Expand to match batch size
    j_ids = j_ids.expand(batch_size, -1)  # Shape (batch_size, n_cbps)
    
    # Perform interleaving (using advanced indexing)
    res = torch.zeros_like(raw_bits)
    res.scatter_(1, j_ids, raw_bits)  # Scatter raw_bits into res at indices specified by j_ids
    
    return res

def deinterleave(raw_bits: Tensor, n_bpsc: int) -> Tensor:
    """
    PyTorch version of the deinterleave function supporting batched input.
    
    Args:
        raw_bits (torch.Tensor): A 2D tensor of shape (batch_size, n_cbps) containing binary input bits.
        n_bpsc (int): Number of bits per symbol.
        
    Returns:
        torch.Tensor: A 2D tensor of shape (batch_size, n_cbps) with the interleaved bits.
    """
    batch_size, n_cbps = raw_bits.shape
    s = max(n_bpsc // 2, 1)
    
    j_ids = torch.arange(n_cbps, dtype=torch.long, device=raw_bits.device).unsqueeze(0)  # Shape (1, n_cbps)
    i_ids = s * (j_ids // s) + (j_ids + 16 * j_ids // n_cbps) % s
    k_ids = 16 * i_ids - (n_cbps - 1) * (16 * i_ids // n_cbps)
    
    # Expand to match batch size
    k_ids = k_ids.expand(batch_size, -1)  # Shape (batch_size, n_cbps)
    
    # Perform deinterleaving (using advanced indexing)
    res = torch.zeros_like(raw_bits)
    res.scatter_(1, k_ids, raw_bits)  # Scatter raw_bits into res at indices specified by j_ids
    
    return res


__trellis = convcode.Trellis(
    np.array([6]), np.array([[0o133, 0o171]]), polynomial_format="Matlab"
)

def viterbi_decode(coded_bits):
    """
    Decodes a stream of convolutionally encoded bits using the Viterbi Algorithm.
    
    Parameters
    ----------
    coded_bits : 2D Tensor (B, N)
        Batch of streams of convolutionally encoded bits which are to be decoded.
    trellis : Trellis object
        Trellis representing the convolutional code.
    tb_depth : int
        Traceback depth. Default is 5 times the number of memories in the code.
    decoding_type : str {'hard', 'soft', 'unquantized'}
        Type of decoding to be used.
    
    Returns
    -------
    decoded_bits : 2D Tensor (B, L)
        Batch of decoded bit streams.
    """
    decoded = [convcode.viterbi_decode(coded_bits[i].cpu().numpy(), __trellis) for i in range(coded_bits.shape[0])]
    decoded = np.array(decoded)
    decoded = torch.tensor(decoded, dtype=torch.int32, device=coded_bits.device)
    return decoded