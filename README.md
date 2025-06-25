# About SigChord

SigChord is a physical layer sniffing demo that provides wide and deep views about the radio spectrum. It accepts sub-Nyquist samples to perform spectrum sensing, protocol identification and open header decoding. SigChord provides a based for wireless network telemetry and research. Key features include:

- **Wideband spectrum monitoring**: SigChord accepts sub-Nyquist IQ samples and can faithfully reconstruct the spectrum with sampling rates even **below twice the Landau rate**, which is regarded as the minimum rate for faithful blind signal recovery.
- **Deep learning based signal analysis**: By using simple Transformer-based models, SigChord simplifies protocol-specific signal processing and enables the following tasks:
    - spectrum sensing
    - protocol identification
    - open header decoding

# About Multi-coset Sub-Nyquist Sampling

SigChord is based on multi-coset sub-Nyquist sampling.

Traditionally, faithful signal sampling requires sampling rate at least the Nyquist rate. If the target spectrum is sparse, based on the compressed sensing theory, we can use psedo-random sampling to reduce the sampling rate. Multi-coset sampling is one of the most popular sub-Nyquist sampling methods. It uses multiple parallel low-speed ADCs each with unique time offset to sample the signals. For more information, you can refer to the following papers:

- Mishali, M., & Eldar, Y. C. (2010). From theory to practice: Sub-Nyquist sampling of sparse wideband analog signals. IEEE Journal of selected topics in signal processing, 4(2), 375-391.
- Mishali, M., & Eldar, Y. C. (2009). Blind multiband signal reconstruction: Compressed sensing for analog signals. IEEE Transactions on signal processing, 57(3), 993-1009.

---

# User Guide

## Environment Setup

We have verified in the environment below:

- Intel Xeon Platinum 8352V CPU @ 2.10GHz
- NVIDIA GeForce RTX 4090
- OS: Ubuntu 22.04
- CUDA 11.8
- Python 3.10.11
- PyTorch 2.5.1+cu118
- Matlab R2024a

In this repository,
- [models](./models) contains the neural network models.
- [data](./data) contains the dataset.
- [sigproc](./sigproc) contains the code for signal processing, especially multi-coset sub-Nyquist sampling.
- [scripts](./scripts/) contains the scripts for dataset synthesis, training and testing.

## Installation

Run the following commands to install the Python dependencies:

```base
conda create -n sigchord python=3.10
conda activate sigchord
python3 -m pip install -r requirements.txt
mkdir data
```

Besides, we rely on Matlab to synthesize DVB-S2 and Wi-Fi signals. Please have Matlab and Matlab Engine API for Python [(refer to this page)](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) installed.

### Synthesize Dataset

SigChord is trained on synthetic dataset. To synthesize the signals, run the following command:

```bash
python3 -m scripts.synthesize --out ./data/synthetic_train.h5 --data_len 300000
python3 -m scripts.synthesize --out ./data/synthetic_test.h5 --data_len 5000
```

Be aware that the synthesis process is time-consuming and requires a log of disk space.

Beside, to evaluate the generalization performance on spectrum sensing and signal recovery, you can synthesize Gaussian random signals with the following command:
```bash
python3 -m scripts.synthesize_random
```

## Training

### Spectrum Sensing

Run with default settings:
```bash
python3 -m scripts.train_ss --train ./data/synthetic_train.h5 --test ./data/synthetic_test.h5 --cosets 8
```

Or with custom arguments:
```bash
python3 -m scripts.train_ss --train ./data/synthetic_train.h5 --test ./data/synthetic_test.h5 --batch_size 512 --layers 2 --d_model 128 --cosets 8 --epoch 100
```

For WrT benchmark, run (note that number of layers is 3):

```bash
python3 -m scripts.train_ss --train ./data/synthetic_train.h5 --test ./data/synthetic_test.h5 --layers 3 --cosets 8 --model wrt
```

Or:
```bash
python3 -m scripts.train_ss --train ./data/synthetic_train.h5 --test ./data/synthetic_test.h5 --batch_size 512 --layers 3 --d_model 128 --cosets 8 --epoch 100 --model wrt
```

### Protocol Identification

Run with default settings:
```bash
python3 -m scripts.train_pkt_cls --train ./data/synthetic_train.h5 --test ./data/synthetic_test.h5 --cosets 8
```

Or with custom arguments:
```bash
python3 -m scripts.train_pkt_cls --train ./data/synthetic_train.h5 --test ./data/synthetic_test.h5 --batch_size 128 --layers 2 --d_model 128 --cosets 8 --epoch 100
```

For T-Prime benchmark, run (note that number of layers is 4):
```bash
python3 -m scripts.train_pkt_cls --train ./data/synthetic_train.h5 --test ./data/synthetic_test.h5 --cosets 8 --layers 3 --model tprime
```

### Header Decoding

Make sure you have trained the packet classifier before running the header decoding.
For DVB-S2/Non-HT Wi-Fi/HT Wi-Fi, run with default settings:
```bash
python3 -m scripts.train_decode --train ./data/synthetic_train.h5 --test ./data/synthetic_test.h5 --cosets 8 --model dvbs2
python3 -m scripts.train_decode --train ./data/synthetic_train.h5 --test ./data/synthetic_test.h5 --cosets 8 --model nonHT
python3 -m scripts.train_decode --train ./data/synthetic_train.h5 --test ./data/synthetic_test.h5 --cosets 8 --model HT
```

To run with custom arguments (take DVB-S2 as an example), please make sure the number of cosets matches the packet classifier (modify in the script). Although in principle they can be different, it is recommended to keep them the same. Then run:
```bash
python3 -m scripts.train_decode --train ./data/synthetic_train.h5 --test ./data/synthetic_test.h5 --batch_size 128 --layers 3 --d_model 384 --cosets 6 --epoch 100 --model dvbs2
```

## Inference

### Spectrum Sensing

Just make sure the model parameters match the arguments you used for training.

Run with default settings:
```bash
python3 -m scripts.infer_ss --params ./params/ss_L2_D128_C8_F16.pth --test ./data/multi_usrp_test.h5 --cosets 8
python3 -m scripts.infer_ss --params ./params/ss_L2_D128_C8_F16.pth --test ./data/multi_random.h5 --cosets 8
```

Or run with your own arguments:
```bash
python3 -m scripts.infer_ss --params ./params/ss_L2_D128_C8_F16.pth --test ./data/multi_usrp_test.h5 --cosets 8 --layers 2 --d_model 128 --batch_size 512
```

For WrT benchmark, run:
```bash
python3 -m scripts.infer_ss --params ./params/wrt_L3_D128_C8_F16.pth --test ./data/multi_usrp_test.h5 --cosets 8 --layers 3 --model wrt
python3 -m scripts.infer_ss --params ./params/wrt_L3_D128_C8_F16.pth --test ./data/multi_random.h5 --cosets 8 --layers 3 --model wrt
```

For DTMP benchmark, run:
```bash
python3 -m scripts.dtmp
```
You can configure the parameters in the script.

### Signal recovery

Run with trained spectrum sensor:
```bash
python3 -m scripts.recover_signals --test ./data/multi_usrp_test.h5 --params ./params/ss_L2_D128_C8_F16_epoch_100.pth --cosets 8
python3 -m scripts.recover_signals --test ./data/multi_random.h5 --params ./params/ss_L2_D128_C8_F16_epoch_100.pth --cosets 8
```

And you can also specify the arguments, just make sure they match the training settings of the spectrum sensor:
```bash
python3 -m scripts.recover_signals --test ./data/multi_usrp_test.h5 --params ./params/ss_L2_D128_C8_F16_epoch_100.pth --cosets 8 --layers 2 --d_model 128 --batch_size 512
```

For DTMP benchmark, run:
```bash
python3 -m scripts.dtmp
```
You can configure the parameters in the script.

To run DCS benchmarks, you need a different Python environment with Tensorflow 1.15.

To prepare the environment:
```bash
conda create -n dcs python=3.7.16
conda activate dcs
python -m pip install -r requirements_dcs.txt
```

Then run:
```bash
python3 -m scripts.dcs_recon_tf
```
You can configure the parameters in the script. To specify the dataset, modify [models/benchmark/dcs_tf/get_data.py](./models/benchmark/dcs_tf/get_data.py).

### Protocol Identification

Run with trained physical layer packet classifier:
```bash
python3 -m scripts.infer_pkt_cls --params ./params/pkt_cls_L2_D128_C8_F16.pth --test ./data/multi_usrp_test.h5 --cosets 8
```

Or run with your own arguments, just make sure they match the training settings of the packet classifier:
```bash
python3 -m scripts.infer_pkt_cls --params ./params/pkt_cls_L2_D128_C8_F16.pth --test ./data/multi_usrp_test.h5 --cosets 8 --layers 2 --d_model 128
```

### Header Decoding

Still, make sure you have trained the packet classifier before running the header decoding.
To evaluate DVB-S2/Non-HT Wi-Fi/HT Wi-Fi on the synthetic signals, run with default settings:
```bash
python3 -m scripts.infer_decode --test ./data/synthetic_test.h5 --cosets 8 --params params/dvbs2_L3_D384_C8_F32_epoch_600.pth --model dvbs2
python3 -m scripte.infer_decode --test ./data/synthetic_test.h5 --cosets 8 --params params/nonHT_L3_D384_C8_F32_epoch_600.pth --model nonHT
python3 -m scripts.infer_decode --test ./data/synthetic_test.h5 --cosets 8 --params params/HT_L3_D384_C8_F32_epoch_600.pth --model HT
```

Or you can run with custom arguments (take DVB-S2 as an example):
```bash
python3 -m scripts.infer_decode --test ./data/synthetic_test.h5 --cosets 8 --params params/dvbs2_L3_D384_C8_F32_epoch_600.pth --model dvbs2 --layers 3 --d_model 384 --batch_size 1024
```

To fine-tune on over-the-air signals, run with default settings:
```bash
python3 -m scripts.train_decode --train ./data/multi_usrp_train.h5 --test ./data/multi_usrp_test.h5 --cosets 8 --model dvbs2 --params params/dvbs2_L3_D384_C8_F32_epoch_600.pth --epoch 5
python3 -m scripts.train_decode --train ./data/multi_usrp_train.h5 --test ./data/multi_usrp_test.h5 --cosets 8 --model nonHT --params params/nonHT_L3_D384_C8_F32_epoch_600.pth --epoch 5
python3 -m scripts.train_decode --train ./data/multi_usrp_train.h5 --test ./data/multi_usrp_test.h5 --cosets 8 --model HT --params params/HT_L3_D384_C8_F32_epoch_600.pth --epoch 5
```
