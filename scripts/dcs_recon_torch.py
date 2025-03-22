import argparse
import torch
import models.benchmark.dcs.dcs_torch.train as train
import models.benchmark.dcs.dcs_torch.net as net
from torch.cuda import device_count

def get_args_parser():
    parser = argparse.ArgumentParser('CSGAN', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_epochs', default=66, type=int)
    parser.add_argument('--num_measurements', default=32, type=int)
    parser.add_argument('--num_latents', default=25, type=int)
    parser.add_argument('--num_z_iters', default=3, type=int)
    parser.add_argument('--z_step_size', default=0.01, type=float)
    parser.add_argument('--z_project_method', default='norm',type=str)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--n_Channels', default=8, type=int)
    parser.add_argument('--item_index', default=0, type=int)
    parser.add_argument('--output_dir', default='./output',type=str)
    return parser


def main(args):
    model = net.MLPGenerator(args.num_latents)
    model = train.train_model(model,args.num_latents,args.num_epochs,args.num_z_iters,args.batch_size,
                               args.learning_rate,args.z_step_size,args.n_Channels,args.num_index,args.output_dir)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

