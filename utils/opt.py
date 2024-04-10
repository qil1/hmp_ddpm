import os
import argparse
from pprint import pprint


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def boolean(arg):
    return arg.lower() == 'true'


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--seed', type=int, default=111)
        self.parser.add_argument('--gpu_index', type=int, default=0, help='cuda idx')
        self.parser.add_argument('--data_dir', type=str,
                                 default='/home/data/xuanqi/HMPdataset/h3.6m/dataset',
                                 help='path to dataset')
        self.parser.add_argument('--iter', type=str, default='199')
        self.parser.add_argument('--save_dir_name', type=str, default='tmp')
        self.parser.add_argument('--log_dir', type=str, default='./logs')
        self.parser.add_argument('--skip_rate', type=int, default=1, help='skip rate of samples')
        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--T', type=int, default=1000)
        self.parser.add_argument('--beta_1', type=float, default=1e-4)
        self.parser.add_argument('--beta_T', type=float, default=0.02)
        self.parser.add_argument('--w', type=float, default=0.)
        self.parser.add_argument('--S_model_dims', type=int, default=256)
        self.parser.add_argument('--S_trans_enc_num_layers', type=int, default=2)
        self.parser.add_argument('--S_num_heads', type=int, default=8)
        self.parser.add_argument('--S_dim_feedforward', type=int, default=1024)
        self.parser.add_argument('--S_dropout_rate', type=float, default=0.)
        self.parser.add_argument('--T_enc_hiddims', type=int, default=512)
        self.parser.add_argument('--T_dec_hiddims', type=int, default=512)
        self.parser.add_argument('--fusion_add', type=boolean, default='true')
        self.parser.add_argument('--t_his', type=int, default=10, help='t_his')
        self.parser.add_argument('--t_pred', type=int, default=25, help='t_pred')
        self.parser.add_argument('--joint_num', type=int, default=22, help='number of joints')
        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--lr', type=float, default=0.0001, help='lr')
        self.parser.add_argument('--num_epoch', type=int, default=200, help='total training epoch')
        self.parser.add_argument('--milestones', type=list_of_ints, default=[50, 100, 150, 180])
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--test_batch_size', type=int, default=32)
        self.parser.add_argument('--test_act', type=str, default="walking")
        self.parser.add_argument('--test_sample_num', type=int, default=-1, help='the num of sample, '
                                                                                 'that sampled from test dataset'
                                                                                 '{8,256,-1(all dataset)}')

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        self._print()
        return self.opt
