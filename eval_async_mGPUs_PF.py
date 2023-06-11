# License: MIT

import numpy as np
import torch
import os
import argparse
import random

import matplotlib.pyplot as plt
from mabo import ParallelOptimizer, space as sp
from multiprocessing import current_process, Queue

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from exp.exp_main import Exp_Main

import time

fix_seed = 2022
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Define Search Space
space = sp.Space()
seq_len = sp.Int("seq_len", 4, 96, q=4)
label_rate = sp.Categorical('label_rate', choices=[1/4,1/2,1])
period_rate = sp.Categorical('period_rate', choices=[1/8, 1/4, 1/2])
moving_avg = sp.Int("moving_avg", 17, 247, q=10, default_value=27)
drop = sp.Real("drop", 0.01, 0.5, default_value=0.3)
lr = sp.Real("lr", 1e-5, 1e-1, log=True)

space.add_variables([seq_len, label_rate, period_rate, moving_avg, drop, lr])


def get_args():

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int,  default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='AutoPeriodformerED',
                        help='model name, options: [AutoPeriodformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, default='ETTm2', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../data/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTm2.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # hyper parameters
    # parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    # parser.add_argument('--label_len', type=int, default=96, help='input sequence length')
    # parser.add_argument('--moving_avg', type=int, default=37, help='window size of moving average')
    # parser.add_argument('--period', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    # parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    
    parser.add_argument('--loss', type=str, default='mae', help='loss function')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')

    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)

    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    #parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    return parser.parse_args()

def get_vars(config):

    args = get_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    ### Todo Change Variable

    args.seq_len = config['seq_len']
    args.label_len = int(config['label_rate'] * args.seq_len)
    args.period = int(config['period_rate'] * args.seq_len)

    args.moving_avg = config['moving_avg']
    args.dropout = config['drop']
    args.learning_rate = config['lr']

    # print(args.seq_len, args.label_len, args.period, 
    # args.moving_avg, args.dropout,  args.learning_rate )
    # exit()
    # setting record of experiments
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_ll{}_pe{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.label_len,
        args.period,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des)

    return args, setting


def objective(config):

    gpu_id = queue.get()
    torch.cuda.set_device(gpu_id)
    torch.cuda.is_available()
    
    args, setting = get_vars(config)
    args.gpu_id = gpu_id
    print('Args in experiment:')
    print(args)

    exp = Exp_Main(args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    val_error =  exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)
    torch.cuda.empty_cache()
    queue.put(gpu_id)
    return {'objectives': (val_error,)}


NUM_GPUS = 8

if __name__ == "__main__":

    queue = Queue()
    for gpu_ids in range(NUM_GPUS):
        queue.put(gpu_ids)

    # Parallel Evaluation on Local Machine
    opt = ParallelOptimizer(
        objective,
        space,
        parallel_strategy='async',
        batch_size=NUM_GPUS,
        batch_strategy='default',
        num_objectives=1,
        num_constraints=0,
        max_runs=16,
        surrogate_type='gp',
        #surrogate_type='auto',
        time_limit_per_trial=18000,
        task_id='parallel_async',
    )

    begin_time = time.time()
    history = opt.run()
    print('time = ', time.time() - begin_time)

    print(history)

    # history.plot_convergence(true_minimum=0.397887)
    # plt.show()

    # history.visualize_hiplot()
