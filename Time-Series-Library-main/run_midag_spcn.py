import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='MGFFN')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='MIDAG_SPCN',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/home/ljr/raw_files/ETT-small/', help='root path of the data file')
    parser.add_argument('--label_path', type=str, default='ETTm1.csv', help='label file')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--layers', type=int, default=1, help='MIDGCN layers')
    parser.add_argument('--use_revin', type=int, default=1, help='use revin')
    parser.add_argument('--use_last', type=int, default=0, help='use last time step')
    parser.add_argument('--num_clusters',type=int,default=64,help='numbers of cluster centers')
    parser.add_argument('--id_dim',type=int,default=8)
    parser.add_argument('--cluster_dim',type=int,default=8)
    parser.add_argument('--graph_dim',type=int,default=32)
    parser.add_argument('--enc_in',type=int,default=1)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--d_model',type=int,default=512)
    parser.add_argument('--use_freq',type=bool,default=True)

    parser.add_argument('--c_date',type=int,default=4)
    parser.add_argument('--embed', type=str, default='timeF',help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    # optimization
    parser.add_argument('--num_workers', type=int, default=32, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_pl{}_cel{}_update{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.layers,
                "True",
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_pl{}_cel{}_update{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.layers,
                "True",
                args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
