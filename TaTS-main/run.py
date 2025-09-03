import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np
import re

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='midgcn',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='Traffic.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=8, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=4, help='start token length')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--text_emb', type=int, default=96, help='prediction sequence length')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # midgcn model define
    parser.add_argument('--layers', type=int, default=1, help='MIDGCN layers')
    parser.add_argument('--use_revin', type=int, default=1, help='use revin')
    parser.add_argument('--use_last', type=int, default=1, help='use revin')
    parser.add_argument('--num_clusters',type=int,default=32,help='numbers of cluster centers')
    parser.add_argument('--id_dim',type=int,default=8)
    parser.add_argument('--cluster_dim',type=int,default=8)
    parser.add_argument('--graph_dim',type=int,default=32)
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')

    # model define
    parser.add_argument('--order', type=int, default=4, help='P-net order for STPTN')
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=4, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=2, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default="avg",
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                    help='whether to use future_temporal_feature; True 1 False 0')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    #  new pars
    parser.add_argument('--llm_model', type=str, default='BERT', help='LLM model') # LLAMA2, LLAMA3, GPT2, BERT, GPT2M, GPT2L, GPT2XL, Doc2Vec, ClosedLLM
    parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--text_path', type=str, default="None")
    parser.add_argument('--type_tag', type=str, default="#F#")
    parser.add_argument('--text_len', type=int, default=3)
    parser.add_argument('--learning_rate2', type=float, default=1e-2, help='mlp learning rate')
    parser.add_argument('--learning_rate3', type=float, default=1e-3, help='proj learning rate')
    parser.add_argument('--prompt_weight', type=float, default=0.01, help='prompt weight')#please tune this hyperparameter for combining
    parser.add_argument('--prior_weight', type=float, default=0.01, help='prompt weight')#please tune this hyperparameter for combining
    parser.add_argument('--pool_type', type=str, default='avg', help='pooling type') #avg min max attention
    parser.add_argument('--date_name', type=str, default='end_date', help='matching date name in csv') #mlp linear
    parser.add_argument('--addHisRate', type=float, default=0.5, help='add historical rate')
    parser.add_argument('--init_method', type=str, default='normal', help='init method of combined weight')
    parser.add_argument('--learning_rate_weight', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--save_name', type=str, default='result_longterm_forecast', help='save name')
    parser.add_argument('--use_fullmodel', type=int, default=0, help='use full model or just encoder')
    parser.add_argument('--use_closedllm', type=int, default=0, help='use closedllm or not')    
    parser.add_argument('--huggingface_token', type=str, help='your token of huggingface;need for llama3')
    parser.add_argument('--patch_len', type=int, default=2, help='patch length')
    parser.add_argument('--stride', type=int, default=2, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')

    # freq
    parser.add_argument('--embedding', type=str, default="fourier_interplate", help='embedding type')
    parser.add_argument('--quantile', type=float, default=0.90, help='quantile for freq')
    parser.add_argument('--bandwidth', type=float, default=1, help='static filter bandwidth')
    parser.add_argument('--filter_type', type=str, default='all', help='filter type')
    parser.add_argument('--top_K_static_freqs', type=int, default=10, help='build static filter')

    # GNN
    parser.add_argument('--tvechidden', type=int, default=1, help='scale vec dim')
    parser.add_argument('--nvechidden', type=int, default=1, help='variable vec dim')
    parser.add_argument('--use_tgcn', type=int, default=1, help='use cross-scale gnn')
    parser.add_argument('--use_ngcn', type=int, default=1, help='use cross-variable gnn')
    parser.add_argument('--anti_ood', type=int, default=1, help='simple strategy to solve data shift')
    parser.add_argument('--scale_number', type=int, default=4, help='scale number')
    parser.add_argument('--hidden', type=int, default=8, help='channel dim')
    parser.add_argument('--tk', type=int, default=10, help='constant w.r.t corss-scale neighbors')


    # MSGNet
    parser.add_argument('--num_nodes', type=int, default=7, help='to create Graph')
    parser.add_argument('--subgraph_size', type=int, default=1, help='neighbors number')
    parser.add_argument('--tanhalpha', type=float, default=3, help='')
    parser.add_argument('--node_dim', type=int, default=100, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=16, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')


    # sageformer params
    parser.add_argument('--cls_len', type=int, default=3, help='global token length')
    parser.add_argument('--graph_depth', type=int, default=3, help='graph aggregation depth')
    parser.add_argument('--knn', type=int, default=16, help='graph nearest neighbors')
    parser.add_argument('--embed_dim', type=int, default=16, help='node embed dim')

    parser.add_argument('--begin_order', type=int, default=1, help='begin_order')

    
    #ModernTCN
    parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
    parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
    parser.add_argument('--ffn_ratio', type=int, default=2, help='ffn_ratio')
    parser.add_argument('--patch_size', type=int, default=2, help='the patch size')
    parser.add_argument('--patch_stride', type=int, default=2, help='the patch stride')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1,1], help='num_blocks in each stage')
    parser.add_argument('--large_size', nargs='+',type=int, default=[51,51,51,51], help='big kernel size')
    parser.add_argument('--small_size', nargs='+',type=int, default=[5,5,5,5], help='small kernel size for structral reparam')
    parser.add_argument('--dims', nargs='+',type=int, default=[64,64,64,64], help='dmodels in each stage')
    parser.add_argument('--dw_dims', nargs='+',type=int, default=[64,64,64,64])

    parser.add_argument('--small_kernel_merged', type=bool, default=False, help='small_kernel has already merged or not')
    parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
    parser.add_argument('--use_multi_scale', type=bool, default=False, help='use_multi_scale fusion')

    # LATS
    parser.add_argument('--text_dim', type=int, default=12, help='text embedding dimension after low rank projection')
    args = parser.parse_args()
    domain= re.search(r'/([^/]+)$', args.root_path).group(1)
    print("now running on domain {} model {} ".format(domain,args.model))
    if args.model=="LightTS":   
        if args.pred_len<args.seq_len:
            args.seq_len=args.pred_len
    if args.llm_model=="BERT":
        args.llm_dim=768
    elif args.llm_model=="GPT2":
        args.llm_dim=768
    elif args.llm_model=="LLAMA2":
        args.llm_dim=4096
    elif args.llm_model=="LLAMA3":
        args.llm_dim=4096
    elif args.llm_model=="GPT2M":
        args.llm_dim=1024
    elif args.llm_model=="GPT2L":
        args.llm_dim=1280
    elif args.llm_model=="GPT2XL":
        args.llm_dim=1600
    elif args.llm_model=="Doc2Vec":
        args.llm_dim=64
    elif args.llm_model=="ClosedLLM":
        args.llm_model="BERT" #just for encoding
        args.llm_dim=768
        args.use_closedllm=1


    args.features = 'S' #'M', 'S'
    args.enc_in = 1 + args.text_emb
    args.dec_in = 1 + args.text_emb
    args.c_out = 1 + args.text_emb
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!overwrite features to 'S' for univariate time series data and dim=1")
    fix_seed = args.seed
    args.prompt_weight=args.prior_weight
    print("Now using seed {}".format(fix_seed))
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

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
        raise ValueError('Task name not supported')

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            now_mse=exp.test(setting, test=1)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        now_mse=exp.test(setting, test=1)

        torch.cuda.empty_cache()
    
