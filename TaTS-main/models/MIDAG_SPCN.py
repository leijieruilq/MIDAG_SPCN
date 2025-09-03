import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import gumbel_softmax
import math
import torch.fft
from einops import rearrange

class Mahalanobis_mask(nn.Module):
    def __init__(self, input_size):
        super(Mahalanobis_mask, self).__init__()
        frequency_size = input_size // 2 + 1
        self.A = nn.Parameter(torch.randn(frequency_size, frequency_size), requires_grad=True)
    def calculate_prob_distance(self, X): #(batchsize, channels, seq_len)
        XF = torch.abs(torch.fft.rfft(X, dim=-1)) #(batchsize, channels, int(seq_len)/2 + 1)
        X1 = XF.unsqueeze(2)
        X2 = XF.unsqueeze(1)
        # B x C x C x D
        diff = X1 - X2 #(batchsize, channels, channels, int(seq_len)/2 + 1)
        temp = torch.einsum("dk,bxck->bxcd", self.A, diff) #(int(seq_len)/2 + 1,int(seq_len)/2 + 1), (batchsize, channels, channels, int(seq_len)/2 + 1)
        dist = torch.einsum("bxcd,bxcd->bxc", temp, temp) #(batchsize, channels, channels)
        # exp_dist = torch.exp(-dist)
        exp_dist = 1 / (dist + 1e-10) #(batchsize, channels, channels)
        # 对角线置零
        identity_matrices = 1 - torch.eye(exp_dist.shape[-1]) #(batchsize, channels, channels)
        mask = identity_matrices.repeat(exp_dist.shape[0], 1, 1).to(exp_dist.device) #(batchsize, channels, channels)
        exp_dist = torch.einsum("bxc,bxc->bxc", exp_dist, mask) #(batchsize, channels, channels)
        exp_max, _ = torch.max(exp_dist, dim=-1, keepdim=True) #(batchsize, channels, 1)
        exp_max = exp_max.detach() #将相似度矩阵的对角线元素置零。这是因为在构建通道间关系掩码时，通常不考虑一个通道与自身的关系
        # B x C x C
        p = exp_dist / exp_max #(batchsize, channels, channels) 对每个通道（矩阵的每一行）的相似度进行归一化，使得该通道与其他通道的最大相似度为1。这形成了一个概率化的关系矩阵 P
        identity_matrices = torch.eye(p.shape[-1])
        p1 = torch.einsum("bxc,bxc->bxc", p, mask) #(batchsize, channels, channels) 去对角线化
        diag = identity_matrices.repeat(p.shape[0], 1, 1).to(p.device)
        p = (p1 + diag) * 0.99 #再次确保非对角线元素是基于 p1（对角线为0的概率），然后将对角线元素设置为1（表示一个通道与自身是完全相关的，这在注意力机制中通常是期望的）
        #，最后乘以一个折扣因子 0.99 。这个折扣因子 γ 用于避免绝对的连接。
        return p #(batchsize, channels, chaneels)

    def bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape
        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        r_flatten_matrix = 1 - flatten_matrix
        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix) #(32*7*7,1)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix) #(32*7*7,1)
        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1) #(32*7*7,2)
        resample_matrix = gumbel_softmax(new_matrix, hard=True) #(32,7,7)
        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)
        return resample_matrix

    def forward(self, X):
        p = self.calculate_prob_distance(X) #利用马氏距离度量通道间概率
        # bernoulli中两个通道有关系的概率
        mask = self.bernoulli_gumbel_rsample(p) #(batchsize,channels,channels)
        return mask

class RevIN(nn.Module):
    def __init__(self, num_channels, num_nodes, eps = 1e-5, affine = True):
        super().__init__()
        self.num_channels = num_channels
        self.num_nodes = num_nodes
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, self.num_channels, self.num_nodes, 1))
            self.beta = nn.Parameter(torch.zeros(1, self.num_channels, self.num_nodes, 1))
        self.mean = None
        self.std = None

    def forward(self, x):
        self.mean = torch.mean(x, dim=-1, keepdim=True)
        self.std = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps)
        x_norm = (x - self.mean) / self.std
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm

    def reverse(self, x_pred_norm):
        if self.affine:
            x_no_affine = (x_pred_norm - self.beta) / (self.gamma + 1e-8) 
            x_denorm = x_no_affine * self.std + self.mean
        else:
            x_denorm = x_pred_norm * self.std + self.mean
        return x_denorm

def multi_order(s_out, order_0, n):
        solves = []
        stats = []
        for i in range(1,int(n)):
            c = 4 * (order_0**i)
            m = n-i
            order_low = (s_out/c)**(1/(m+1)) 
            order_up = (s_out/c)**(1/m)
            order_1 = order_up//1
            if (not ((order_1 <= order_up) and (order_1 > order_low))) or (order_1 == 1):
                continue
            else:
                solves.append([order_0,order_1,i,m])
                stats.append(order_0*i+order_1*m)
        idx = np.argmin(stats)
        solves = solves[idx]
        order_list = []
        for i in range(int(n)):
            idx = np.argmax(solves[2:4])
            order_list.append(int(solves[idx]))
            solves[2+idx] -= 1
        return order_list

def calculate_order(c_in, s_in, s_out, order_in, order_out):
    n_in = (np.log(s_in)/np.log(order_in))//1 #5
    order_out_low = (s_out/c_in)**(1/(1+n_in))
    order_out_up = (s_out/c_in)**(1/(n_in))
    order_out = order_out_up//1
    n_out = (np.log(s_out/2)/np.log(order_out))//1
    if (not ((order_out <= order_out_up) and (order_out > order_out_low))) or (order_out == 1):
        Warning('Order {} is not good for s_in, s_out')
        order_out_list = multi_order(s_out, order_out, n_in)
    else:
        order_out_list = [int(order_out)]*int(n_out)
    order_in_list = [int(order_in)]*int(n_in)
    return int(n_in), order_in_list, order_out_list

class FreqConv(nn.Module):
    def __init__(self, c_in, inp_len, pred_len, kernel_size=3, dilation=1, order=2):
        nn.Module.__init__(self)
        self.inp_len = inp_len
        self.pred_len = pred_len
        self.order_in = order
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.c_in = c_in
        self.projection_init()

    def projection_init(self):
        kernel_size = self.kernel_size
        dilation = self.dilation
        padding = (kernel_size-1)*(dilation-1) + kernel_size -1
        self.pad_front = padding//2
        self.pad_behid = padding - self.pad_front

        inp_len = self.inp_len
        pred_len = self.pred_len
        order_in = self.order_in
        s_in = (inp_len+1)//2
        s_out = self.c_in
        n, order_in, order_out = calculate_order(self.c_in, s_in, pred_len, order_in, None)
        self.Convs = nn.ModuleList()
        self.Pools = nn.ModuleList()
        for i in range(n):
            self.Convs.append(nn.Conv2d(s_out, order_out[i]*s_out, (1,kernel_size), dilation=(1,self.dilation)))
            self.Pools.append(nn.AvgPool2d((1,order_in[i])))
            s_in = s_in // order_in[i]
            s_out = s_out * order_out[i]
        self.final_conv = nn.Conv2d(s_out,pred_len,(1,s_in))
        self.freq_layers = n

    def forward(self, x1, x2):
        # return x1 + x2 + x3x
        x1_fft = torch.fft.rfft(x1)
        x2_fft = torch.fft.rfft(x2)
        h = torch.cat((x1_fft.imag, x2_fft.imag,
                       x1_fft.real, x2_fft.real),dim=2)
        h = h.transpose(1,2)
        for i in range(self.freq_layers):
            h = F.pad(h,pad=(self.pad_front,self.pad_behid,0,0))
            h = self.Convs[i](h)
            #np.save("conv_weight_"+str(i)+".npy", self.Convs[i].weight.cpu().detach().numpy())
            h = self.Pools[i](h)
        y = self.final_conv(h).permute(0,2,3,1) + x1 + x2
        #np.save("conv_weight_final.npy", self.final_conv.weight.cpu().detach().numpy())
        return y

class Indepent_Linear(nn.Module):
    def __init__(self, s_in, s_out, channels, share=False, dp_rate=0.5):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.randn((channels,1,s_in,s_out)))
        self.bias = nn.Parameter(torch.randn((channels,1,s_out)))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.bias)
        self.share = share
        self.dropout = nn.Dropout(dp_rate)
        if share:
            self.weight = nn.Parameter(torch.randn((1,1,s_in,s_out)))
            self.bias = nn.Parameter(torch.randn((1,1,s_out)))
            nn.init.xavier_uniform_(self.weight)
            nn.init.xavier_uniform_(self.bias)

    def forward(self, x):
        h = torch.einsum('BCNI,CNIO->BCNO',(x,self.weight))+self.bias
        return h
    
class fconv(nn.Module):
    def __init__(self, c_in, inp_len):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(c_in, c_in, kernel_size=3, padding=1)
        self.linear1 = nn.Linear(4,1)
        self.linear2 = nn.Linear((inp_len+1)//2 + 1, inp_len)
    def forward(self, x1, x2):
        x1_fft = torch.fft.rfft(x1)
        x2_fft = torch.fft.rfft(x2)
        h = torch.cat((x1_fft.imag, x2_fft.imag,
                       x1_fft.real, x2_fft.real),dim=2)
        h = self.linear2(h)
        h = self.linear1(h.permute(0,1,3,2)).permute(0,1,3,2)
        return h

class fft_mlp(nn.Module):
    def __init__(self,seq_in,seq_out,channels):
        nn.Module.__init__(self)
        self.amp_linear = Indepent_Linear(seq_in//2 + 1, seq_out, channels)
        self.phase_linear = Indepent_Linear(seq_in//2 + 1, seq_out, channels)
    def forward(self, x):
        x = torch.fft.rfft(x)
        x = self.amp_linear(x.real) + self.phase_linear(x.imag)
        return x  

class gated_mlp(nn.Module):
    def __init__(self, seq_in, seq_out, d_model, channels, dp_rate=0.3):
        nn.Module.__init__(self)
        self.channels = channels
        self.fft = fft_mlp(seq_in,seq_out,channels)     
        self.update = nn.Linear(seq_out, d_model)
        self.dropout = nn.Dropout(dp_rate)

    def forward(self, x):
        h = x
        h = self.fft(x)
        h = self.update(h)
        h = F.tanh(h)
        h = self.dropout(h)
        return h
    
class MIDGCN(nn.Module):
    def __init__(self, configs, out_len):
        super(MIDGCN, self).__init__()
         # 1. 变量身份嵌入 (静态+动态)
        self.id_emb= nn.Embedding(configs.enc_in, configs.id_dim)
        self.dynamic_id_proj = nn.Sequential(nn.Linear(configs.d_model, configs.d_model // 2),
                                             nn.ReLU(),
                                             nn.Linear(configs.d_model // 2,configs.id_dim))
        # 2. 可学习的聚类中心，通过 nn.Embedding 实现，其 .weight 属性即为聚类中心
        self.cluster_emb = nn.Embedding(configs.num_clusters, configs.cluster_dim)
        # 3. 用于将身份嵌入投影到与聚类中心进行相似度计算的空间（可选，但常用于匹配维度或增加表达力）
        self.id_to_cluster = nn.Linear(configs.id_dim, configs.cluster_dim)
        self.neg_inf = -1e9 * torch.eye(configs.enc_in, device="cuda:" + str(configs.gpu))
        self.graph_proj = nn.Linear(configs.id_dim + configs.cluster_dim, configs.graph_dim)
        #self.graph_proj = nn.Linear(configs.id_dim, configs.graph_dim)
        gcn_input_dim = configs.d_model + configs.id_dim  + configs.cluster_dim
        self.context_weight = nn.Parameter(torch.randn(configs.enc_in, gcn_input_dim))
        nn.init.xavier_normal_(self.context_weight)
        self.linear = nn.Linear(gcn_input_dim,out_len)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self,x):
        b,c,n,t = x.shape
        device = x.device
        # --- 1. 静态身份嵌入 + 动态身份嵌入 ---
        var_indices = torch.arange(c, device=device)
        static_id_embeds = self.id_emb(var_indices) # (c, id_dim)
        dynamic_id_embeds = self.dynamic_id_proj(x).squeeze(dim=-2) #(b,c,id_dim)
        id_embeds = static_id_embeds.unsqueeze(0).expand(b, -1, -1) + dynamic_id_embeds #(b,c,id_dim)
        #id_embeds = static_id_embeds.expand(b, -1, -1)
        # id_embeds = dynamic_id_embeds
        # --- 2. 可学习的软聚类分配 ---
        cluster_centers = self.cluster_emb.weight # (num_clusters, cluster_dim)
        id_for_clustering = self.id_to_cluster(id_embeds) # (b, c, cluster_dim)
        cluster_assignment_probs = F.softmax(torch.einsum("bcl,nl->bcn",id_for_clustering,cluster_centers),dim=-1) # (b,c,num_clusters)
        cluster_embeds = torch.matmul(cluster_assignment_probs, cluster_centers) # (b,c,cluster_dim)
        # --- 3. 自适应图结构学习 ---
        graph_emb = torch.cat([id_embeds, cluster_embeds], dim=-1)  # (b, c, id_dim + cluster_dim)
        #graph_emb = torch.cat([id_embeds], dim=-1)
        graph_emb = self.graph_proj(graph_emb) # (b, c, graph_dim)
        adj = torch.einsum("bcg,bmg->bcm",graph_emb,graph_emb)
        adj = F.relu(adj)
        adj = adj + self.neg_inf.unsqueeze(0).expand(b, -1, -1)
        adj = torch.where(adj<=0.0,self.neg_inf,adj)
        adj = F.softmax(adj, dim=-1) # (b, c, c)
        # --- 4. 特征融合 ---
        gcn_input = torch.cat([x.squeeze(dim=-2), id_embeds, cluster_embeds], dim=-1) # (b, c, d_model + id_dim + cluster_dim)
        #gcn_input = torch.cat([x.squeeze(dim=-2), id_embeds], dim=-1) 
        # gcn_input = torch.cat([x.squeeze(dim=-2)], dim=-1) 
        w = F.tanh(torch.einsum("bod,ol->bdl",torch.einsum('boi,bid->bod', adj, gcn_input),self.context_weight))
        #w = F.tanh(torch.einsum("bod,ol->bdl",gcn_input,self.context_weight))
        gcn_input = self.linear(torch.einsum("bdl,bod->bol",self.dropout(w),gcn_input) + gcn_input)
        # np.save("tra_s_id.npy",static_id_embeds.cpu().detach().numpy())
        # np.save("tra_d_id.npy",dynamic_id_embeds[0].cpu().detach().numpy())
        # np.save("tra_cluster_id.npy",cluster_centers.cpu().detach().numpy())
        return gcn_input.unsqueeze(dim=-2) #(b,c,n,t)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.temporal_encoder_in = nn.ModuleList([gated_mlp(seq_in = self.seq_len, seq_out = self.seq_len, 
                                              d_model = configs.d_model, channels = configs.enc_in) #configs.d_model configs.enc_in
                                              for i in range(configs.layers)])
        self.GNN_encoder_in = nn.ModuleList([MIDGCN(configs=configs,out_len=self.seq_len)
                                          for i in range(configs.layers)])
        self.fconv_in = FreqConv(4, self.seq_len, self.seq_len)
        self.fc_idp = Indepent_Linear(self.seq_len, self.pred_len, configs.enc_in)
        self.temporal_encoder_out = nn.ModuleList([gated_mlp(seq_in = self.pred_len, seq_out = self.pred_len, 
                                              d_model = configs.d_model, channels = configs.enc_in) #configs.d_model configs.enc_in
                                              for i in range(1)])
        self.GNN_encoder_out = nn.ModuleList([MIDGCN(configs=configs,out_len=self.pred_len)
                                          for i in range(1)])
        # self.fconv_in = fconv(c_in=configs.enc_in,inp_len=self.seq_len)
        # self.fconv_out = fconv(c_in=configs.enc_in,inp_len=self.pred_len)
        # self.conv_in = nn.Conv2d(in_channels=configs.enc_in,out_channels=configs.enc_in,kernel_size=3,padding=1)
        # self.conv_out = nn.Conv2d(in_channels=configs.enc_in,out_channels=configs.enc_in,kernel_size=3,padding=1)
        self.fconv_out = FreqConv(4, self.pred_len, self.pred_len)
        if configs.use_revin:
            self.revin = RevIN(num_channels=configs.enc_in,num_nodes=1)
        self.use_revin = configs.use_revin
        self.use_last = configs.use_last
    
    def forecast(self,x):#(b,t,1,c)
        # t_dim, c_dim = x.shape[1], x.shape[2]
        # indices_c = torch.randperm(c_dim)
        # x = x[:,:,indices_c]

        x = x.unsqueeze(dim=-2).permute(0,3,2,1)
        if self.use_revin:
            x = self.revin.forward(x)
        if self.use_last:
            last_seq = x[:,:,:,-1].unsqueeze(dim=-1)
            x = x - last_seq
        for (mlp,gnn) in zip(self.temporal_encoder_in,self.GNN_encoder_in):
            x_1 = mlp(x)
            x_2 = gnn(x_1)
            # x_2 = x_1
        x_2 = self.fconv_in(x,x_2)
        # x_2 = self.conv_in(x_2)
        y = self.fc_idp(x_2)
        for (mlp,gnn) in zip(self.temporal_encoder_out,self.GNN_encoder_out):
            y_1 = mlp(y)
            y_2 = gnn(y_1)
            # y_2 = y_1
        # y = self.conv_out(y_2)
        y = self.fconv_out(y,y_2)
        # y = y_2
        if self.use_last:
            y = y + last_seq
        if self.use_revin:
            y = self.revin.reverse(y)
        y = y.squeeze(dim=-2).permute(0,2,1)
        return y

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out#,loss

        # x = x.unsqueeze(dim=-2).permute(0,3,2,1)
        # # np.save("tra_input_x.npy",x[0,:,0,:].cpu().detach().numpy())
        # if self.use_revin:
        #     x = self.revin.forward(x)
        # if self.use_last:
        #     last_seq = x[:,:,:,-1].unsqueeze(dim=-1)
        #     x = x - last_seq
        # x = self.fconv_in(x,x)
        # for (mlp,gnn) in zip(self.temporal_encoder_in,self.GNN_encoder_in):
        #     x_1 = mlp(x)
        #     x_2 = gnn(x_1)
        #     # x_2 = x_1
        # #x_2 = self.fconv_in(x,x_2)
        # # x_2 = self.conv_in(x,x_2)

        # y = self.fc_idp(x_2)
        # y = self.fconv_out(y,y)
        # for (mlp,gnn) in zip(self.temporal_encoder_out,self.GNN_encoder_out):
        #     y_1 = mlp(y)
        #     y_2 = gnn(y_1)
        #     # y_2 = y_1
        #     y = y_2
        # #y = self.fconv_out(y,y_2)
        # # y = y_2
        # # y = self.conv_out(y,y_2)
        # if self.use_last:
        #     y = y + last_seq
        # if self.use_revin:
        #     y = self.revin.reverse(y)
        # y = y.squeeze(dim=-2).permute(0,2,1)