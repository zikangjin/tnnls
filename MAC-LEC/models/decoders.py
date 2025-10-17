import torch
import torch.nn as nn
import torch.distributions as distr
import torch.nn.functional as F
from torch.distributions import Categorical
# from _base_network import PointerDecoder
from ._base_network import PointerDecoder
import numpy as np
class OuterProductDecoder(nn.Module):
	"""
       改造后的外积解码器：逐步解码动作，同时保留中间状态
       输出：
           actions: (B, T)
           mask_scores: (B, T, R)
           s_list, h_list, c_list: 用于 TD/critic/log_softmax 的解码状态轨迹
       """
	def __init__(self, hidden_dim, num_regions, seq_length):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.num_regions = num_regions
		self.seq_length = seq_length

		# 初始化每个位置的起始状态
		self.query_proj = nn.Linear(hidden_dim, hidden_dim)
		self.key_proj = nn.Linear(hidden_dim, hidden_dim)
		self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
		self.action_proj = nn.Linear(hidden_dim, num_regions)
		self.context_proj = nn.Linear(hidden_dim * hidden_dim, hidden_dim)

	def forward(self, h):
		batch_size = h.size(0)
		device = h.device

		hidden = h  # 初始解码隐藏状态
		query = self.query_proj(h)
		key = self.key_proj(h)

		actions = []
		mask_scores = []
		s_list, h_list, c_list = [], [], []

		used = torch.zeros(batch_size, self.num_regions, device=device)

		for t in range(self.seq_length):
			context = torch.bmm(query.unsqueeze(2), key.unsqueeze(1))  # [B, H, H]
			context_flat = context.view(batch_size, -1)
			context_embed = self.context_proj(context_flat)  # [B, hidden_dim] = [B, 256]

			# 当前步骤状态
			s_list.append(context_embed)
			h_list.append(hidden)
			c_list.append(hidden)

			# 用于决策的 logits
			logits = self.action_proj(hidden)  # [B, R]
			logits = logits.masked_fill(used.bool(), float('-inf'))  # Mask已选

			probs = F.softmax(logits, dim=-1)
			action = torch.argmax(probs, dim=-1)

			actions.append(action)
			mask_scores.append(logits)

			# 更新mask和GRU状态
			used = used.scatter(1, action.unsqueeze(1), 1.0)
			hidden = self.gru_cell(key, hidden)

		# Stack输出
		actions = torch.stack(actions, dim=1)  # [B, T]
		mask_scores = torch.stack(mask_scores, dim=1)  # [B, T, R]
		s_list = torch.stack(s_list, dim=1)  # [B, T, D]
		h_list = torch.stack(h_list, dim=1)  # [B, T, D]
		c_list = torch.stack(c_list, dim=1)  # [B, T, D]

		return actions, mask_scores, s_list, h_list, c_list

	def log_softmax(self, input, position, mask, state_0=None, state_1=None):
		"""
        计算 OuterProductDecoder 每一步选中 action 的 log 概率（和旧 decoder 对齐）

        input:      [B*T, D] 每一步的 decoder 输入
        position:   [B*T]，每一步选中的 action（index）
        mask:       [B*T, R]，每一步的 action mask（0 表示可选，1 表示 mask）

        """
		# print("input",input.shape)
		position = position.reshape(-1)  # [B*T]
		logits = self.action_proj(input)  # [B*T, R]
		# print("logits",logits.shape)
		if isinstance(mask, np.ndarray):
			mask = torch.tensor(mask, dtype=torch.bool, device=input.device)
		else:
			mask = mask.to(dtype=torch.bool, device=input.device)
		# print("mask",mask.shape)

		# logits = logits.masked_fill(mask, float('-inf'))  # 屏蔽非法 action

		# 得到 log_softmax 分布
		prob_dist = Categorical(logits=logits)
		log_prob = prob_dist.log_prob(position)  # [B*T]

		return log_prob




class GRUDecoder(nn.Module):
	"""GRU解码器：生成脑区连接矩阵"""

	def __init__(self, hidden_dim, num_regions):
		super().__init__()
		self.num_regions = num_regions
		# 动态生成连接矩阵的GRU单元
		self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
		# 输出层（对称矩阵约束）
		self.fc = nn.Linear(hidden_dim, num_regions * num_regions)

	def forward(self, h):
		# 初始化解码状态
		batch_size = h.size(0)
		h_dec = h.clone()  # (batch, hidden_dim)

		# 通过GRU迭代细化连接矩阵（可扩展为多步解码）
		for _ in range(2):  # 2次迭代足够生成稳定输出
			h_dec = self.gru_cell(h, h_dec)

		# 生成对称连接矩阵
		adj_flat = self.fc(h_dec)  # (batch, num_regions*num_regions)
		adj = adj_flat.view(batch_size, self.num_regions, self.num_regions)
		adj = (adj + adj.transpose(1, 2)) / 2  # 强制对称
		return torch.sigmoid(adj),None,  # 归一化到[0,1]




class LSTMDecoder(PointerDecoder):
	"""LSTM + Pointer Network"""

	def __init__(self, input_dim, hidden_dim, device=None) -> None:
		# input of Decoder is output of Encoder, e.g. embed_dim
		super(LSTMDecoder, self).__init__(input_dim=input_dim,
										  hidden_dim=hidden_dim,
										  device=device)
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.device = device
		self.lstm_cell = nn.LSTMCell(input_size=hidden_dim,
									 hidden_size=hidden_dim,
									 device=self.device)

	def forward(self, x) -> tuple:
		""""""
		self.batch_size = x.shape[0]
		self.seq_length = x.shape[1]
		self.encoder_output = x  # 保存起来有用

		s_i = torch.mean(x, 1)
		hi_ci = (torch.zeros((self.batch_size, self.hidden_dim), device=s_i.device).float(),
				 torch.zeros((self.batch_size, self.hidden_dim), device=s_i.device).float())
		h_list = []
		c_list = []
		s_list = []
		action_list = []
		prob_list = []
		for step in range(self.seq_length):
			h_list.append(hi_ci[0])
			c_list.append(hi_ci[1])
			s_list.append(s_i)

			s_i, hi_ci, pos, prob = self.step_decode(input=s_i, state=hi_ci)

			action_list.append(pos)
			prob_list.append(prob)

		h_list = torch.stack(h_list, dim=1).squeeze()  # [Batch,seq_length,hidden]
		c_list = torch.stack(c_list, dim=1).squeeze()  # [Batch,seq_length,hidden]
		s_list = torch.stack(s_list, dim=1).squeeze()  # [Batch,seq_length,hidden]

		# Stack visited indices
		actions = torch.stack(action_list, dim=1)  # [Batch,seq_length]
		mask_scores = torch.stack(prob_list, dim=1)  # [Batch,seq_length,seq_length]
		self.mask = torch.zeros(1, device=self.device)

		return actions, mask_scores, s_list, h_list, c_list


class MLPDecoder(PointerDecoder):
	"""Multi Layer Perceptions + Pointer Network"""

	def __init__(self, input_dim, hidden_dim, device=None) -> None:
		super(MLPDecoder, self).__init__(input_dim=input_dim,
										  hidden_dim=hidden_dim,
										 device=device)
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.device = device
		self.mlp = self.feedforward_mlp

	def forward(self, x) -> tuple:

		self.batch_size = x.shape[0]
		self.seq_length = x.shape[1]
		self.encoder_output = x

		s_i = torch.mean(x, 1)

		s_list = []
		action_list = []
		prob_list = []
		for step in range(self.seq_length):
			s_list.append(s_i)
			s_i, _, pos, prob = self.step_decode(input=s_i, state=None)

			action_list.append(pos)
			prob_list.append(prob)
		s_list = torch.stack(s_list, dim=1).squeeze()  # [Batch,seq_length,hidden]

		# Stack visited indices
		actions = torch.stack(action_list, dim=1)  # [Batch,seq_length]
		mask_scores = torch.stack(prob_list, dim=1)  # [Batch,seq_length,seq_length]
		self.mask = torch.zeros(1, device=self.device)

		return actions, mask_scores, s_list, s_list, s_list

class BiLSTM_Attention(PointerDecoder):

	def __init__(self, input_dim, hidden_dim, device=None) -> None:
		super(BiLSTM_Attention, self).__init__(input_dim=input_dim,
										  hidden_dim=hidden_dim,
										 device=device)

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.device = device
		self.lstm = nn.LSTM(
			input_size=self.input_dim,
			hidden_size=self.hidden_dim,
			num_layers=2,
			batch_first=True,
			bidirectional=True,
			device=self.device
		)
		self.out = nn.Linear(self.hidden_dim * 2, self.input_dim, device=self.device)
		self.emb = nn.Parameter(
			torch.zeros(*(1, 50, self.hidden_dim), device=self.device))   #10改成self.seq_length
		self.bn_layer = nn.BatchNorm1d(self.hidden_dim, device=self.device)
		self.w_o = nn.Parameter(
			torch.zeros(self.hidden_dim * 2, self.hidden_dim * 2,device=self.device))
		self.u_o = nn.Parameter(torch.zeros(self.hidden_dim * 2, 1,device=self.device))
		self.fc = nn.Linear(self.hidden_dim * 2, self.input_dim,device=self.device)
		self.conv1 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.input_dim, kernel_size=1,
							   bias=True,device=self.device)

		self.Q_layer = nn.Sequential(nn.Linear(in_features=self.hidden_dim * 2, out_features=self.hidden_dim * 2,device=self.device),
									 nn.ReLU())
		self.K_layer = nn.Sequential(nn.Linear(in_features=self.hidden_dim * 2, out_features=self.hidden_dim * 2,device=self.device),
									 nn.ReLU())
		self.V_layer = nn.Sequential(nn.Linear(in_features=self.hidden_dim * 2, out_features=self.hidden_dim * 2,device=self.device),
									 nn.ReLU())

		self.relu = nn.ReLU()

		nn.init.uniform_(self.w_o, -0.1, 0.1)
		nn.init.uniform_(self.u_o, -0.1, 0.1)
		nn.init.xavier_uniform_(self.emb)

	def att(self, inputs, dropout_rate=0.1):
		input_dimension = inputs.shape[2]
		# inputs = inputs.permute(0, 2, 1)

		Q = self.Q_layer(inputs)  # [batch_size, seq_length, n_hidden]
		K = self.K_layer(inputs)  # [batch_size, seq_length, n_hidden]
		V = self.V_layer(inputs)  # [batch_size, seq_length, n_hidden]

		# Split and concat
		Q_ = torch.cat(torch.split(Q, int(input_dimension), dim=2),
					   dim=0)  # [batch_size, seq_length, n_hidden/num_heads]
		K_ = torch.cat(torch.split(K, int(input_dimension), dim=2),
					   dim=0)  # [batch_size, seq_length, n_hidden/num_heads]
		V_ = torch.cat(torch.split(V, int(input_dimension), dim=2),
					   dim=0)  # [batch_size, seq_length, n_hidden/num_heads]

		# Multiplication
		outputs = torch.matmul(Q_, K_.permute([0, 2, 1]))  # num_heads*[batch_size, seq_length, seq_length]

		# Scale
		outputs = outputs / (K_.shape[-1] ** 0.5)

		# Activation
		outputs = F.softmax(outputs,dim=1)  # num_heads*[batch_size, seq_length, seq_length]

		# Dropouts
		outputs = F.dropout(outputs, p=dropout_rate, training=True)

		# Weighted sum
		outputs = torch.matmul(outputs, V_)  # num_heads*[batch_size, seq_length, n_hidden/num_heads]

		# Restore shape
		outputs = torch.cat(torch.split(outputs, int(outputs.shape[0]), dim=0),
							dim=2)  # [batch_size, seq_length, n_hidden]

		# Residual connection
		outputs = outputs + inputs  # [batch_size, seq_length, n_hidden]

		# Normalize
		# outputs = self.bn_layer(outputs)  # [batch_size, seq_length, n_hidden]

		return outputs

	def forward(self, inputs):  # inputs[64, 10, 256]

		self.batch_size = inputs.shape[0]
		self.seq_length = inputs.shape[1]
		self.encoder_output = inputs

		# Embed input sequence
		W_embed = self.emb
		W_embed_ = W_embed.permute(2, 1, 0)
		self.embedded_input = F.conv1d(inputs, W_embed_, stride=1)

		# Batch Normalization
		self.enc = self.bn_layer(self.embedded_input)

		# final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
		# output : [seq_len, batch_size, n_hidden * num_directions(=2)]
		# input = self.enc.transpose(1, 2)
		output, (final_hidden_state, final_cell_state) = self.lstm(inputs)
		out = self.att(output)
		out = self.fc(out)
		out = self.relu(out)   # [64, 10, 256]

		# adj_prob = out
		# self.adj_prob = adj_prob.permute(0, 2, 1)

		action_list = []
		prob_list = []
		for step in range(self.seq_length):
			s_i = out[ : , step, : ]  #  decoder输出
			_, _, pos, prob = self.step_decode(input=s_i, state=None)
			action_list.append(pos)
			prob_list.append(prob)

		# Stack visited indices
		actions = torch.stack(action_list, dim=1)  # [Batch,seq_length]
		mask_scores = torch.stack(prob_list, dim=1)  # [Batch,seq_length,seq_length]
		self.mask = torch.zeros(1, device=self.device)

		return actions, mask_scores, self.encoder_output, out, out

		#。

		# self.mask = 0
		# self.samples = []
		# self.mask_scores = []
		# self.entropy = []
		#
		# # inputs = inputs.permute(0, 2, 1)
		#
		# for i in range(self.seq_length):
		# 	position = torch.ones([inputs.shape[0]]) * i
		# 	position = position.type(torch.LongTensor)
		# 	# if self.config.device_type == 'gpu':
		# 	#     position = position.cuda(self.config.device_ids)
		# 	# Update mask
		# 	self.mask = torch.zeros(inputs.shape[0], self.seq_length).scatter_(1, position.view(
		# 		inputs.shape[0], 1), 1)
		# 	# self.mask = self.mask + F.one_hot(action, self.seq_length)
		# 	# if self.config.device_type == 'gpu':
		# 	# 	self.mask = self.mask.cuda(self.device_ids)
		# 	self.mask = self.mask.cuda(self.device_ids)
		# 	masked_score = self.adj_prob[:, i, :] - 100000000. * self.mask
		# 	prob = distr.Bernoulli(logits=masked_score)  # probs input probability, logit input log_probability
		#
		# 	sampled_arr = prob.sample()  # Batch_size, seqlenght for just one node
		# 	sampled_arr.requires_grad = True
		# 	#
		# 	# self.samples.append(sampled_arr)
		# 	# self.mask_scores.append(masked_score)
		# 	# self.entropy.append(prob.entropy())
		# 	self.samples.append(sampled_arr.cpu())
		# 	self.mask_scores.append(masked_score.cpu())
		# 	self.entropy.append(prob.entropy().cpu())
		#
		# return self.samples, self.mask_scores, self.entropy



if __name__ == '__main__':
	import numpy as np
	X = torch.Tensor(64, 10, 256).to(device=0)
	# X = torch.Tensor(64, 10, 256)
	# decoder = LSTMDecoder(256,256)
	decoder = BiLSTM_Attention(256,256,device=0)
	# decoder = Bi_LSTMDecoder(256,256)
	actions, mask_scores, encoder_output, _, out = decoder(X)
	print(actions.shape)
	print(mask_scores.shape)
	print(encoder_output.shape)
	print(out.shape)
