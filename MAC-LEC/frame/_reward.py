
import torch

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class Reward(object):
    """
    Used for calculate reward for ordering-based Causal discovery

    In ordering-based methods, only the variables selected in previous decision
    steps can be the potential parents of the currently selected variable.
    Hence, author design the rewards in the following cases:
    `episodic reward` and `dense reward`.

    """

    def __init__(self, input_data, reward_mode='episodic',
                 score_type='BIC', regression_type='LR', alpha=1.0):


        self.input_data = input_data
        self.reward_mode = reward_mode
        self.score_type = score_type
        self.regression_type = regression_type

    def calc_reward(self, X: torch.Tensor, graph: torch.Tensor,
                    lambda_sparse=1e-2, lambda_sym=0.15) -> torch.Tensor:
        """
        计算 reward = -loss + 稀疏奖励 + 对称惩罚
        """
        num_nodes = X.shape[1]
        device = X.device
        total_loss = 0.0
        count = 0

        for i in range(num_nodes):
            parents = (graph[:, i] != 0)
            parents = parents.squeeze()
            if parents.ndim == 0:
                parents = parents.unsqueeze(0)
            if torch.sum(parents) == 0:
                continue

            X_parents = X[:, parents]  # [num_samples, num_parents]
            y = X[:, i]  # [num_samples]

            XTX = X_parents.T @ X_parents
            XTy = X_parents.T @ y
            eye = torch.eye(XTX.size(0), device=device, dtype=XTX.dtype)
            w = torch.linalg.solve(XTX + 1e-3 * eye, XTy)
            y_pred = X_parents @ w
            loss = torch.mean((y - y_pred) ** 2)
            total_loss += loss
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)

        fit_reward = -total_loss / count
        fit_reward = torch.tanh(fit_reward / 5.0)

        sparse_reward = -lambda_sparse * graph.sum()

        # ✅ 对称性惩罚项
        symmetry_penalty = torch.sum(torch.abs(graph * graph.T))
        symmetry_reward = -lambda_sym * symmetry_penalty

        return fit_reward + sparse_reward + symmetry_reward

    def cal_rewards(self, graphs, data=None, gamma=0.98):
        rewards_batches = []
        i = 0
        best_local_graph = None
        max_reward_batch = -float('inf')
        for graphi in graphs:
            reward_ = self.cal_reward_single(graphi,data)
            i += 1
            rewards_batches.append(reward_)
            normalized_reward = -reward_
            if normalized_reward > max_reward_batch:
                max_reward_batch = normalized_reward
                best_local_graph = graphi

        return best_local_graph, max_reward_batch

    def cal_reward_single(self, graph: torch.Tensor, data=None):
        """
        基于线性回归和BIC，附加稀疏正则，鼓励稀疏结构
        """
        X=self.input_data
        X=X.float()
        num_samples, n_nodes_data = X.shape
        n_nodes = graph.shape[0]
        device = X.device

        if n_nodes > n_nodes_data:
            graph = graph[:n_nodes_data, :n_nodes_data]
            n_nodes = n_nodes_data

        total_score = 0.0
        count = 0
        total_edges = 0  # 统计边数

        for target in range(n_nodes):
            parents = (graph[:, target] != 0).nonzero(as_tuple=False).squeeze()
            if parents.numel() == 0:
                continue
            if parents.dim() == 0:
                parents = parents.unsqueeze(0)
            if (parents >= n_nodes_data).any() or target >= n_nodes_data:
                continue
            parents = parents.to(device)

            X_parents = X[:, parents]
            y_target = X[:, target]
            # 强制类型一致（确保都不是 LazyTensor）
            X_parents = X_parents.to(dtype=torch.float32)
            y_target = y_target.to(dtype=torch.float32)

            XTX = X_parents.T @ X_parents
            XTy = X_parents.T @ y_target
            # 确保类型、设备一致，并强制 materialize 为普通张量
            XTX = XTX.contiguous().clone()
            XTy = XTy.contiguous().clone()

            eye = torch.eye(XTX.shape[0], dtype=XTX.dtype, device=XTX.device)
            reg_eye = 1e-6 * eye

            XTX_reg = XTX + reg_eye

            # 也要 materialize
            XTX_reg = XTX_reg.contiguous().clone()
            XTy = XTy.contiguous().clone()

            w = torch.linalg.solve(XTX_reg, XTy)
            y_pred = X_parents @ w

            n = y_target.shape[0]
            k = X_parents.shape[1]
            mse = torch.mean((y_target - y_pred) ** 2)
            bic = n * torch.log(mse) + k * torch.log(torch.tensor(n + 1e-6, device=device))

            score = -bic
            total_score += score
            count += 1
            total_edges += k

        if count == 0:
            return torch.tensor(0.0, device=device)

        final_score = total_score / count
        return final_score-0.04*total_edges


import torch


class Reward1(torch.nn.Module):
    def __init__(self, input_data, reward_mode='episodic',
                 score_type='BIC', regression_type='LR', alpha=1.0, device='cuda:1'):
        """
        参数：
          - input_data: 形状为 (n_samples, seq_length) 的 torch.Tensor，
                        请预先将数据放在目标设备上（如 GPU）
          - reward_mode, score_type, regression_type, alpha 等参数控制计算细节
          - device: 使用的设备 ('cuda' 或 'cpu')
        """
        super(Reward1, self).__init__()
        self.device = input_data.device
        self.input_data = input_data  # 保证数据已在目标设备上
        self.reward_type = reward_mode
        self.alpha = alpha
        self.n_samples, self.seq_length = input_data.shape

        # 缓存字典（图结构对应的分数及各时刻奖励）
        self.d = {}
        # 每个时刻的 RSS 缓存（列表中每个元素为一个字典）
        self.d_RSS = [{} for _ in range(self.seq_length)]
        # BIC 惩罚项（此处为常数）
        self.bic_penalty = torch.log(
            torch.tensor(self.n_samples, dtype=input_data.dtype, device=self.device)) / self.n_samples

        self.score_type = score_type
        self.regression_type = regression_type

        # 构造带有常数项的设计矩阵
        self.ones = torch.ones((self.n_samples, 1), dtype=input_data.dtype, device=self.device)
        self.X = torch.cat((self.input_data, self.ones), dim=1)
        self.XtX = self.X.T @ self.X

    def cal_rewards(self, graphs, positions=None, true_flag=False, gamma=0.98):
        """
        计算多个图的奖励、归一化奖励、最大奖励以及 TD-target（折扣累计奖励）

        参数：
          - graphs: 图的列表，每个图为描述图结构的张量（例如 shape=(seq_length, )）
          - positions: 当 true_flag 为 False 时，用于从每个时刻的 RSS 中选取部分分量（可为索引、切片或列表）
          - true_flag: 若 True 则忽略缓存
          - gamma: 折扣因子
        返回：
          - reward_list_tensor: 形状 (batch_size, T) 的奖励序列（已取负，与原代码一致）
          - normal_batch_rewards: 每个图对应的标量归一化奖励（BIC 的负值）
          - max_reward_batch: 归一化奖励中的最大值（标量）
          - td_target: 按时间步计算的折扣累计奖励，形状 (T, batch_size)
        """
        rewards_batches = []
        if not true_flag:
            for graph, pos in zip(graphs, positions):
                reward_val, reward_list = self.calculate_reward_single_graph(
                    graph, position=pos, true_flag=False
                ) #reward_val为BIC用来衡量
                rewards_batches.append((reward_val, reward_list))
        else:
            for graph in graphs:
                reward_val, reward_list = self.calculate_reward_single_graph(
                    graph, true_flag=True
                )
                rewards_batches.append((reward_val, reward_list))

        reward_vals = []
        reward_lists = []
        normal_batch_rewards = []
        max_reward_batch = -float('inf')
        for reward_val, reward_list in rewards_batches:
            reward_lists.append(reward_list)
            normalized_reward = reward_val  # 与原代码保持一致
            normal_batch_rewards.append(normalized_reward)
            if normalized_reward > max_reward_batch:
                max_reward_batch = normalized_reward
            reward_vals.append(reward_val)
        # 形状为 (batch_size, T)
        reward_list_tensor = torch.stack(reward_lists)
        # 原代码中对 reward_list 取负，这里保持一致
        reward_list_tensor = -reward_list_tensor

        # 将标量奖励转换为张量（注意：这里 reward_val 原本就为标量张量）
        normal_batch_rewards = torch.tensor(normal_batch_rewards, dtype=self.input_data.dtype, device=self.device)
        reward_vals_tensor = torch.tensor(reward_vals, dtype=self.input_data.dtype, device=self.device)

        # 计算折扣累计奖励（TD-target）
        # 原代码先转置 (batch_size, T) -> (T, batch_size)，然后逆序累加
        td_target = self.discount_cumsum(reward_list_tensor.transpose(0, 1), gamma)
        return  normal_batch_rewards, max_reward_batch, td_target

    def discount_cumsum(self, rewards, gamma):
        """
        利用向量化方式计算折扣累计和
        参数：
          - rewards: 张量，形状 (T, batch_size)
          - gamma: 折扣因子
        返回：
          - discounted: 张量，形状 (T, batch_size)
        计算公式：discounted[t] = rewards[t] + gamma * rewards[t+1] + gamma^2 * rewards[t+2] + ...
        """
        T, B = rewards.shape
        # 构造折扣因子序列：形状 (T, 1)
        discount_factors = gamma ** torch.arange(T, dtype=rewards.dtype, device=rewards.device).unsqueeze(1)
        # 反转后乘上折扣因子、做累加、再反转回来
        discounted = torch.flip(
            torch.cumsum(torch.flip(rewards * discount_factors, dims=[0]), dim=0) / discount_factors,
            dims=[0]
        )
        return discounted


    def calculate_reward_single_graph(self, graph_batch, position=None, true_flag=False):
        """
        计算单个图对应的整体分数（BIC）和各时刻奖励    BIC贝叶斯信息准则，用于衡量模型拟合优度，帮助评估模型在数据上的表现，约小越好
        参数：
          - graph_batch: 表示图结构的张量，形状为 (seq_length, )
          - position: 用于选取 RSS 中的部分分量（例如一个索引、列表或切片），若为 None 则表示全部时刻
          - true_flag: 若 True 则不使用缓存
        返回：
          - BIC: 标量张量（BIC 分数）
          - reward_list: 张量，形状取决于 position 的选取
        """
        if position is not None:
            # 将 position 转换为 tuple 作为缓存键
            pos_tuple = self.convert_position_to_tuple(position)
        else:
            pos_tuple = None

        if not true_flag and pos_tuple is not None and pos_tuple in self.d:
            BIC, cached_reward_list = self.d[pos_tuple]
            return BIC, cached_reward_list

        # 批量计算每个时刻的 RSS
        graph_batch_tensor = graph_batch  # 转为 Tensor
        RSS_ls = self.batch_cal_RSSi(graph_batch_tensor)

        if position is None:
            pos_sel = slice(None)
        else:
            pos_sel = position
            # 如果是 tensor，则移动到相同 device
            if isinstance(pos_sel, torch.Tensor):
                pos_sel = pos_sel.to(RSS_ls.device)

        reward_list = RSS_ls[pos_sel] / self.n_samples

        # 计算 BIC 分数，1e-8 避免对 0 取对数
        BIC = torch.log(torch.sum(RSS_ls) / self.n_samples + 1e-8)
        # lambda_sparse=0.4
        # num_edges = torch.sum(graph_batch_tensor)
        # max_edges = graph_batch_tensor.shape[0] * (graph_batch_tensor.shape[1] - 1)  # 去掉自环
        # sparsity_penalty = lambda_sparse * (num_edges / max_edges)  # 比例惩罚

        # 把惩罚加到 BIC 上
        BIC = BIC

        if not true_flag and pos_tuple is not None:
            self.d[pos_tuple] = (BIC, reward_list)
        return -BIC, reward_list

    def convert_position_to_tuple(self, position):
        """
        将 position 转换为 tuple 格式，兼容 Tensor 和 Python 列表/元组
        """
        if isinstance(position, torch.Tensor):
            pos_list = position.tolist()
            return tuple(int(x) for x in pos_list)
        elif isinstance(position, (list, tuple)):
            return tuple(int(x) for x in position)
        else:
            return (int(position),)

    def batch_cal_RSSi(self, graph_batch_tensor):
        """
        批量计算 RSS，可以充分利用 GPU 并行加速
        """
        # 在批量上计算每个时刻的 RSS
        y_err = graph_batch_tensor - torch.mean(graph_batch_tensor, dim=0, keepdim=True)
        RSS = torch.sum(y_err ** 2, dim=0)
        return RSS




