# coding=utf-8
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from torch.nn.functional import pad

from infomap import Infomap
import os
import logging
import platform
import random
import time
from castle.algorithms.pc.pc import find_skeleton
import networkx as nx
import torch.nn.functional as F
from frame import score_function as Score_Func
import scipy.io as sio
from common import BaseLearner, consts, GraphDAG
from tqdm import tqdm
import numpy as np
import torch
from common import BaseLearner, consts
from frame import Actor, EpisodicCritic, Reward, DenseCritic, Meta_Critic, Hot_Plug
from frame._critic import  EpisodicCritic, CriticSmall, CriticMedium, CriticLarge, EpisodicCritic1
from frame._reward import Reward1
from utils.data_loader import DataGenerator
from utils.graph_analysis import get_graph_from_order, pruning_by_coef
from utils.graph_analysis import pruning_by_coef_2nd
from common.validator import check_args_value
from metrics import MetricsDAG
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass


def get_adaptive_critic(node_count, embed_dim, device=None):
    if node_count < 10:
        return CriticSmall(embed_dim, device=device)
    elif node_count <= 20:
        return CriticMedium(embed_dim, device=device)
    else:
        return CriticLarge(embed_dim, device=device)
def get_adaptive_actor(node_count,input_dim,embed_dim,max_length,device):
    if node_count < 10:
        return Actor(
            input_dim=input_dim,
            embed_dim=embed_dim,
            encoder_name='lstm',
            decoder_name='mlp',
            max_length=max_length,
            device=device
        ).to(dtype=torch.float32, device=device)
    elif node_count <= 20:
        return Actor(
            input_dim=input_dim,
            embed_dim=embed_dim,
            encoder_name='temporal ',
            decoder_name='lstm',
            max_length=max_length,
            device=device
        ).to(dtype=torch.float32, device=device)
    else:
        return Actor(
            input_dim=input_dim,
            embed_dim=embed_dim,
            encoder_name='transformer',
            decoder_name='bilstm',
            max_length=max_length,
            device=device
        ).to(dtype=torch.float32, device=device)

class CommunityActorWrapper:

    def __init__(self, input_dim, embed_dim,encoder_name, decoder_name,
                 max_length, device, actor_lr,critic_lr):
        self.actor = Actor(
            input_dim=input_dim,
            embed_dim=embed_dim,
            encoder_name=encoder_name,
            decoder_name=decoder_name,
            max_length=max_length,
            device=device
        ).to(dtype=torch.float32, device=device)

        self.hotplug = Hot_Plug(self.actor.encoder)
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.encoder.parameters(), 'lr': actor_lr},
            {'params': self.actor.decoder.parameters(), 'lr': actor_lr}
        ])

        self.critic = EpisodicCritic1(input_dim=embed_dim,
                                device=device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.community_id = None
        self.node_indices = []
        self.comm_size = 0




class MetaRLEC(BaseLearner):

    @check_args_value(consts.MetaRLEC_VALID_PARAMS)
    def __init__(self, batch_size=128, input_dim=64, embed_dim=128,
                 normalize=False,
                 encoder_name='transformer',
                 encoder_heads=4,
                 encoder_blocks=2,
                 encoder_dropout_rate=0.1,
                 decoder_name='lstm',
                 reward_mode='episodic',
                 reward_score_type='BIC',
                 reward_regression_type='LR',
                 reward_gpr_alpha=1.0,
                 iteration=10,
                 lambda_iter_num=20,
                 actor_lr=1e-3,
                 critic_lr=1e-4,
                 alpha=0.99,  # for score function
                 init_baseline=-1.0,
                 random_seed=0,
                 device_type='gpu',
                 device_ids=0
                 ):
        super(MetaRLEC, self).__init__()
        self.batch_size             = batch_size
        self.input_dim              = input_dim
        self.embed_dim              = embed_dim
        self.normalize              = normalize
        self.encoder_name           = encoder_name
        self.encoder_heads          = encoder_heads
        self.encoder_blocks         = encoder_blocks
        self.encoder_dropout_rate   = encoder_dropout_rate
        self.decoder_name           = decoder_name
        self.reward_mode            = reward_mode
        self.reward_score_type      = reward_score_type
        self.reward_regression_type = reward_regression_type
        self.reward_gpr_alpha       = reward_gpr_alpha
        self.iteration              = iteration
        self.lambda_iter_num        = lambda_iter_num
        self.actor_lr               = actor_lr
        self.critic_lr              = critic_lr
        self.alpha                  = alpha
        self.init_baseline          = init_baseline
        self.random_seed            = random_seed
        self.device_type            = device_type
        self.device_ids             = device_ids
        self.reward_history = []
        self.graph_diff_history = []
        self.sparsity_history = []
        self.best_reward = float('-inf')
        self.best_graph = None
        self.last_graph = None
        self.all_actor_losses = []
        self.gamma=0.99

        if reward_mode == 'dense':
            self.avg_baseline = torch.tensor(init_baseline, requires_grad=False)

        if torch.cuda.is_available():
            logging.info('GPU is available.')
        else:
            logging.info('GPU is unavailable.')
            if self.device_type == 'gpu':
                raise ValueError("GPU is unavailable, "
                                 "please set device_type = 'cpu'.")
        if self.device_type == 'gpu':
            if self.device_ids:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_ids)
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        self.device = device
        print("device:", self.device)


    def learn(self, data, clusters_level2=None, **kwargs) -> None:
        X = torch.tensor(data).to(self.device)
        self.n_samples = X.shape[0]
        self.seq_length = X.shape[1] # seq_length == n_nodes

        self.dag_mask = getattr(kwargs, 'dag_mask', None)
        causal_matrix,graph = self._rl_search(clusters_level2,X)
        self.causal_matrix = causal_matrix
        self.graph=graph

    def _rl_search(self, clusters_level2,X) -> torch.Tensor:

        set_seed(self.random_seed)
        logging.info('Python version is {}'.format(platform.python_version()))
        num_communities = len(clusters_level2)
        print('Number of communities:', num_communities)
        community_actors = {}
        for comm_id,node_indices in clusters_level2.items():
            comm_size=len(node_indices)
            actor_wrapper=CommunityActorWrapper(input_dim=self.input_dim,
                                                embed_dim=self.embed_dim,
                                                encoder_name=self.encoder_name,
                                                decoder_name=self.decoder_name,
                                                max_length=comm_size,
                                                device=self.device,
                                                actor_lr=self.actor_lr,
                                                critic_lr=self.critic_lr)
            actor_wrapper.community_id=comm_id
            actor_wrapper.node_indices=node_indices
            community_actors[comm_id]=actor_wrapper
            actor_wrapper.comm_size=comm_size

        # generate observed data
        data_generactors = {}
        for comm_id, wrapper in community_actors.items():
            comm_data = X[:, wrapper.node_indices]
            # print('comm_data:',comm_data.shape)
            data_generactors[comm_id] = DataGenerator(dataset=comm_data,
                                                      normalize=self.normalize,
                                                      device=self.device)


        # Instantiating an Critic


        reward=Reward(input_data=X,
                       reward_mode=self.reward_mode,
                       score_type=self.reward_score_type,
                       regression_type=self.reward_regression_type,
                       alpha=self.reward_gpr_alpha)
        # Instantiating an Reward
        reward1 =Reward1(input_data=X,
                       reward_mode=self.reward_mode,
                       score_type=self.reward_score_type,
                       regression_type=self.reward_regression_type,
                       alpha=self.reward_gpr_alpha)
        #  Inter-Community Critic


        max_reward = float('-inf')
        best_graph=None

        print(f'Shape of input batch: {self.batch_size}, 'f'{self.seq_length}, {self.input_dim}')

        print('Starting training.')
        best_grap = None

        for i in tqdm(range(1, self.iteration + 1)):



            with ThreadPoolExecutor(max_workers=num_communities) as executor:
                futures = []
                for comm_id,wrapper in community_actors.items():
                    futures.append(executor.submit(
                        self._process_community,
                        wrapper,
                        data_generactors[comm_id],
                        i,
                        reward1
                    ))
                community_outputs = [f.result() for f in futures]



            causal_graph,reward_value,graph = self._merge_actions_to_graph(community_outputs,reward1)


            if max_reward < reward_value:
                max_reward = reward_value
                best_graph = causal_graph
            # metrics=MetricsDAG(pruned_graph,true_dag)
            # met=metrics.metrics

            print('reward_value',reward_value)


        return best_graph

    def _merge_actions_to_graph(self, community_outputs, reward=None, data=None):
        n = self.seq_length
        global_graph = torch.zeros((n, n), device=self.device)

        for output in community_outputs:
            node_indices = output['node_indices']  #
            actions = output['actions']  # [B, seq_len]

            for b in range(actions.shape[0]):
                local_order = actions[b].cpu().numpy().tolist()
                local_dag = get_graph_from_order(local_order).cpu()  # [comm_len, comm_len]

                for i_local, i_global in enumerate(node_indices):
                    for j_local, j_global in enumerate(node_indices):
                        global_graph[i_global, j_global] += local_dag[i_local, j_local]
        graph = global_graph.detach().cpu().numpy()

        final_score,_= reward.calculate_reward_single_graph(global_graph) if reward else 0.0

        return global_graph, final_score,graph

    def _process_community(self,wrapper,data_generator,i,reward):
        time_start = time.time()
        actor=wrapper.actor.to(self.device)
        critic=wrapper.critic.to(self.device)
        optimizer=wrapper.critic_optimizer
        torch.autograd.set_detect_anomaly(True)
        input_batch = data_generator.draw_batch(batch_size=self.batch_size,
                                                dimension=self.input_dim).float().to(self.device)  # [batch_size, n_nodes, input_dim]
        # [batch_size, n_nodes, input_dim]
                                                                                                 #input_dim
        # print("Sample input batch device:", input_batch.device)
        encoder_output=actor.encode(input_batch)    #[batch_size, embed_dim]
        # print("encoder_output_shape:",encoder_output.shape)
        decoder_output=actor.decode(encoder_output)
        actions,mask_scores,s_list,h_list,c_list=decoder_output



        zero_matrix = get_graph_from_order(actions[0])

        reward,_ = reward.calculate_reward_single_graph(zero_matrix)


        # men
        predicted_value =critic.predict_env(stats_x=s_list[:, :-1, :]).float()  #
        #
        target_value = reward.detach().float()  #
        # print('predicted_value',predicted_value)
        # print('target_value',target_value)

        critic_loss = F.mse_loss(predicted_value, target_value)
        optimizer.zero_grad()  #
        critic_loss.backward()  #
        optimizer.step()  #

        predicted_value = critic.predict_env(stats_x=s_list[:, :-1, :])
        actor_loss = -predicted_value.mean()

        # 清空梯度并反向传播
        wrapper.optimizer.zero_grad()
        actor_loss.backward()
        wrapper.optimizer.step()
        time_end = time.time()
        print('call tiome', time_end - time_start)
        # end_time = time.time()
        # print('time2', end_time - start_time)
        # 在训练函数（_process_community）末尾

        return {
            'comm_id': wrapper.community_id,
            'node_indices': wrapper.node_indices,
            'actions': actions.to(self.device),
            's_list': s_list.detach(),
            'h_list': h_list.detach(),
            'c_list': c_list.detach(),
            'actor': actor,
            'wrapper': wrapper,
            'mask_scores': mask_scores,
        }



if __name__ == '__main__':

      all_real = []
      data_file = "mdd_90.mat"

      data1 = sio.loadmat(data_file, squeeze_me=True)
      data = data1['bold']
      print(data.shape)


      size = data.shape[0]
      all_labels = np.array(data1['label'])
      print(all_labels.shape)

      skeleton=np.load("mdd_90.mat")
      G=nx.Graph(skeleton)
      im = Infomap("--tree")

      #
      for u, v in G.edges():
          im.add_link(u, v)
      im.run()

      partition_level2 = {}
      for node in im.nodes:
          path = node.path  #
          if len(path) >= 1:  #

              partition_level2[node.node_id] = path[0] # 取前两层

      num_modules_level2 = len(set(partition_level2.values()))

      num_modules_level2 = len(set(partition_level2.values()))
      #
      unique_modules = sorted(set(partition_level2.values()))

      module_mapping = {mod: i + 1 for i, mod in enumerate(unique_modules)}

      #
      partition_level2 = {node: module_mapping[mod] for node, mod in partition_level2.items()}

      #
      clusters_level2 = defaultdict(list)
      for node_id, module_id in partition_level2.items():
          clusters_level2[module_id].append(node_id)



      time_start = time.time()
      for id_subject in [0,1,2,3,121,122,123]:

          # print('prunX:', prunX.shape)
          simple = data[id_subject]
          n=MetaRLEC()
          n.learn(simple,clusters_level2)
          pruned_graph= n.causal_matrix.detach().cpu().numpy()
          all_real.append(pruned_graph)

          # print("causal",n.causal_matrix.cpu().numpy())
      all_bold = np.stack(all_real, axis=0)
      sio.savemat("results/real/mad-lec_mdd_160.mat", {"bold": all_bold})



