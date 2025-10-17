import torch
import torch.nn as nn



class EpisodicCritic1(nn.Module):
    """"""

    def __init__(self, input_dim, neurons=(512, 256, 1),
                 activation=nn.ReLU(), device=None) -> None:
        super(EpisodicCritic1, self).__init__()
        self.input_dim = input_dim
        self.neurons = neurons
        self.output_dim = neurons[-1]
        self.hidden_units = neurons[:-1]
        self.activation = activation
        self.device = device

        # trainable parameters
        env_w0 = nn.init.xavier_uniform_(
            torch.empty(self.input_dim, self.neurons[0], device=self.device,dtype=torch.float32)
            )
        self.env_w0 = nn.Parameter(env_w0.requires_grad_(True))

        env_w1 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[0], self.neurons[1], device=self.device,dtype=torch.float32)
            )
        self.env_w1 = nn.Parameter(env_w1.requires_grad_(True))

        env_w2 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[1], self.neurons[-1], device=self.device,dtype=torch.float32)
            )
        self.env_w2 = nn.Parameter(env_w2.requires_grad_(True))

        env_b1 = torch.tensor([0.], requires_grad=True, device=self.device,dtype=torch.float32)
        self.env_b1 = nn.Parameter(env_b1)

        # Un-trainable parameters
        self.tgt_w0 = nn.init.xavier_uniform_(
            torch.empty(self.input_dim, self.neurons[0], device=self.device,dtype=torch.float32)
            )
        self.tgt_w1 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[0], self.neurons[1], device=self.device,dtype=torch.float32)
            )
        self.tgt_w2 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[1], self.neurons[-1], device=self.device,dtype=torch.float32)
            )
        self.tgt_b1 = torch.tensor([0.], device=self.device,dtype=torch.float32)

    def predict_env(self, stats_x) -> None:
        """predict environment reward"""

        stats_x = stats_x.detach()
        h0 = torch.einsum('ijk, kl->ijl', stats_x, self.env_w0)
        h0 = self.activation(h0)
        h1 = torch.einsum('ijk, kl->ijl', h0, self.env_w1)
        h1 = self.activation(h1)
        h2 = torch.einsum('ijk, kl->ijl', h1, self.env_w2)
        h2 = self.activation(h2)

        # [batch_size, seq_length - 1]
        self.prediction_env = (h2 + self.env_b1).squeeze()
        return self.prediction_env

    def predict_tgt(self, stats_y) -> None:
        """predict target reward"""

        stats_y = stats_y.detach()
        h0 = torch.einsum('ijk, kl->ijl', stats_y, self.tgt_w0)
        h0 = self.activation(h0)
        h1 = torch.einsum('ijk, kl->ijl', h0, self.tgt_w1)
        h1 = self.activation(h1)
        h2 = torch.einsum('ijk, kl->ijl', h1, self.tgt_w2)
        h2 = self.activation(h2)

        self.prediction_tgt = (h2 + self.tgt_b1).squeeze()

    def soft_replacement(self) -> None:
        # soft_replacement
        self.tgt_w0 = 0.95 * self.tgt_w0 + 0.05 * self.env_w0.detach()
        self.tgt_w1 = 0.95 * self.tgt_w1 + 0.05 * self.env_w1.detach()
        self.tgt_w2 = 0.95 * self.tgt_w2 + 0.05 * self.env_w2.detach()
        self.tgt_b1 = 0.95 * self.tgt_b1 + 0.05 * self.env_b1.detach()




class EpisodicCritic(nn.Module):
    def __init__(self, input_dim, neurons=(512, 256, 1),
                 activation=nn.ReLU(), device=None) -> None:
        super(EpisodicCritic, self).__init__()
        self.input_dim = input_dim
        self.neurons = neurons
        self.output_dim = neurons[-1]
        self.hidden_units = neurons[:-1]
        self.activation = activation
        self.device = device

        # 可训练参数
        self.env_w0 = nn.Parameter(torch.empty(self.input_dim, self.neurons[0], device=self.device, dtype=torch.float32))
        nn.init.xavier_uniform_(self.env_w0)
        self.env_w1 = nn.Parameter(torch.empty(self.neurons[0], self.neurons[1], device=self.device, dtype=torch.float32))
        nn.init.xavier_uniform_(self.env_w1)
        self.env_w2 = nn.Parameter(torch.empty(self.neurons[1], self.neurons[-1], device=self.device, dtype=torch.float32))
        nn.init.xavier_uniform_(self.env_w2)
        self.env_b1 = nn.Parameter(torch.zeros(1, device=self.device, dtype=torch.float32))

        # 非训练参数（目标网络）
        self.tgt_w0 = torch.empty_like(self.env_w0)
        self.tgt_w1 = torch.empty_like(self.env_w1)
        self.tgt_w2 = torch.empty_like(self.env_w2)
        self.tgt_b1 = torch.zeros_like(self.env_b1)

        with torch.no_grad():
            self.tgt_w0.copy_(self.env_w0)
            self.tgt_w1.copy_(self.env_w1)
            self.tgt_w2.copy_(self.env_w2)

    def predict_env(self, stats_x) -> None:
        stats_x = stats_x.detach()  # [n_comm, batch, seq_len, embed_dim]
        h0 = self.activation(torch.einsum('...jk, kl -> ...jl', stats_x, self.env_w0))
        h1 = self.activation(torch.einsum('...jk, kl -> ...jl', h0, self.env_w1))
        h2 = self.activation(torch.einsum('...jk, kl -> ...jl', h1, self.env_w2))
        self.prediction_env = (h2 + self.env_b1).squeeze(-1).mean(dim=2)

    def predict_tgt(self, stats_y) -> None:
        stats_y = stats_y.detach().float()
        h0 = self.activation(torch.einsum('...jk, kl -> ...jl', stats_y, self.tgt_w0))
        h1 = self.activation(torch.einsum('...jk, kl -> ...jl', h0, self.tgt_w1))
        h2 = self.activation(torch.einsum('...jk, kl -> ...jl', h1, self.tgt_w2))
        self.prediction_tgt = (h2 + self.tgt_b1).squeeze(-1).mean(dim=2)

    def soft_replacement(self) -> None:
        self.tgt_w0 = 0.95 * self.tgt_w0 + 0.05 * self.env_w0.detach()
        self.tgt_w1 = 0.95 * self.tgt_w1 + 0.05 * self.env_w1.detach()
        self.tgt_w2 = 0.95 * self.tgt_w2 + 0.05 * self.env_w2.detach()
        self.tgt_b1 = 0.95 * self.tgt_b1 + 0.05 * self.env_b1.detach()

class CriticMedium(nn.Module):
    def __init__(self, input_dim, activation=nn.ReLU(), device=None):
        super().__init__()
        self.input_dim = input_dim
        self.neurons = (128, 64, 32, 1)
        self.hidden_units = self.neurons[:-1]
        self.output_dim = self.neurons[-1]
        self.activation = activation
        self.device = device

        # 可训练参数
        self.env_w0 = nn.Parameter(nn.init.xavier_uniform_(
            torch.empty(input_dim, self.neurons[0], device=device)).float())
        self.env_w1 = nn.Parameter(nn.init.xavier_uniform_(
            torch.empty(self.neurons[0], self.neurons[1], device=device)).float())
        self.env_w2 = nn.Parameter(nn.init.xavier_uniform_(
            torch.empty(self.neurons[1], self.neurons[2], device=device)).float())
        self.env_w3 = nn.Parameter(nn.init.xavier_uniform_(
            torch.empty(self.neurons[2], self.neurons[3], device=device)).float())
        self.env_b1 = nn.Parameter(torch.zeros(1, device=device).float())

        # 目标参数
        self.tgt_w0 = nn.init.xavier_uniform_(
            torch.empty(input_dim, self.neurons[0], device=device))
        self.tgt_w1 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[0], self.neurons[1], device=device))
        self.tgt_w2 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[1], self.neurons[2], device=device))
        self.tgt_w3 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[2], self.neurons[3], device=device))
        self.tgt_b1 = torch.zeros(1, device=device)

    def _compute(self, stats_x):
        h0 = self.activation(torch.einsum('ijk,kl->ijl', stats_x, self.env_w0))
        h1 = self.activation(torch.einsum('ijk,kl->ijl', h0, self.env_w1))
        h2 = self.activation(torch.einsum('ijk,kl->ijl', h1, self.env_w2))
        h3 = self.activation(torch.einsum('ijk,kl->ijl', h2, self.env_w3))
        return (h3 + self.env_b1).squeeze()

    def forward(self, stats_x):
        return self._compute(stats_x)

    def predict_env(self, stats_x):
        return self._compute(stats_x.detach())


class CriticSmall(nn.Module):
    def __init__(self, input_dim, neurons=(512, 256, 1),
                 activation=nn.ReLU(), device=None) -> None:
        super(CriticSmall, self).__init__()
        self.input_dim = input_dim
        self.neurons = neurons
        self.output_dim = neurons[-1]
        self.hidden_units = neurons[:-1]
        self.activation = activation
        self.device = device

        # trainable parameters
        env_w0 = nn.init.xavier_uniform_(
            torch.empty(self.input_dim, self.neurons[0], device=self.device).float()
        )
        self.env_w0 = nn.Parameter(env_w0.requires_grad_(True))

        env_w1 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[0], self.neurons[1], device=self.device).float()
        )
        self.env_w1 = nn.Parameter(env_w1.requires_grad_(True)).float()

        env_w2 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[1], self.neurons[-1], device=self.device).float()
        )
        self.env_w2 = nn.Parameter(env_w2.requires_grad_(True)).float()

        env_b1 = torch.tensor([0.], requires_grad=True, device=self.device).float()
        self.env_b1 = nn.Parameter(env_b1)

        # Un-trainable parameters
        self.tgt_w0 = nn.init.xavier_uniform_(
            torch.empty(self.input_dim, self.neurons[0], device=self.device).float()
        )
        self.tgt_w1 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[0], self.neurons[1], device=self.device)
        )
        self.tgt_w2 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[1], self.neurons[-1], device=self.device)
        )
        self.tgt_b1 = torch.tensor([0.], device=self.device)

    def predict_env(self, stats_x) -> None:
        """predict environment reward"""

        stats_x = stats_x
        h0 = torch.einsum('ijk, kl->ijl', stats_x, self.env_w0)
        h0 = self.activation(h0)
        h1 = torch.einsum('ijk, kl->ijl', h0, self.env_w1)
        h1 = self.activation(h1)
        h2 = torch.einsum('ijk, kl->ijl', h1, self.env_w2)
        h2 = self.activation(h2)

        # [batch_size, seq_length - 1]
        self.prediction_env = (h2 + self.env_b1).squeeze()
        return self.prediction_env

class CriticLarge(nn.Module):
    def __init__(self, input_dim, activation=nn.ReLU(), device=None):
        super().__init__()
        self.input_dim = input_dim
        self.neurons = (256, 256, 1)
        self.hidden_units = self.neurons[:-1]
        self.output_dim = self.neurons[-1]
        self.activation = activation
        self.device = device

        # 可训练参数
        self.env_w0 = nn.Parameter(nn.init.xavier_uniform_(
            torch.empty(input_dim, self.neurons[0], device=device)))
        self.env_w1 = nn.Parameter(nn.init.xavier_uniform_(
            torch.empty(self.neurons[0], self.neurons[1], device=device)))
        self.env_w2 = nn.Parameter(nn.init.xavier_uniform_(
            torch.empty(self.neurons[1], self.neurons[2], device=device)))
        self.env_b1 = nn.Parameter(torch.zeros(1, device=device))

        # 目标参数
        self.tgt_w0 = nn.init.xavier_uniform_(
            torch.empty(input_dim, self.neurons[0], device=device))
        self.tgt_w1 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[0], self.neurons[1], device=device))
        self.tgt_w2 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[1], self.neurons[2], device=device))
        self.tgt_b1 = torch.zeros(1, device=device)

    def _compute(self, stats_x):
        h0 = self.activation(torch.einsum('ijk,kl->ijl', stats_x, self.env_w0))
        h1 = self.activation(torch.einsum('ijk,kl->ijl', h0, self.env_w1)) + h0  # 残差
        h2 = self.activation(torch.einsum('ijk,kl->ijl', h1, self.env_w2))
        return (h2 + self.env_b1).squeeze()

    def forward(self, stats_x):
        return self._compute(stats_x)

    def predict_env(self, stats_x):
        return self._compute(stats_x.detach())

class intra_EpisodicCritic(nn.Module):
    """"""

    def __init__(self, input_dim, neurons=(512, 256, 1),
                 activation=nn.ReLU(), device=None) -> None:
        super(intra_EpisodicCritic, self).__init__()
        self.input_dim = input_dim
        self.neurons = neurons
        self.output_dim = neurons[-1]
        self.hidden_units = neurons[:-1]
        self.activation = activation
        self.device = device

        # trainable parameters
        env_w0 = nn.init.xavier_uniform_(
            torch.empty(self.input_dim, self.neurons[0], device=self.device)
            )
        self.env_w0 = nn.Parameter(env_w0.requires_grad_(True))

        env_w1 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[0], self.neurons[1], device=self.device)
            )
        self.env_w1 = nn.Parameter(env_w1.requires_grad_(True))

        env_w2 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[1], self.neurons[-1], device=self.device)
            )
        self.env_w2 = nn.Parameter(env_w2.requires_grad_(True))

        env_b1 = torch.tensor([0.], requires_grad=True, device=self.device)
        self.env_b1 = nn.Parameter(env_b1)

        # Un-trainable parameters
        self.tgt_w0 = nn.init.xavier_uniform_(
            torch.empty(self.input_dim, self.neurons[0], device=self.device)
            )
        self.tgt_w1 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[0], self.neurons[1], device=self.device)
            )
        self.tgt_w2 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[1], self.neurons[-1], device=self.device)
            )
        self.tgt_b1 = torch.tensor([0.], device=self.device)

    def predict_env(self, stats_x) -> None:
        """predict environment reward"""

        stats_x = stats_x.detach()
        h0 = torch.einsum('ijk, kl->ijl', stats_x, self.env_w0)
        h0 = self.activation(h0)
        h1 = torch.einsum('ijk, kl->ijl', h0, self.env_w1)
        h1 = self.activation(h1)
        h2 = torch.einsum('ijk, kl->ijl', h1, self.env_w2)
        h2 = self.activation(h2)

        # [batch_size, seq_length - 1]
        self.prediction_env = (h2 + self.env_b1).squeeze()
        return self.prediction_env




import torch
import torch.nn as nn

class onlyCritic(nn.Module):
    def __init__(self, neurons=(512, 256, 1),
                 activation=nn.ReLU(), device=None):
        super(onlyCritic, self).__init__()
        self.device = device or torch.device('cpu')

        # 使用 LazyLinear 延迟确定输入维度
        self.env_model = nn.Sequential(
            nn.LazyLinear(neurons[0]),
            nn.LayerNorm(neurons[0]),
            activation,
            nn.Dropout(0.1),
            nn.Linear(neurons[0], neurons[1]),
            activation,
            nn.Linear(neurons[1], neurons[2])
        ).to(self.device)

        self.target_model = nn.Sequential(
            nn.LazyLinear(neurons[0]),
            nn.LayerNorm(neurons[0]),
            activation,
            nn.Dropout(0.1),
            nn.Linear(neurons[0], neurons[1]),
            activation,
            nn.Linear(neurons[1], neurons[2])
        ).to(self.device)

        self._init_target()

    def _init_target(self):
        self.target_model.load_state_dict(self.env_model.state_dict())

    def soft_update(self, tau=0.05):
        for tgt, src in zip(self.target_model.parameters(), self.env_model.parameters()):
            tgt.data.copy_(tau * src.data + (1. - tau) * tgt.data)

    def predict_env(self, x):
        x = x.detach()
        out = self.env_model(x)
        return out.squeeze(-1)

    def predict_tgt(self, x):
        x = x.detach()
        out = self.target_model(x)
        return out.squeeze(-1)





class DenseCritic(nn.Module):
    """Critic network for `dense reward` type

    Only one layer network.
    """

    def __init__(self, input_dim, output_dim, device=None) -> None:
        super(DenseCritic, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.h0 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim,
                      device=device),
            nn.ReLU().to(device=device)
        )
        self.w1 = torch.rand(self.output_dim, 1,
                             device=device).requires_grad_(True)
        self.b1 = torch.tensor([0.], requires_grad=True, device=device)
        self.params = nn.ParameterList([nn.Parameter(self.w1),
                                        nn.Parameter(self.b1)])

    def predict_reward(self, encoder_output) -> torch.Tensor:
        """Predict reward for `dense reward` type"""

        frame = torch.mean(encoder_output, 1).detach()
        h0 = self.h0(frame)
        prediction = torch.matmul(h0, self.w1) + self.b1

        return prediction.squeeze()

