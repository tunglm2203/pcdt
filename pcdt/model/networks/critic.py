import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf


class MLPQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        last_layer_activation: str = "Identity",
        init_w: float = 1e-3,
    ):
        super(MLPQNetwork, self).__init__()
        self.fc_layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.out = nn.Linear(hidden_dim, 1)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)
        self.last_layer_activation = getattr(nn, last_layer_activation)()

    def forward(self, q_input):
        num_layers = len(self.fc_layers)
        x = F.silu(self.fc_layers[0](q_input))
        for i in range(1, num_layers):
            x = F.silu(self.fc_layers[i](x))
        return self.last_layer_activation((self.out(x)))


class Critic(nn.Module):
    """Returns the action value function of a given observation, action pair"""

    def __init__(
        self,
        state_dim: int,
        goal_dim: int = 0,
        action_dim: int = 16,
        num_layers: int = 3,
        hidden_dim: int = 256,
    ):
        super(Critic, self).__init__()
        self.Q = MLPQNetwork(
            input_dim=state_dim + goal_dim + action_dim,
            num_layers=num_layers, hidden_dim=hidden_dim
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Performs a forward pass to obtain an action value function prediction"""
        # Case for batch size == 1
        if len(action.shape) == 2 and action.shape[0] == 1 and len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        q_input = torch.cat((obs, action), dim=-1)
        return self.Q(q_input)


