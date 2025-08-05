import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models.base import Model
from skrl.models.torch import GaussianMixin, DeterministicMixin

class MLP(nn.Module):
    def __init__(self, units, input_dim, device=None):
        super(MLP, self).__init__()
        self.device = device
        layers = []
        for output_dim in units:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ELU())
            input_dim = output_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def set_mode(self, mode: str) -> None:
        """Set the model mode (training or evaluation)

        :param mode: Mode: ``"train"`` for training or ``"eval"`` for evaluation.
            See `torch.nn.Module.train <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train>`_
        :type mode: str

        :raises ValueError: If the mode is not ``"train"`` or ``"eval"``
        """
        if mode == "train":
            self.train(True)
        elif mode == "eval":
            self.train(False)
        else:
            raise ValueError("Invalid mode. Use 'train' for training or 'eval' for evaluation")

# define the model
class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device,
                 privileged_obs_size=32,
                 privileged_dims=8,
                 base_net_units=[256, 256, 128]):
        Model.__init__(self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
            reduction="sum",
            role="policy",
        )
        self.privileged_observations_dims = privileged_obs_size
        self.prop_actions_dims = observation_space.shape[0] - self.privileged_observations_dims

        base_input_size = self.prop_actions_dims + privileged_dims
        self.base_net = MLP(units=base_net_units, input_dim=base_input_size).to(device)

        self.policy_layer = nn.LazyLinear(out_features=action_space.shape[0], device=device)
        self.log_std_parameter = nn.Parameter(torch.full(size=(action_space.shape[0],), fill_value=0.0), requires_grad=True).to(device)

    def act(self, inputs, role):
        return GaussianMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        # Extract privileged observations and prop actions
        prop_actions = inputs["obs"][:, :self.prop_actions_dims]
        privileged_features = inputs["privileged_features"]
        privileged_features = torch.tanh(privileged_features)  # Ensure privileged features are in the range [-1, 1]
        # print("privileged_features", privileged_features[0])
        # Concatenate privileged features with prop actions
        concatenated_input = torch.cat((prop_actions, privileged_features), dim=-1)
        
        self.policy_output = self.base_net(concatenated_input)
        output = self.policy_layer(self.policy_output)
        
        return output, self.log_std_parameter, {}
    
class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device,
                 value_net_units=[256, 256, 128]):
        Model.__init__(self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self)

        self.critic_net = MLP(units=value_net_units, input_dim=state_space.shape[0]).to(device)
        self.value_layer = nn.LazyLinear(out_features=1, device=device)

    def act(self, inputs, role):
        return DeterministicMixin.act(self, inputs, role)
    
    def compute(self, inputs, role):
        output = self.critic_net(inputs)
        value = self.value_layer(output)
        return value, {}

class VisionEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # (batch, 32, 32, 32)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (batch, 64, 16, 16)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (batch, 128, 8, 8)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # nn.Flatten(), # (batch, 128*8*8)
            # nn.Linear(128 * 8 * 8, 128),
            nn.AvgPool2d(8),
        )
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        z = self.encoder(x)
        z = self.fc(z.view(-1, 128))
        
        return z

class PropActionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            nn.ELU(),
            nn.Linear(hidden_layers[0], hidden_layers[1])
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return z

class InputEmbedding(nn.Module):
    def __init__(self, input_dim, model_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.embedding(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=30):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[None]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class AdaptationModule(nn.Module):
    def __init__(self, visuo_prop_transformer_input_dim, visuo_prop_transformer_model_dim,
                 visuo_prop_transformer_layers, visuo_prop_transformer_heads, visuo_prop_sequence_length, device):
        super().__init__()

        # vision encoder
        self.vision_encoder = VisionEncoder(latent_dim=32).to(device)

        # proprioceptive & action encoder
        self.prop_action_encoder = PropActionEncoder(input_dim=32, hidden_layers=[32,32]).to(device)

        # Positional Encoding
        self.input_embedding = InputEmbedding(input_dim=visuo_prop_transformer_input_dim, model_dim=visuo_prop_transformer_model_dim).to(device)
        self.positional_encoding = PositionalEncoding(model_dim=visuo_prop_transformer_model_dim, max_len=visuo_prop_sequence_length+1).to(device)

        # CLS tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, visuo_prop_transformer_model_dim, device=device))
        torch.nn.init.normal_(self.cls_token, std=0.02)

        self.out_layer = nn.Linear(visuo_prop_transformer_model_dim, 8).to(device)

        # Visuo-proprioceptive Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=visuo_prop_transformer_model_dim,
            nhead=visuo_prop_transformer_heads,
            dim_feedforward=visuo_prop_transformer_model_dim*2,
            batch_first=True,
            # dropout=0.5,
            device=device
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=visuo_prop_transformer_layers,
        )

    # def forward(self, prop_action, prev_input_sequence):
    def forward(self, image, prop_action, prev_input_sequence):
        vision_features = self.vision_encoder(image)  # (batch_size, latent_dim)
        # vision_features = torch.zeros_like(vision_features, device=vision_features.device)
        # process prop actions
        prop_action_features = self.prop_action_encoder(prop_action)  # (batch_size, hidden_dim)

        # Concatenate vision features and prop action features
        concatenated_features = torch.cat((vision_features,prop_action_features), dim=-1)

        # process input sequence
        input_sequence = torch.roll(prev_input_sequence, shifts=-1, dims=1)  # Shift input sequence to align with vision and prop action features
        input_sequence[:,-1] = concatenated_features  # Replace the last token with the concatenated features

        # process input sequence
        embedded_sequence = self.input_embedding(input_sequence)
        # add CLS token
        embedded_sequence = torch.column_stack((self.cls_token.repeat(embedded_sequence.size(0), 1, 1), embedded_sequence))  # Add CLS token at the beginning
        embedded_sequence = self.positional_encoding(embedded_sequence)

        # Structure: Encoder-only Transformer
        transformer_output = self.transformer_encoder(
            src=embedded_sequence,
        )

        output = transformer_output[:, 0, :]
        output = self.out_layer(output.flatten(1))  # Final output layer
        
        return output, input_sequence