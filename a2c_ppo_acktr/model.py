import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
from rep_learning import RewardInducedRep


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None,
                 is_cnn=True,
                 is_rew_pred=False,
                 is_rew_pred_finetune=False):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                if is_cnn:
                    print('CNNBase chosen')
                    base = CNNBase
                elif is_rew_pred:
                    print('RewardPredictionBase chosen')
                    base = RewardPredictionBase

            elif len(obs_shape) == 1:
                print('MLPBase chosen')
                base = MLPBase    # State input try this
            else:
                print('Shit happens')
                raise NotImplementedError

        # ------------------ Manually change the code below ------------------
        if is_rew_pred:
            if is_rew_pred_finetune:
                self.base = base(finetune=True, **base_kwargs)
            else:
                print('RewardPredictionBase without finetune chosen')
                self.base = base(**base_kwargs)
        else:
            self.base = base(obs_shape[0], **base_kwargs)
        # ------------------ Manually change the code above ------------------
        # self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
            print('Box chosen; num_outputs = ', num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        # print('num_inputs: ', num_inputs)

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            # init_(nn.Linear(32 * 7 * 7, hidden_size)),
            init_(nn.Linear(4608, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        # print('inputs.shape in forward of CNNBase: ', inputs.shape)
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


# Added code for the base for reward prediction baseline
class RewardPredictionBase(NNBase):
    def __init__(self, recurrent=False, finetune=False, hidden_size=256):
        super(RewardPredictionBase, self).__init__(recurrent, hidden_size, hidden_size)

        num_time_step_predicted = 3
        num_tot_time_step = 8
        length_gen_seq = num_tot_time_step + num_time_step_predicted - 1

        model = RewardInducedRep(in_resolution=128,
                                 k_tasks=4,
                                 lstm_hidden_size=128,
                                 pred_step_size=num_time_step_predicted,
                                 in_sequence_length=num_tot_time_step,
                                 mlp_lstm_in_dim=256)

        # save_model = torch.load('model weights\\rep_learning_9_25.pth')
        # save_model = torch.load('model weights\\rep_learning_10_25_3shape_4rewards.pth')
        save_model = torch.load('model weights\\rep_learning_10_28_3shape_4rewards_400epochs.pth')
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        image_encoder = list(model.children())[1]

        # print('list(model.children()): ', list(model.children()))

        # if not finetune:
        #     for param in image_encoder.parameters():
        #         param.requires_grad = False

        self.main = image_encoder

        if not finetune:
            for param in self.main.parameters():
                param.requires_grad = False

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        # --------------- Added extra layers here for model tuning ---------------
        self.linear1 = init_(nn.Linear(hidden_size*4, hidden_size*2))
        self.linear2 = init_(nn.Linear(hidden_size*2, hidden_size))

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        # print('shape of inputs in forward of RewardPredictionBase: ', inputs.shape)
        # print('testing purpose: ', torch.unsqueeze(inputs[:, 0, :, :], 1).shape)
        x = None
        for i in range(inputs.shape[1]):
            inputs_part = torch.unsqueeze(inputs[:, 0, :, :], 1)
            inputs_part = torch.unsqueeze(inputs_part, 1)
            x_part = self.main(inputs_part)
            if i == 0:
                x = x_part
            else:
                x = torch.cat((x, x_part), 1)

        # print('checking x.shape in RewardPredictionBase: ', x.shape)

        x = x.reshape(x.shape[0], -1)

        # print('checking x.shape in RewardPredictionBase again: ', x.shape)

        x = self.linear1(x)
        x = self.linear2(x)

        # The output x will be in dimension: [num_envs, 4->framestack_len, num_channels*1*1]

        # x = self.main(inputs)
        # # The output x will be in dimension: [batch_size, sequence_length, num_channels*1*1]
        # x = torch.squeeze(x, dim=1)
        # # print('shape of outputs in forward of RewardPredictionBase: ', x.shape)

        return self.critic_linear(x), x, rnn_hxs
