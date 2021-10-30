import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import os
import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from general_utils import AttrDict
from sprites_datagen.utils.template_blender import TemplateBlender
from sprites_datagen.utils.trajectory import ConstantSpeedTrajectory
from sprites_datagen.moving_sprites import DistractorTemplateMovingSpritesGenerator, MovingSpriteDataset
from general_utils import make_image_seq_strip
from sprites_datagen.rewards import ZeroReward, AgentXReward, AgentYReward, TargetXReward, TargetYReward


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RewardRegHead(nn.Module):
    def __init__(self, dim_in):
        super(RewardRegHead, self).__init__()
        self.fc1 = nn.Linear(dim_in, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        self.fc1 = self.fc1.cuda()
        self.fc2 = self.fc2.cuda()
        out = self.fc1(x)
        out = self.fc2(out)
        return out


class MultiRewardRegHead(nn.Module):
    def __init__(self):
        super(MultiRewardRegHead, self).__init__()
        self.head1 = RewardRegHead(128)
        # ----------------------- Uncomment the section to allow more heads -----------------------
        self.head2 = RewardRegHead(128)
        self.head3 = RewardRegHead(128)
        self.head4 = RewardRegHead(128)

    def forward(self, x):
        final_output = self.head1(x)
        # curr_head_output should have dimension: [batch_size, #steps that we want to predict, 1] -> 每个future target time step对应一个reward value
        final_output = torch.unsqueeze(final_output, 0)

        # ----------------------- Uncomment the section to allow more heads -----------------------
        head2_out = self.head2(x)
        head2_out = torch.unsqueeze(head2_out, 0)
        final_output = torch.cat((final_output, head2_out), 0)

        head3_out = self.head3(x)
        head3_out = torch.unsqueeze(head3_out, 0)
        final_output = torch.cat((final_output, head3_out), 0)

        head4_out = self.head4(x)
        head4_out = torch.unsqueeze(head4_out, 0)
        final_output = torch.cat((final_output, head4_out), 0)

        # print('final_output.shape: ', final_output.shape)

        # Currently: final output has dimension: [#reward_functions, batch_size, #steps that we want to predict, 1]
        final_output = final_output.permute(1, 0, 2, 3)
        # Now: final output has dimension: [batch_size, #reward_functions, #steps that we want to predict, 1]
        final_output = torch.squeeze(final_output, 3)
        # Now: final output has dimension: [batch_size, #reward_functions, #steps that we want to predict->each corresponds to a reward]
        return final_output


# This part only includes the process:
# 1) encoding the images with the encoder
# 2) flattening the output embeddings -> omitted to fit the task of RL down-stream
# -> does not include the MLP after the encoder
class ImageEncoder(nn.Module):
    def __init__(self, input_resolution, input_seq_len):
        super(ImageEncoder, self).__init__()
        self.input_resolution = input_resolution
        self.init_num_channels = 4
        self.conv_layers = []
        self.input_seq_len = input_seq_len

        indicator = input_resolution

        # ----------------------- Uncomment the section to change the number of channels of the input image -----------------------
        # in_num_channels = 3
        in_num_channels = 1
        out_num_channels = self.init_num_channels

        while indicator > 1:
            # Encoder应该是一个FCN, 没有池化和激活函数
            temp_conv_layer = nn.Conv2d(in_num_channels, out_num_channels, 2, stride=2).to(device)
            self.conv_layers.append(temp_conv_layer)
            relu_layer = nn.ReLU().to(device)
            self.conv_layers.append(relu_layer)
            in_num_channels = out_num_channels
            out_num_channels = out_num_channels * 2
            indicator = indicator / 2

        self.conv_layers = nn.Sequential(*self.conv_layers)

    def forward(self, x):
        # print('input.shape in ImageEncoder forward: ', x.shape)
        b_size = x.size(0)
        input_seq_length = x.size(1)
        # ----------------------- Uncomment the section to change the num of channels -----------------------
        # x = x.reshape(b_size * input_seq_length, 3, self.input_resolution, self.input_resolution)
        x = x.reshape(b_size * input_seq_length, 1, self.input_resolution, self.input_resolution)
        out = x

        self.conv_layers = self.conv_layers.cuda()

        out = self.conv_layers(out)

        out = out.view(b_size * input_seq_length, -1).transpose(0, 1).contiguous().view(-1, b_size, input_seq_length)
        out = out.permute(1, 2, 0)
        # The output will be in dimension: [batch_size, sequence_length, num_channels*1*1]
        return out

    def num_flattend_feat(self, x):
        size = x.size()[1:]
        num_feat = 1
        for dim in size:
            num_feat *= dim
        return num_feat


# Single-layer LSTM defined below
class PredictorLSTM(nn.Module):
    def __init__(self, input_dim, pred_step_len, hidden_size=128, num_layers=1):
        super(PredictorLSTM, self).__init__()
        self.LSTM_hidden_size = hidden_size
        self.LSTM_layer_num = num_layers
        self.overall_input_dim = input_dim
        self.preproc_fc = nn.Linear(input_dim, 64)
        self.pred_step_len = pred_step_len
        self.LSTM_structure = nn.LSTM(64, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # print(x.shape)
        self.preproc_fc = self.preproc_fc.cuda()
        self.LSTM_structure = self.LSTM_structure.cuda()
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # Note: 这里要手动unroll这个LSTM
        fc_out = self.preproc_fc(x)
        # fc_out dimension should be：[batch_size, sequence_length, 64]
        # print('fc_out\'s dimension in PredictorLSTM class forward function: ', fc_out.shape)

        # Initialize the parameters for hidden states h and memory cells C
        b_size = fc_out.size(0)
        h0 = torch.zeros(self.LSTM_layer_num, fc_out.size(0), self.LSTM_hidden_size).to(device)
        c0 = torch.zeros(self.LSTM_layer_num, fc_out.size(0), self.LSTM_hidden_size).to(device)

        final_out = []

        for i in range(self.pred_step_len):
            # Debug: h0和c0是否需要在每次循环的时候重新初始化
            if i == 0:
                out, (h_n, h_c) = self.LSTM_structure(fc_out, (h0, c0))
                # print("shape of h_n in PredictorLSTM forward function before transformed: ", h_n.shape)
                h_n = h_n.transpose_(0, 1).contiguous().view(b_size, -1)
                final_out = torch.unsqueeze(h_n, 1)
                # print("shape of h_n in PredictorLSTM forward function after transformed: ", final_out.shape)

            else:
                padding = torch.zeros(fc_out.size(0), i, fc_out.size(2)).to(device)
                curr_input = torch.cat((fc_out[:, i:, :], padding), 1)
                out, (h_n, h_c) = self.LSTM_structure(curr_input, (h0, c0))
                h_n = h_n.transpose_(0, 1).contiguous().view(b_size, -1)
                h_n = torch.unsqueeze(h_n, 1)
                final_out = torch.cat((final_out, h_n), 1)
                # print("shape of h_n in PredictorLSTM forward function after transformed (non-zero loop): ", final_out.shape)

            h0 = torch.zeros(self.LSTM_layer_num, out.size(0), self.LSTM_hidden_size).to(device)
            c0 = torch.zeros(self.LSTM_layer_num, out.size(0), self.LSTM_hidden_size).to(device)

        # final out should be: [batch_size, #steps that we want to predict, #hidden_state_features=128]
        # print('final_out.shape is: ', final_out.shape)
        return final_out


# Code for the class of the integrated model
# pred_step_size = the size of future steps we want to predict into
class RewardInducedRep(nn.Module):
    def __init__(self, in_resolution, k_tasks, lstm_hidden_size, pred_step_size, in_sequence_length, mlp_lstm_in_dim):
        super(RewardInducedRep, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        # Definition for each components of the model
        self.reward_heads = MultiRewardRegHead()
        self.encoder = ImageEncoder(input_seq_len=in_sequence_length, input_resolution=in_resolution)
        self.LSTM_structure = PredictorLSTM(input_dim=mlp_lstm_in_dim, pred_step_len=pred_step_size, hidden_size=self.lstm_hidden_size, num_layers=1)

    def forward(self, x):
        encoder_output = self.encoder(x)
        lstm_output = self.LSTM_structure(encoder_output)
        # lstm_output dimension: [batch_size, #steps that we want to predict, #hidden_state_features=128]
        # final_output = []
        final_output = self.reward_heads(lstm_output)
        # for k in range(len(self.reward_heads)):
        #     curr_head_output = self.reward_heads[k](lstm_output)
        #     # curr_head_output should have dimension: [batch_size, #steps that we want to predict, 1] -> 每个future target time step对应一个reward value
        #
        #     curr_head_output = torch.unsqueeze(curr_head_output, 0)
        #     if k == 0:
        #         final_output = curr_head_output
        #         # print('final_output shape in RewardInducedRep forward = ', final_output.shape)
        #     else:
        #         final_output = torch.cat((final_output, curr_head_output), 0)
        #         # print('final_output shape in RewardInducedRep forward = ', final_output.shape)
        #
        # # Currently: final output has dimension: [#reward_functions, batch_size, #steps that we want to predict, 1]
        # final_output = final_output.permute(1, 0, 2, 3)
        # # Now: final output has dimension: [batch_size, #reward_functions, #steps that we want to predict, 1]
        # final_output = torch.squeeze(final_output, 3)
        # # Now: final output has dimension: [batch_size, #reward_functions, #steps that we want to predict->each corresponds to a reward]
        return final_output


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


# def train(model, train_loader, criterion, optimizer, epoch, lists):
def train(model, train_traj, num_traj_per_epoch, batch_size, criterion, optimizer, epoch, lists):
    losses = AverageMeter('Loss', ':.4e')

    model.train()

    # for i, item in enumerate(train_loader):
    for i in range(int(num_traj_per_epoch / batch_size)):
        item = train_traj.__getitem__(0)
        item['images'] = item['images'][:, :num_tot_time_step, :, :, :]
        item['images'] = item['images'].to(device)
        indicate = 0
        target_rewards = []

        for key in item['rewards'].keys():
            if indicate == 0:
                target_rewards = torch.unsqueeze(item['rewards'][key][:, num_tot_time_step-1:], 1)  # Debug: 这里可能有问题 10/29
                indicate = 1
            else:
                target_rewards = torch.cat((target_rewards, torch.unsqueeze(item['rewards'][key][:, num_tot_time_step-1:], 1)), 1)

        # Current reward中的reward type顺序：[agent_x, agent_y, target_x, target_y]

        target_rewards = target_rewards.to(device)

        # print('item[images].shape: ', item['images'].shape)
        output = model(item['images'])
        loss = criterion(output, target_rewards)

        losses.update(loss.item(), item['images'].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lists['loss_train'].append(losses.avg)
    print(epoch, '-th epoch     ', 'train loss sum: ', losses.sum, '   train loss avg: ', losses.avg)
    return losses.avg


# def validate(model, val_loader, criterion, lists, whether_print):
def validate(model, val_traj, num_traj_per_epoch, batch_size, criterion, lists, whether_print):
    losses = AverageMeter('Loss', ':.4e')

    model.eval()

    with torch.no_grad():
        # for i, item in enumerate(val_loader):
        for i in range(int(num_traj_per_epoch / batch_size)):
            item = val_traj.__getitem__(0)
            item['images'] = item['images'][:, :num_tot_time_step, :, :, :]
            item['images'] = item['images'].to(device)
            indicate = 0
            target_rewards = []

            for key in item['rewards'].keys():
                if indicate == 0:
                    target_rewards = torch.unsqueeze(item['rewards'][key][:, num_tot_time_step-1:], 1)
                    indicate = 1
                else:
                    target_rewards = torch.cat((target_rewards, torch.unsqueeze(item['rewards'][key][:, num_tot_time_step-1:], 1)), 1)

            target_rewards = target_rewards.to(device)
            output = model(item['images'])
            loss = criterion(output, target_rewards)

            losses.update(loss.item(), item['images'].size(0))

        # if whether_print:
        #     # TODO: print loss plot
        #     # fpr, tpr, thresholds_roc = metrics.roc_curve(y_true, y_score, pos_label=1)
        #     # plot_roc(fpr, tpr, thresholds_roc, iteration)
        #     # precision, recall, thresholds_prc = metrics.precision_recall_curve(y_true, y_score, pos_label=1)
        #     # plot_prc(precision, recall, thresholds_prc, iteration)

        print('                ', 'valid loss sum: ', losses.sum, '   valid loss avg: ', losses.avg)
        lists['loss_val'].append(losses.avg)

    return losses.avg
    # return accuracy_curr, losses.avg, auc_roc_float, auc_prc_float


def plot_losses(lst_loss, title):
    plt.plot(lst_loss, '-r', label='loss')
    plt.xlabel('nth iteration')
    plt.legend(loc='upper left')
    plt.title(title)
    save_path = os.path.normpath("%s\%s" % ('plots', title+'.png'))
    plt.savefig(save_path)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    # ----------------------- Change / comment & uncomment the code below to change the number of training samples -----------------------
    num_train_traj = 15000
    num_val_traj = 3000

    # num_train_traj = 7000
    # num_val_traj = 1500
    # num_train_traj = 5000
    # num_val_traj = 1000

    batch_size = 50
    num_workers = 0
    # epochs = 400
    epochs = 80

    num_time_step_predicted = 3
    num_tot_time_step = 8
    length_gen_seq = num_tot_time_step + num_time_step_predicted - 1

    # ----------------------- Uncomment the section to allow more heads -----------------------
    spec = AttrDict(
        resolution=128,
        max_seq_len=length_gen_seq,  # Need the reward ground truth for several timesteps after the given input sequence
        max_speed=0.1,               # total image range [0, 1]
        obj_size=0.2,                # size of objects, full images is 1.0
        shapes_per_traj=3,           # number of shapes per trajectory
        # shapes_per_traj=2,
        rewards=[AgentXReward, AgentYReward, TargetXReward, TargetYReward],
        # rewards=[AgentXReward]
    )

    # ----------------------- Uncomment the section to allow more heads -----------------------
    model = RewardInducedRep(in_resolution=128,
                             k_tasks=4,
                             # k_tasks=1,
                             lstm_hidden_size=128,
                             pred_step_size=num_time_step_predicted,
                             in_sequence_length=num_tot_time_step,
                             mlp_lstm_in_dim=256)

    # train_traj = MovingSpriteDataset(spec, num_samples=num_train_traj)
    train_traj = MovingSpriteDataset(spec, batch_size)
    # train_loader = torch.utils.data.DataLoader(train_traj, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # val_traj = MovingSpriteDataset(spec, num_samples=num_val_traj)
    val_traj = MovingSpriteDataset(spec, batch_size)
    # val_loader = torch.utils.data.DataLoader(val_traj, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model.to(device)
    model.train()

    lst_loss_train = []
    # lst_acc_train = []
    lst_loss_val = []
    # lst_acc_val = []

    lowest_avg_loss_train = float('inf')
    lowest_avg_loss_val = float('inf')

    lists = {'loss_train': lst_loss_train,
             # 'acc_train': lst_acc_train,
             'loss_val': lst_loss_val
             # 'acc_val': lst_acc_val
             }

    for epoch in range(0, epochs):
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.MSELoss()

        # curr_loss_train = train(model, train_loader, criterion, optimizer, epoch, lists)
        curr_loss_train = train(model, train_traj, num_train_traj, batch_size, criterion, optimizer, epoch, lists)
        lowest_avg_loss_train = min(curr_loss_train, lowest_avg_loss_train)
        print("Currently the lowest average training loss =", lowest_avg_loss_train)

        # curr_loss_val = validate(model, val_loader, criterion, lists, whether_print=True)
        curr_loss_val = validate(model, val_traj, num_val_traj, batch_size, criterion, lists, whether_print=True)
        lowest_avg_loss_val = min(curr_loss_val, lowest_avg_loss_val)
        print("Currently the lowest average validation loss =", lowest_avg_loss_val)

    plot_losses(lists['loss_train'], 'train_loss_plot_10_28_3shape_4rewards_400epochs')
    plot_losses(lists['loss_val'], 'valid_loss_plot_10_28_3shape_4rewards_400epochs')

    torch.save(model.state_dict(), 'rep_learning_10_28_3shape_4rewards_400epochs.pth')


