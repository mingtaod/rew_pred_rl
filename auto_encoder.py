import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from general_utils import AttrDict
from rep_learning import RewardInducedRep
from sprites_datagen.moving_sprites import MovingSpriteDataset
from sprites_datagen.rewards import AgentYReward, TargetXReward, TargetYReward, AgentXReward

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_tot_time_step = 8


class ConvAutoencoder(nn.Module):
    def __init__(self, encoder, fine_tune=False):
        super(ConvAutoencoder, self).__init__()
        self.encoder = encoder

        # 10/28 Notes: 检查这里是不是写错了，有可能并没有把requires_grad改成false
        if not fine_tune:
            for p in self.parameters():
                # print('parameter p printed: ', p)
                p.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (2, 2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (2, 2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (2, 2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (2, 2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, (2, 2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, (2, 2), stride=2),
            nn.ReLU(),
            # --------------- Change the code below: number of output channels ---------------
            # nn.ConvTranspose2d(4, 3, (2, 2), stride=2)
            nn.ConvTranspose2d(4, 1, (2, 2), stride=2)
        )

    def forward(self, x):
        print('init x input.shape: ', x.shape)
        x = self.encoder(x)

        b_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(b_size *seq_len, -1)

        x = torch.unsqueeze(x, 2)
        x = torch.unsqueeze(x, 3)
        print('x.shape in forward of the autoencoder after: ', x.shape)

        x = self.decoder(x)
        print('x.shape in ConvAutoencoder.forward output: ', x.shape)

        # 这里还需要改输出的dimension, 把前两个dimension恢复
        # x = x.view(b_size, seq_len, 3, 128, 128)
        x = x.view(b_size, seq_len, 1, 128, 128)

        return x


def train(net, dataloader, test_dataloader, epochs=5, loss_fn=nn.MSELoss(), title=None):
    net.to(device)
    net.train()

    optim = torch.optim.Adam(net.parameters())

    train_losses = []
    validation_losses = []

    for i in range(epochs):
        for idx, item in enumerate(dataloader):
            # batch = batch.to(device)
            item['images'] = item['images'][:, :num_tot_time_step, :, :, :]
            item['images'] = item['images'].to(device)

            # print('item[images].shape in train: ', item['images'].shape)

            optim.zero_grad()
            loss = loss_fn(item['images'], net(item['images']))
            loss.backward()
            optim.step()

            train_losses.append(loss.item())
        if title:
            image_title = f'{title} - Epoch {i}'

        evaluate(validation_losses, net, test_dataloader, title=image_title)


def show_visual_progress(model, test_dataloader, rows=5, flatten=True, vae=False, conditional=False, title=''):
    if title:
        plt.title(title)

    iter(test_dataloader)
    # image_rows = []

    for i, item in enumerate(test_dataloader):
        item['images'] = item['images'][:, :num_tot_time_step, :, :, :]
        item['images'] = item['images'].to(device)

        # images = model(item['images']).detach().cpu().numpy()
        images = torch.squeeze(model(item['images'])).detach().cpu().numpy()

        for k in range(1, 9):
            plt.subplot(2, 8, k)
            print('item[images][0, k-1]: ', item['images'][0, k-1].shape)
            # plt.imshow(item['images'][0, k-1].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
            plt.imshow(torch.squeeze(item['images'][0, k-1]).detach().cpu().numpy(), cmap='gray')

        for k in range(9, 17):
            # temp = np.transpose(images[0, k-9], (1, 2, 0))
            # r, g, b = temp[:, :, 0], temp[:, :, 1], temp[:, :, 2]
            # img_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            # plt.subplot(2, 8, k)
            # plt.imshow(img_gray, cmap='gray')
            plt.subplot(2, 8, k)
            plt.imshow(images[0, k-9], cmap='gray')

    if title:
        title = title.replace(" ", "_")
        plt.savefig(title)
    plt.show()


def evaluate(losses, autoencoder, dataloader, flatten=True, vae=False, conditional=False, title=""):
    loss = calculate_loss(autoencoder, dataloader, flatten=flatten)
    show_visual_progress(autoencoder, dataloader, flatten=flatten, vae=vae, conditional=conditional, title=title)
    losses.append(loss)


def calculate_loss(model, dataloader, loss_fn=nn.MSELoss(), flatten=True):
    losses = []
    for i, item in enumerate(dataloader):
        item['images'] = item['images'][:, :num_tot_time_step, :, :, :]
        item['images'] = item['images'].to(device)
        # print('item[images].shape in calculate_loss: ', item['images'].shape)

        loss = loss_fn(item['images'], model(item['images']))
        losses.append(loss)

    return (sum(losses)/len(losses)).item()


if __name__ == '__main__':
    num_train_traj = 5000
    num_val_traj = 1000

    batch_size = 50
    num_workers = 0
    epochs = 10

    num_time_step_predicted = 3
    num_tot_time_step = 8
    length_gen_seq = num_tot_time_step + num_time_step_predicted - 1

    # --------------- Change shapes per traj here starts ---------------
    spec = AttrDict(
        resolution=128,
        max_seq_len=length_gen_seq,  # Need the reward ground truth for several timesteps after the given input sequence
        max_speed=0.1,               # total image range [0, 1]
        obj_size=0.2,                # size of objects, full images is 1.0
        # shapes_per_traj=2,           # number of shapes per trajectory
        shapes_per_traj=3,
        rewards=[AgentXReward, AgentYReward, TargetXReward, TargetYReward],
    )
    # --------------- Change shapes per traj here ends ---------------

    loaded_model = RewardInducedRep(in_resolution=128,
                             # --------------- Uncomment here to recover multiple rewards training ---------------
                             k_tasks=4,  # 10/28 Notes: 之前这里写错了，但是好像不是特别重要，因为我们只需要提取出模型中的image encoder部分
                             # k_tasks=1,
                             lstm_hidden_size=128,
                             pred_step_size=num_time_step_predicted,
                             in_sequence_length=num_tot_time_step,
                             mlp_lstm_in_dim=256)

    # save_model = torch.load('model weights\\rep_learning_10_6_AgentX_reward_three_shape.pth')
    # save_model = torch.load('model weights\\rep_learning_10_25_3shape_4rewards.pth')
    save_model = torch.load('model weights\\rep_learning_10_28_3shape_4rewards_400epochs.pth')
    model_dict = loaded_model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    loaded_model.load_state_dict(model_dict)

    image_encoder = list(loaded_model.children())[1]
    model = ConvAutoencoder(encoder=image_encoder, fine_tune=False)

    train_traj = MovingSpriteDataset(spec, num_samples=num_train_traj)
    train_loader = torch.utils.data.DataLoader(train_traj, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_traj = MovingSpriteDataset(spec, num_samples=num_val_traj)
    val_loader = torch.utils.data.DataLoader(val_traj, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    train(model, train_loader, val_loader, epochs=epochs, loss_fn=nn.MSELoss(), title='conv_autoencoder_output')
