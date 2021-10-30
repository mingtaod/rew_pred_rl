import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from general_utils import AttrDict
from sprites_datagen.utils.template_blender import TemplateBlender
from sprites_datagen.utils.trajectory import ConstantSpeedTrajectory
from sprites_datagen.moving_sprites import DistractorTemplateMovingSpritesGenerator, MovingSpriteDataset
from rep_learning import RewardInducedRep


def run():
    torch.multiprocessing.freeze_support()


if __name__ == '__main__':
    import cv2
    from general_utils import make_image_seq_strip
    from sprites_datagen.rewards import ZeroReward, AgentXReward, AgentYReward, TargetYReward, TargetXReward

    run()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # spec = AttrDict(
    #     resolution=128,
    #     max_seq_len=30,
    #     max_speed=0.05,      # total image range [0, 1]
    #     obj_size=0.2,       # size of objects, full images is 1.0
    #     shapes_per_traj=4,      # number of shapes per trajectory
    #     rewards=[AgentYReward, ZeroReward, AgentXReward],
    # )
    #
    # gen = DistractorTemplateMovingSpritesGenerator(spec)
    # for i in range(3):
    #     traj = gen.gen_trajectory()
    #     print('-----------------------Iteration number: ', i, '-----------------------')
    #     print(traj.rewards)
    #
    #     print(traj.images.shape)
    #
    #     img = make_image_seq_strip([traj.images[None, :, None].repeat(3, axis=2).astype(np.float32)], sep_val=255.0).astype(np.uint8)
    #     cv2.imwrite("test.png", img[0].transpose(1, 2, 0))
    #     print(img.shape)

    spec = AttrDict(
        resolution=128,
        max_seq_len=8,
        max_speed=0.1,      # total image range [0, 1]
        obj_size=0.2,       # size of objects, full images is 1.0
        shapes_per_traj=3,      # number of shapes per trajectory
        rewards=[AgentXReward, AgentYReward, TargetXReward, TargetYReward],
    )

    train_traj = MovingSpriteDataset(spec, num_samples=100)

    train_loader = torch.utils.data.DataLoader(train_traj, batch_size=32, shuffle=True, num_workers=0)

    # for i, (train_data, target) in enumerate(train_loader):
    for i, item in enumerate(train_loader):
        # print("train_data: ", train_data)
        # print("target: ", target)
        print("Item number ", i, ": ", item["images"].shape)

    num_time_step_predicted = 3
    num_tot_time_step = 8
    length_gen_seq = num_tot_time_step + num_time_step_predicted - 1

    model = RewardInducedRep(in_resolution=128,
                             k_tasks=4,
                             lstm_hidden_size=128,
                             pred_step_size=num_time_step_predicted,
                             in_sequence_length=num_tot_time_step,
                             mlp_lstm_in_dim=256)

    # model.load_state_dict(torch.load('rep_learning_9_22.pth'))

    save_model = torch.load('rep_learning_9_25.pth')
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    print(state_dict.keys())
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    child_counter = 0
    for child in model.children():
        print(" child", child_counter, "is -")
        print(child)
        child_counter += 1

    print('i lost myself...: ', type(list(model.children())[0]))
