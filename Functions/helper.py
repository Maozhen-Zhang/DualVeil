import math
import os

import numpy as np
import torch
import torchvision

from Metrics.metrics import Metrics

DEVICE = 'cuda:0'


class Helper:

    @staticmethod
    def copyParams(target_model, source_weight):
        target_model.load_state_dict(source_weight)
        return target_model

    @staticmethod
    def metric(cfg, model, dataloader, IsBackdoor=False, poison_method=None, trigger=None, epoch=None):
        device = cfg.device
        model.eval()
        model = model.to(device)

        total_loss = 0.0
        correct = 0
        datasize = 0
        with torch.no_grad():
            for batch_id, (data, target) in enumerate(dataloader):
                data, target = data.to(device), torch.tensor(target).to(device)
                if IsBackdoor:
                    # print((trigger[0] * trigger[1])[:, 3:6, 6:9])
                    index = np.where(target.cpu() != cfg.target_label)[0]
                    ori_data, ori_target = data[index], target[index]
                    data, target = poison_method(cfg, (data, target), trigger, IsTest=True)
                if data.shape[0] == 0:
                    continue
                output = model(data)
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
                datasize += data.shape[0]
                if batch_id == 0 and IsBackdoor:
                    save_image = torch.concat(
                        [ori_data[:7], data[:7], (data[:7] - ori_data[:7]), ((data[:7] - ori_data[:7]) * 10)], dim=0)
                    os.makedirs(f"visual/{cfg.attack}", exist_ok=True)
                    torchvision.utils.save_image(save_image,
                                                 f"visual/{cfg.attack}/{cfg.attack}-batchid_{batch_id}-backdoor_{IsBackdoor}.png",
                                                 nrow=7)
                    if epoch is not None:
                        for i, img in enumerate(ori_data):
                            os.makedirs(f"visual/{cfg.attack}-0.4", exist_ok=True)
                            img = Helper.denormalized(img) if cfg.normalize else img
                            torchvision.utils.save_image(img, f"visual/{cfg.attack}-0.4/{cfg.attack}-epoch_{epoch}-{batch_id}-{i}-ori.png")
                        for i, img in enumerate(data):
                            os.makedirs(f"visual/{cfg.attack}-0.4", exist_ok=True)
                            img = Helper.denormalized(img) if cfg.normalize else img
                            torchvision.utils.save_image(img, f"visual/{cfg.attack}-0.4/{cfg.attack}-epoch_{epoch}-{batch_id}-{i}-poison.png")


                    # torchvision.utils.save_image(torch.cat([data[:7]], dim=0), f"visual/{cfg.attack}-batchid_{batch_id}-backdoor_{IsBackdoor}.png",
                    #                  nrow=7)
        if datasize == 0:
            return 0, 0, 0, 0
        loss = total_loss / datasize
        acc = float(correct) / datasize
        return acc, loss, correct, datasize

    @staticmethod
    def saveinfo(cfg, save_dict, e):
        if cfg.epsilon == 8/255.0:
            epsilon = 8
        else:
            epsilon = 4
        torch.save(save_dict, f"./checkpoints-new/{cfg.model}-{cfg.dataset}-{cfg.attack}-{cfg.defense}-{cfg.n_client}/{cfg.lr}-{cfg.agglr}-{cfg.agglr}-{cfg.n_client}-{epsilon}-epoch_{e}.pth")

    @staticmethod
    def load_checkpoint(cfg, model):
        current_defense = cfg.defense
        print(f"查找路径：./checkpoints-new/{cfg.model}-{cfg.dataset}-noatt-fedavg-{cfg.n_client}/{cfg.checkpoint_path}")
        save_dict = torch.load(f'./checkpoints-new/{cfg.model}-{cfg.dataset}-noatt-fedavg-{cfg.n_client}/{cfg.checkpoint_path}')
        print(f"|--- Load model from {cfg.checkpoint_path}")
        model.load_state_dict(save_dict['global_weight'])

        return model

    import torch
    @staticmethod
    def denormalized(x):
        mean = torch.tensor([0.485, 0.456, 0.406])  # 归一化时使用的均值
        std = torch.tensor([0.229, 0.224, 0.225])  # 归一化时使用的标准差
        mean = mean.to(x.device)
        std = std.to(x.device)

        # 反向还原归一化
        x_denorm = x * std[:, None, None] + mean[:, None, None]
        return x_denorm
