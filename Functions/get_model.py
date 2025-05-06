import torch
from torch import nn

from Models.Alex import alex_cifar10
from Models.init_model import CNNMnist, LogisticRegression
from Models.resnet import resnet34, resnet50
from Models.simple import SimpleNet, CNNCifar
# from Models.vgg import VGG
from Models.vgg_model import VGG
from Models.resnet_cifar import ResNet18
# from models_test.resnet import resnet18 as ResNet18
from Models.simple_vit import SimpleViT
import torch.nn.functional as F
import timm

def interpolate_pos_encoding(model, img_size):
    # Extract the patch size correctly
    patch_size = model.patch_embed.patch_size[0]

    old_num_patches = model.patch_embed.num_patches
    model.patch_embed.num_patches = (img_size // patch_size) ** 2

    # Interpolate the positional embeddings
    new_pos_embed = nn.Parameter(
        F.interpolate(
            model.pos_embed[:, 1:]
            .view(1, int(old_num_patches**0.5), int(old_num_patches**0.5), -1)
            .permute(0, 3, 1, 2),
            size=(
                img_size // patch_size,
                img_size // patch_size,
            ),
            mode="bicubic",
        )
        .permute(0, 2, 3, 1)
        .view(1, -1, model.embed_dim)
    )
    model.pos_embed = nn.Parameter(
        torch.cat([model.pos_embed[:, :1], new_pos_embed], dim=1)
    )
    return model

def init_model(cfg):
    model = cfg.model
    classes = cfg.classes
    if model == 'lr':
        net = LogisticRegression(784, 10)
    # elif model == 'cnn':
    #     net = CNNMnist(10)
    elif model == 'cnn':
        net = SimpleNet(10)
    elif model == 'CNNCifar':
        net = CNNCifar()
    elif model == 'alex_cifar':
        net = alex_cifar10()
    elif model == 'resnet18':
        net = ResNet18(num_classes=classes)
        net.avgpool_name = cfg.dataset
        if cfg.dataset == 'tiny-imagenet':
            net.avgpool = nn.AdaptiveAvgPool2d(1)
            net.linear = nn.Linear(256, 200)

    elif model == 'resnet34':
        net = resnet34(num_classes=classes)
    elif model == 'resnet50':
        net = resnet50(num_classes=classes)
    elif model == 'dba_resnet18':
        net = ResNet18(name="resnet18")
    elif model == 'vgg11':
        net = VGG('VGG11', num_classes=classes, channels=3)
    elif model == 'vgg16':
        net = VGG('VGG16', num_classes=classes)
    elif model == "vit":
        if cfg.dataset == "tiny-imagenet":
            print("Loading ViT-Tiny model...")
            model = timm.create_model(
                "vit_tiny_patch16_224",  # 改为 ViT-Tiny
                pretrained=True,
                num_classes=200,  # Tiny ImageNet 的类别数
            )

            # Adjust input size to match Tiny ImageNet
            model.default_cfg["input_size"] = (3, 64, 64)

            # Update the PatchEmbed layer's img_size
            model.patch_embed.img_size = (64, 64)
            net = interpolate_pos_encoding(model, img_size=64).to("cuda:0")
        if cfg.dataset == "cifar10":

            print("Loading ViT-Tiny model...")
            model = timm.create_model(
                "vit_tiny_patch16_224",  # 仍然使用 ViT-Tiny
                pretrained=True,
                num_classes=10,  # CIFAR-10 的类别数
            )

            # Adjust input size to match CIFAR-10
            model.default_cfg["input_size"] = (3, 32, 32)  # CIFAR-10 图像尺寸是 32x32

            # Update the PatchEmbed layer's img_size
            model.patch_embed.img_size = (32, 32)

            # 更新位置编码
            net = interpolate_pos_encoding(model, img_size=32).to("cuda:0")
    else:
        assert False, "Invalid model"

    return net
