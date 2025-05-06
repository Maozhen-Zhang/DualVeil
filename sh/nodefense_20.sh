#python main.py --config conf_20 --model vgg11 --n_client 20 --defense fedavg --attack noatt --dataset cifar10 --wandb --task cifar_vgg_20
#python main.py --config conf_20 --model vgg11 --n_client 20 --defense deepsight --attack noatt --dataset cifar10 --wandb --task cifar_vgg_20
#python main.py --config conf_20 --model vgg11 --n_client 20 --defense foolsgold --attack noatt --dataset cifar10 --wandb --task cifar_vgg_20
#python main.py --config conf_20 --model vgg11 --n_client 20 --defense mkrum --attack noatt --dataset cifar10 --wandb --task cifar_vgg_20
#python main.py --config conf_20 --model vgg11 --n_client 20 --defense rlr --attack noatt --dataset cifar10 --wandb --task cifar_vgg_20
#
#
#python main.py --config conf_20 --model resnet18 --n_client 20 --defense fedavg --attack noatt --dataset cifar10 --wandb --task cifar_resnet_20
#python main.py --config conf_20 --model resnet18 --n_client 20 --defense deepsight --attack noatt --dataset cifar10 --wandb --task cifar_resnet_20
#python main.py --config conf_20 --model resnet18 --n_client 20 --defense foolsgold --attack noatt --dataset cifar10 --wandb --task cifar_resnet_20
#python main.py --config conf_20 --model resnet18 --n_client 20 --defense mkrum --attack noatt --dataset cifar10 --wandb --task cifar_resnet_20
#python main.py --config conf_20 --model resnet18 --n_client 20 --defense rlr --attack noatt --dataset cifar10 --wandb --task cifar_resnet_20
#

#python main.py --config conf_20 --model vgg11 --n_client 20 --defense fedavg --attack noatt --dataset cifar10 --wandb --task cifar_vgg11
#python main.py --config conf_20 --model vgg11 --n_client 20 --defense deepsight --attack noatt --dataset cifar10 --wandb --task cifar_vgg11
#python main.py --config conf_20 --model vgg11 --n_client 20 --defense foolsgold --attack noatt --dataset cifar10 --wandb --task cifar_vgg11
#python main.py --config conf_20 --model vgg11 --n_client 20 --defense mkrum --attack noatt --dataset cifar10 --wandb --task cifar_vgg11
#python main.py --config conf_20 --model vgg11 --n_client 20 --defense rlr --attack noatt --dataset cifar10 --wandb --task cifar_vgg11


python main.py --config conf_20 --model resnet18 --n_client 20 --defense fedavg --attack noatt --dataset tiny-imagenet --wandb --task imagenet_resnet_20
python main.py --config conf_20 --model resnet18 --n_client 20 --defense deepsight --attack noatt --dataset tiny-imagenet --wandb --task imagenet_resnet_20
python main.py --config conf_20 --model resnet18 --n_client 20 --defense foolsgold --attack noatt --dataset tiny-imagenet --wandb --task imagenet_resnet_20
python main.py --config conf_20 --model resnet18 --n_client 20 --defense mkrum --attack noatt --dataset tiny-imagenet --wandb --task imagenet_resnet_20
python main.py --config conf_20 --model resnet18 --n_client 20 --defense rlr --attack noatt --dataset tiny-imagenet --wandb --task imagenet_resnet_20
