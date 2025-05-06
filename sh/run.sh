#python main.py --config conf_20 --model resnet18 --n_client 20 --defense fedavg --attack flinvert --dataset cifar10 --wandb --task cifar_resnet_20
#python main.py --config conf_20 --model vgg11 --n_client 20 --defense fedavg --attack flinvert --dataset cifar10 --wandb --task cifar_resnet_20
python main.py --config conf_1000 --model resnet18 --n_client 1000 --defense fedavg --attack flinvert --dataset cifar10 --wandb --task cifar_resnet_cifar10_1000
python main.py --config conf_1000 --model vgg11 --n_client 1000 --defense fedavg --attack flinvert --dataset cifar10 --wandb --task cifar_vgg_1000


python main.py --config conf_20 --model vit --n_client 20 --defense fedavg --attack flinvert --dataset cifar10 --wandb --task cifar_resnet_20
python main.py --config conf_20 --model vit --n_client 20 --defense fedavg --attack flinvert --dataset tiny-imagenet --wandb --task cifar_resnet_20




python main.py --config conf_20 --model resnet18 --n_client 20 --defense fedavg --attack flinvert --dataset cifar10
python main.py --config conf_20 --model resnet18 --n_client 20 --defense fedavg --attack dba --dataset cifar10
python main.py --config conf_20 --model resnet18 --n_client 20 --defense fedavg --attack neurotoxin --dataset cifar10
python main.py --config conf_20 --model resnet18 --n_client 20 --defense fedavg --attack cerp --dataset cifar10
python main.py --config conf_20 --model resnet18 --n_client 20 --defense fedavg --attack f3ba --dataset cifar10
python main.py --config conf_20 --model resnet18 --n_client 20 --defense fedavg --attack iba --dataset cifar10


python main.py --config conf_20 --model resnet18 --n_client 20 --defense fedavg --attack flinvert --dataset cifar10