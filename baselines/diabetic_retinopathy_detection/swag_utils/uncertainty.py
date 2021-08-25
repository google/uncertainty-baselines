import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import tqdm

from swag import data, losses, models, utils
from swag.posteriors import SWAG, KFACLaplace


parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument("--file", type=str, default=None, required=True, help="checkpoint")

parser.add_argument(
    "--dataset", type=str, default="CIFAR10", help="dataset name (default: CIFAR10)"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/scratch/datasets/",
    metavar="PATH",
    help="path to datasets location (default: None)",
)
parser.add_argument(
    "--use_test",
    dest="use_test",
    action="store_true",
    help="use test dataset instead of validation (default: False)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size (default: 128)",
)
parser.add_argument("--split_classes", type=int, default=None)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    metavar="N",
    help="number of workers (default: 4)",
)
parser.add_argument(
    "--model",
    type=str,
    default="VGG16",
    metavar="MODEL",
    help="model name (default: VGG16)",
)
parser.add_argument(
    "--method",
    type=str,
    default="SWAG",
    choices=["SWAG", "KFACLaplace", "SGD", "HomoNoise", "Dropout", "SWAGDrop"],
    required=True,
)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    required=True,
    help="path to npz results file",
)
parser.add_argument("--N", type=int, default=30)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument(
    "--cov_mat", action="store_true", help="use sample covariance for swag"
)
parser.add_argument("--use_diag", action="store_true", help="use diag cov for swag")

parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)


def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.sum(np.log(ps))
    return nll


args = parser.parse_args()

eps = 1e-12
if args.cov_mat:
    args.cov_mat = True
else:
    args.cov_mat = False

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("Using model %s" % args.model)
model_cfg = getattr(models, args.model)

print("Loading dataset %s from %s" % (args.dataset, args.data_path))
loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    model_cfg.transform_train,
    model_cfg.transform_test,
    use_validation=not args.use_test,
    split_classes=args.split_classes,
    shuffle_train=False,
)
"""if args.split_classes is not None:
    num_classes /= 2
    num_classes = int(num_classes)"""

print("Preparing model")
if args.method in ["SWAG", "HomoNoise", "SWAGDrop"]:
    model = SWAG(
        model_cfg.base,
        no_cov_mat=not args.cov_mat,
        max_num_models=20,
        *model_cfg.args,
        num_classes=num_classes,
        **model_cfg.kwargs
    )
elif args.method in ["SGD", "Dropout", "KFACLaplace"]:
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
else:
    assert False
model.cuda()


def train_dropout(m):
    if type(m) == torch.nn.modules.dropout.Dropout:
        m.train()


print("Loading model %s" % args.file)
checkpoint = torch.load(args.file)
model.load_state_dict(checkpoint["state_dict"])

if args.method == "KFACLaplace":
    print(len(loaders["train"].dataset))
    model = KFACLaplace(
        model, eps=5e-4, data_size=len(loaders["train"].dataset)
    )  # eps: weight_decay

    t_input, t_target = next(iter(loaders["train"]))
    t_input, t_target = (
        t_input.cuda(non_blocking=True),
        t_target.cuda(non_blocking=True),
    )

if args.method == "HomoNoise":
    std = 0.01
    for module, name in model.params:
        mean = module.__getattr__("%s_mean" % name)
        module.__getattr__("%s_sq_mean" % name).copy_(mean ** 2 + std ** 2)


predictions = np.zeros((len(loaders["test"].dataset), num_classes))
targets = np.zeros(len(loaders["test"].dataset))
print(targets.size)

for i in range(args.N):
    print("%d/%d" % (i + 1, args.N))
    if args.method == "KFACLaplace":
        ## KFAC Laplace needs one forwards pass to load the KFAC model at the beginning
        model.net.load_state_dict(model.mean_state)

        if i == 0:
            model.net.train()

            loss, _ = losses.cross_entropy(model.net, t_input, t_target)
            loss.backward(create_graph=True)
            model.step(update_params=False)

    if args.method not in ["SGD", "Dropout"]:
        sample_with_cov = args.cov_mat and not args.use_diag
        model.sample(scale=args.scale, cov=sample_with_cov)

    if "SWAG" in args.method:
        utils.bn_update(loaders["train"], model)

    model.eval()
    if args.method in ["Dropout", "SWAGDrop"]:
        model.apply(train_dropout)
        # torch.manual_seed(i)
        # utils.bn_update(loaders['train'], model)

    k = 0
    for input, target in tqdm.tqdm(loaders["test"]):
        input = input.cuda(non_blocking=True)
        ##TODO: is this needed?
        # if args.method == 'Dropout':
        #    model.apply(train_dropout)
        torch.manual_seed(i)

        if args.method == "KFACLaplace":
            output = model.net(input)
        else:
            output = model(input)

        with torch.no_grad():
            predictions[k : k + input.size()[0]] += (
                F.softmax(output, dim=1).cpu().numpy()
            )
        targets[k : (k + target.size(0))] = target.numpy()
        k += input.size()[0]

    print("Accuracy:", np.mean(np.argmax(predictions, axis=1) == targets))
    #nll is sum over entire dataset
    print("NLL:", nll(predictions / (i + 1), targets))
predictions /= args.N

entropies = -np.sum(np.log(predictions + eps) * predictions, axis=1)
np.savez(args.save_path, entropies=entropies, predictions=predictions, targets=targets)
