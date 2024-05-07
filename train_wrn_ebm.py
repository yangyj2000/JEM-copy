# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import utils
import torch as t, torch.nn as nn, torch.nn.functional as tnnF, torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
import functools
#import ipdb
import numpy as np
import wideresnet
import json
# Sampling
from tqdm import tqdm
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
im_sz = 32
n_ch = 3


class DataSubset(Dataset):          # 数据集的子集
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


# 定义一个F类，继承nn.Module
class F(nn.Module):
    # 初始化函数，depth表示深度，width表示宽度，norm表示归一化，dropout_rate表示dropout率，n_classes表示类别数
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(F, self).__init__()
        # 实例化wideresnet.Wide_ResNet，并赋值给self.f
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        # 实例化nn.Linear，并赋值给self.energy_output
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        # 实例化nn.Linear，并赋值给self.class_output
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    # 定义前向传播函数，x表示输入，y表示标签
    def forward(self, x, y=None):
        # 调用self.f函数，获取penult_z
        penult_z = self.f(x)
        # 调用self.energy_output函数，返回energy_output
        return self.energy_output(penult_z).squeeze()

    # 定义分类函数，x表示输入
    def classify(self, x):
        # 调用self.f函数，获取penult_z
        penult_z = self.f(x)
        # 调用self.class_output函数，返回class_output
        return self.class_output(penult_z).squeeze()


# 定义CCF类，继承自F类
class CCF(F):
    # 初始化函数，设置深度、宽度、归一化、dropout_rate和n_classes
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(CCF, self).__init__(depth, width, norm=norm, dropout_rate=dropout_rate, n_classes=n_classes)

    # 定义前向传播函数，输入x和y，如果y为空，返回logits的logsumexp，否则返回logits中y的值
    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:, None])

# 定义一个循环函数，参数为loader
def cycle(loader):
    # 循环
    while True:
        # 遍历loader中的每一个元素
        for data in loader:
            # 返回每一个元素
            yield data


# 计算参数梯度的二范数
def grad_norm(m):
    total_norm = 0
    for p in m.parameters():
        # 获取参数梯度
        param_grad = p.grad
        if param_grad is not None:
            # 计算参数梯度的二范数
            param_norm = param_grad.data.norm(2) ** 2
            total_norm += param_norm
    # 计算总范数
    total_norm = total_norm ** (1. / 2)
    return total_norm.item()


# 用来计算参数梯度的值
def grad_vals(m):
    # 创建一个空列表，用来存放参数梯度的值
    ps = []
    for p in m.parameters():
        if p.grad is not None:
            # 将参数梯度的值添加到列表中
            ps.append(p.grad.data.view(-1))
    # 将列表中的元素拼接成Tensor
    ps = t.cat(ps)
    # 返回参数梯度的均值、标准差、绝对梯度的均值、标准差、绝对梯度的最小值、最大值
    return ps.mean().item(), ps.std(), ps.abs().mean(), ps.abs().std(), ps.abs().min(), ps.abs().max()


# 初始化随机数，参数args和bs
def init_random(args, bs):
    # 返回一个bs大小的n_ch通道的im_sz大小的随机数组，范围在-1到1之间
    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


# 定义函数get_model_and_buffer，用于获取模型和缓冲区
# 参数args：超参数
# 参数device：设备
# 参数sample_q：缓冲区
def get_model_and_buffer(args, device, sample_q):
    # 定义模型类
    model_cls = F if args.uncond else CCF
    # 初始化模型
    f = model_cls(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate, n_classes=args.n_classes)
    # 如果不是无条件模型，则断言缓冲区大小必须可以被类别数整除
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, "Buffer size must be divisible by args.n_classes"
    # 如果加载路径为空，则初始化随机缓冲区
    if args.load_path is None:
        # make replay buffer
        replay_buffer = init_random(args, args.buffer_size)
    else:
        print(f"loading model from {args.load_path}")
        # 加载模型
        ckpt_dict = t.load(args.load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        replay_buffer = ckpt_dict["replay_buffer"]

    # 将模型和缓冲区移动到设备
    f = f.to(device)
    return f, replay_buffer


# 定义一个函数get_data，用于获取数据
def get_data(args):
    # 根据参数args.dataset的值，设置不同的transform_train
    if args.dataset == "svhn":
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(im_sz),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)]
        )
    else:       # 定义训练集变换
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),     # 将图片填充到指定大小
             tr.RandomCrop(im_sz),      # 随机裁剪图片
             tr.RandomHorizontalFlip(),     # 随机水平翻转图片
             tr.ToTensor(),
             # 将图片的像素值转换为[0.5, 0.5, 0.5]
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             # 将图片的像素值加上随机噪声
             lambda x: x + args.sigma * t.randn_like(x)]
        )
    # 根据参数args.dataset的值，设置不同的transform_test
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + args.sigma * t.randn_like(x)]
    )
    # 定义一个函数dataset_fn，用于根据参数args.dataset的值，返回不同的数据集
    def dataset_fn(train, transform):
        if args.dataset == "cifar10":
            return tv.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "cifar100":
            return tv.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
        else:
            return tv.datasets.SVHN(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")
    # 获取所有训练数据的索引
    full_train = dataset_fn(True, transform_train)
    all_inds = list(range(len(full_train)))
    # 设置随机种子
    np.random.seed(1234)
    # 打乱索引
    np.random.shuffle(all_inds)
    # 设置验证集
    if args.n_valid is not None:
        valid_inds, train_inds = all_inds[:args.n_valid], all_inds[args.n_valid:]
    else:
        valid_inds, train_inds = [], all_inds
    # 获取训练集和验证集的索引
    train_inds = np.array(train_inds)
    train_labeled_inds = []
    other_inds = []
    # 获取训练集标签
    train_labels = np.array([full_train[ind][1] for ind in train_inds])
    # 根据参数args.labels_per_class的值，获取训练集标签
    if args.labels_per_class > 0:
        for i in range(args.n_classes):
            print(i)
            train_labeled_inds.extend(train_inds[train_labels == i][:args.labels_per_class])
            other_inds.extend(train_inds[train_labels == i][args.labels_per_class:])
    else:
        train_labeled_inds = train_inds

    # 获取训练集、训练集标签、验证集和测试集
    dset_train = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_inds)
    dset_train_labeled = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_labeled_inds)
    dset_valid = DataSubset(
        dataset_fn(True, transform_test),
        inds=valid_inds)
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    dload_train_labeled = cycle(dload_train_labeled)
    dset_test = dataset_fn(False, transform_test)
    dload_valid = DataLoader(dset_valid, batch_size=4, shuffle=False, num_workers=0, drop_last=False)
    dload_test = DataLoader(dset_test, batch_size=4, shuffle=False, num_workers=0, drop_last=False)
    # 返回训练集、训练集标签、验证集和测试集
    return dload_train, dload_train_labeled, dload_valid,dload_test


def get_sample_q(args, device):
    def sample_p_0(replay_buffer, bs, y=None):
        # 如果replay_buffer为空，返回init_random函数的返回值，[]
        if len(replay_buffer) == 0:
            return init_random(args, bs), []
        # 定义buffer_size为replay_buffer的长度，如果y不为空，buffer_size除以args.n_classes
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // args.n_classes
        # 定义inds为t.randint函数的返回值，bs为bs
        inds = t.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        # 如果y不为空，inds乘以buffer_size加上inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
            assert not args.uncond, "Can't drawn conditional samples without giving me y"
        # 定义buffer_samples为replay_buffer的inds
        buffer_samples = replay_buffer[inds]
        # 定义random_samples为init_random函数的返回值
        random_samples = init_random(args, bs)
        # 定义choose_random为t.rand函数的返回值，bs为bs
        choose_random = (t.rand(bs) < args.reinit_freq).float()[:, None, None, None]
        # 定义samples为choose_random乘以random_samples加上（1-choose_random）乘以buffer_samples
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        # 返回samples.to(device)，inds
        return samples.to(device), inds

    def sample_q(f, replay_buffer, y=None, n_steps=args.n_steps):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        f.eval()        # 设置f为评估模式
        # get batch size
        bs = args.batch_size if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = t.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]       # f_prime是x_k的梯度
            x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k)      # langevin
        f.train()
        final_samples = x_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples
    return sample_q


# 定义一个函数，用于评估分类任务
def eval_classification(f, dload, device):
    # 初始化正确率和损失
    corrects, losses = [], []
    # 遍历dload中的每一个数据对
    for x_p_d, y_p_d in dload:
        # 将数据对转换到指定的设备
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        # 使用f.classify函数对x_p_d进行分类
        logits = f.classify(x_p_d)
        # 计算损失
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().numpy()
        # 将损失添加到损失列表中
        losses.extend(loss)
        # 计算正确率
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        # 将正确率添加到正确率列表中
        corrects.extend(correct)
    # 计算损失
    loss = np.mean(losses)
    # 计算正确率
    correct = np.mean(corrects)
    # 返回正确率和损失
    return correct, loss


def checkpoint(f, buffer, tag, args, device):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "replay_buffer": buffer
    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)


def main(args):
    # 创建保存文件夹
    utils.makedirs(args.save_dir)
    # 将参数写入文件
    with open(f'{args.save_dir}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)
    # 将输出重定向到日志文件
    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

    # 设置随机种子
    t.manual_seed(seed)
    # 设置cuda的随机种子
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

    # datasets
    dload_train, dload_train_labeled, dload_valid, dload_test = get_data(args)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    sample_q = get_sample_q(args, device)
    f, replay_buffer = get_model_and_buffer(args, device, sample_q)

    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))     # 保存图片，nrow表示每行图片的数量，normalize表示是否将图片的像素值归一化到[0, 1]之间

    # optimizer
    params = f.class_output.parameters() if args.clf_only else f.parameters()
    if args.optimizer == "adam":
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)
    else:
        optim = t.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)

    best_valid_acc = 0.0
    cur_iter = 0
    for epoch in range(args.n_epochs):
        if epoch in args.decay_epochs:                              # [160, 180]
            for param_group in optim.param_groups:
                new_lr = param_group['lr'] * args.decay_rate        # 学习率衰减，每次衰减为原来的args.decay_rate倍
                param_group['lr'] = new_lr                          # 更新学习率
            print("Decaying lr to {}".format(new_lr))
        for i, (x_p_d, _) in tqdm(enumerate(dload_train)):          # 遍历训练集，获取每一个数据对，x_p_d为数据，_为标签
            if cur_iter <= args.warmup_iters:                       # -1, 不热身
                lr = args.lr * cur_iter / float(args.warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            x_p_d = x_p_d.to(device)
            x_lab, y_lab = dload_train_labeled.__next__()           # x_lab为训练集数据，y_lab为训练集标签
            x_lab, y_lab = x_lab.to(device), y_lab.to(device)

            L = 0.
            if args.p_x_weight > 0:  # maximize log p(x)
                if args.class_cond_p_x_sample:
                    assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                    y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)       # 随机生成类别
                    x_q = sample_q(f, replay_buffer, y=y_q)         # 从缓冲区中采样，f是模型
                else:
                    x_q = sample_q(f, replay_buffer)  # sample from log-sumexp

                fp_all = f(x_p_d)
                fq_all = f(x_q)
                fp = fp_all.mean()
                fq = fq_all.mean()

                l_p_x = -(fp - fq)          # loss function
                if cur_iter % args.print_every == 0:
                    print('P(x) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq,
                                                                                                   fp - fq))
                L += args.p_x_weight * l_p_x

            if args.p_y_given_x_weight > 0:  # maximize log p(y | x)
                logits = f.classify(x_lab)                              # logits表示x_lab的分类结果
                l_p_y_given_x = nn.CrossEntropyLoss()(logits, y_lab)            # loss function
                if cur_iter % args.print_every == 0:
                    acc = (logits.max(1)[1] == y_lab).float().mean()
                    print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch,
                                                                                 cur_iter,
                                                                                 l_p_y_given_x.item(),
                                                                                 acc.item()))
                L += args.p_y_given_x_weight * l_p_y_given_x

            if args.p_x_y_weight > 0:  # maximize log p(x, y)
                assert not args.uncond, "this objective can only be trained for class-conditional EBM DUUUUUUUUHHHH!!!"
                x_q_lab = sample_q(f, replay_buffer, y=y_lab)
                fp, fq = f(x_lab, y_lab).mean(), f(x_q_lab, y_lab).mean()
                l_p_x_y = -(fp - fq)            # loss function
                if cur_iter % args.print_every == 0:
                    print('P(x, y) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq,
                                                                                                      fp - fq))

                L += args.p_x_y_weight * l_p_x_y

            # break if the loss diverged...easier for poppa to run experiments this way
            if L.abs().item() > 1e8:
                print("BAD BOIIIIIIIIII")
                1/0

            optim.zero_grad()           # 梯度清零
            L.backward()                # 反向传播，计算梯度
            optim.step()                # 更新参数
            cur_iter += 1

            if cur_iter % 100 == 0:         # 每100代
                if args.plot_uncond:        # 生成无条件样本，保存图片
                    if args.class_cond_p_x_sample:
                        assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                        y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                        x_q = sample_q(f, replay_buffer, y=y_q)
                    else:
                        x_q = sample_q(f, replay_buffer)
                    plot('{}/x_q_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q)
                if args.plot_cond:  # generate class-conditional samples
                    y = t.arange(0, args.n_classes)[None].repeat(args.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                    x_q_y = sample_q(f, replay_buffer, y=y)
                    plot('{}/x_q_y{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q_y)

        if epoch % args.ckpt_every == 0:
            checkpoint(f, replay_buffer, f'ckpt_{epoch}.pt', args, device)

        if epoch % args.eval_every == 0 and (args.p_y_given_x_weight > 0 or args.p_x_y_weight > 0):
            f.eval()
            with t.no_grad():
                # validation set
                correct, loss = eval_classification(f, dload_valid, device)
                print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))
                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Best Valid!: {}".format(correct))
                    checkpoint(f, replay_buffer, "best_valid_ckpt.pt", args, device)
                # test set
                correct, loss = eval_classification(f, dload_test, device)
                print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))
            f.train()
        checkpoint(f, replay_buffer, "last_ckpt.pt", args, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "svhn", "cifar100"])
    parser.add_argument("--data_root", type=str, default="./data")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", help="If set, then only train the classifier")
    parser.add_argument("--labels_per_class", type=int, default=-1,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    # loss weighting
    parser.add_argument("--p_x_weight", type=float, default=1.)
    parser.add_argument("--p_y_given_x_weight", type=float, default=1.)
    parser.add_argument("--p_x_y_weight", type=float, default=0.)
    # regularization
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=3e-2,
                        help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"],
                        help="norm to add to weights, none works fine")
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=20,
                        help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
    parser.add_argument("--uncond", action="store_true", help="If set, then the EBM is unconditional")
    parser.add_argument("--class_cond_p_x_sample", action="store_true",
                        help="If set we sample from p(y)p(x|y), othewise sample from p(x),"
                             "Sample quality higher if set, but classification accuracy better if not.")
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true", help="If true, directs std-out to log file")
    parser.add_argument("--plot_cond", action="store_true", help="If set, save class-conditional samples")
    parser.add_argument("--plot_uncond", action="store_true", help="If set, save unconditional samples")
    parser.add_argument("--n_valid", type=int, default=5000)

    args = parser.parse_args()
    args.n_classes = 100 if args.dataset == "cifar100" else 10
    main(args)