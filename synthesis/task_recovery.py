import sys
import os
import numpy as np
import time
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseSynthesis
from ._utils import ImagePool2
from torch.autograd import Variable


def fomaml_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(tar_p.grad.data)   #, alpha=0.67

def reset_l0(model):
    for n,m in model.named_modules():
        if n == "l1.0" or n == "conv_blocks.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = ( self.mmt_rate*mean_mmt+(1-self.mmt_rate)*mean.data,
                         self.mmt_rate*var_mmt+(1-self.mmt_rate)*var.data )

    def remove(self):
        self.hook.remove()

class Synthesizer(BaseSynthesis):
    def __init__(self, args,teacher, student, generator, nz, num_classes, img_size,
                 iterations=5, lr_g=0.1,
                 synthesis_batch_size=128,
                  oh=1,adv=1, bn=10, num_teacher=4,
                 save_dir='run/cmi',transform=None,transform_no_toTensor=None,
                 device='cpu', c_abs_list=None,max_batch_per_class=20):
        super(Synthesizer, self).__init__(teacher, student)
        self.args=args
        self.save_dir = save_dir
        self.loss_fn = nn.CrossEntropyLoss()
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.nz = nz
        self.oh = oh
        self.adv = adv
        self.bn = bn
        self.num_teacher = num_teacher
        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.c_abs_list=c_abs_list
        self.transform = transform
        self.transforms = [1,1,1,1]
        self.generator = generator.to(device).train()
        self.device = device
        self.hooks = []
        self.ep = 0
        self.transform_no_toTensor=transform_no_toTensor
        self.transform_no_toTensors = [1,1,1,1]
        self.cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.data_pool = ImagePool2(args=self.args, root=self.save_dir, num_classes=self.num_classes,
                                    transform=self.transform, max_batch_per_class=max_batch_per_class)

    def synthesize(self, targets=None,student=None,mode=None,c_num=5,support=None):
        self.synthesis_batch_size =len(targets)//c_num
        ########################
        start = time.time()
        targets = torch.LongTensor(targets).to(self.device)
        # self.ep += 1
        # if (self.ep == 300):
        #     reset_l0(self.generator)
        optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g, betas=[0.5, 0.999])
        criteria_adv = F.kl_div
        optimizer.zero_grad()
        best_inputs = []
        for id in range(self.num_teacher):
            self.teacher[id].eval()
            z = torch.randn(size=(len(targets), self.nz), device=self.device).requires_grad_()
            best_cost = 1e6
            best_input = self.generator(z).data
            fast_generator = self.generator.clone()
            fast_optimizer = torch.optim.Adam([
                {'params': fast_generator.parameters()},
                {'params': [z], 'lr': 1e-3}
            ], lr=self.lr_g, betas=[0.5, 0.999])
            for it in range(self.iterations):
                inputs = fast_generator(z)
                if self.args.dataset == 'mix':
                    inputs_change = self.transform_no_toTensors[id](inputs)
                else:
                    inputs_change = self.transform_no_toTensor(inputs)
                #############################################
                #Loss
                #############################################
                t_out = self.teacher[id](inputs_change)
                loss_oh = F.cross_entropy( t_out, targets )
                loss = self.oh * loss_oh

                if student != None:
                    with torch.no_grad():
                        s_out = student(inputs_change)
                    mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                    loss_adv = -(criteria_adv(F.softmax(t_out, dim=-1), F.softmax(s_out.detach(), dim=-1),
                                              reduction='none').sum(1) * mask).mean()
                    loss = loss + self.adv * loss_adv

                fast_optimizer.zero_grad()
                loss.backward()

                if it == (self.iterations - 1):
                    fomaml_grad(self.generator, fast_generator)

                fast_optimizer.step()

            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_input = inputs.detach()
            best_inputs.append(best_input)
        end = time.time()
        if mode != 'warmup':
            self.data_pool.add(imgs=best_input, c_abs_list=self.c_abs_list[id],
                               synthesis_batch_size_per_class=self.synthesis_batch_size, mode=mode)
        optimizer.step()

        return best_inputs, end - start

    def get_random_task(self, num_w=5, num_s=5, num_q=15):
        return self.data_pool.get_random_task(num_w=num_w, num_s=num_s, num_q=num_q)
