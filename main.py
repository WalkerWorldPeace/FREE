import argparse
import os
import random
import shutil
import torch.nn.functional as F
import torch
from torch import nn
from tools import get_model, get_premodel, get_transform, \
    get_transform_no_toTensor, \
    label_abs2relative, get_dataloader, data2supportquery, Timer, setup_seed, compute_confidence_interval, Generator, \
    pretrains
from methods.maml import Maml, MamlKD
from synthesis.task_recovery import Synthesizer
import time
import logging
from torch.autograd import Variable
import wandb

parser = argparse.ArgumentParser(description='DFML')
#basic
parser.add_argument('--multigpu', type=str, default='0', help='seen gpu')
parser.add_argument('--gpu', type=int, default=0, help="gpu")
parser.add_argument('--dataset', type=str, default='cifar100', help='test dataset')
parser.add_argument('--pretrained_path_prefix', type=str, default='./pretrained_model', help='user-defined')
#memorys
parser.add_argument('--way_train', type=int, default=5, help='way')
parser.add_argument('--num_sup_train', type=int, default=5)
parser.add_argument('--num_qur_train', type=int, default=15)
parser.add_argument('--way_test', type=int, default=5, help='way')
parser.add_argument('--num_sup_test', type=int, default=5)
parser.add_argument('--num_qur_test', type=int, default=15)
parser.add_argument('--backbone', type=str, default='conv4',help='architecture of the meta model')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--warmup', type=int, default=20)
parser.add_argument('--episode_test', type=int, default=600)
parser.add_argument('--start_id', type=int, default=1)
parser.add_argument('--inner_update_num', type=int, default=5)
parser.add_argument('--test_inner_update_num', type=int, default=10)
parser.add_argument('--inner_lr', type=float, default=0.01)
parser.add_argument('--outer_lr', type=float, default=0.001)
parser.add_argument('--approx', action='store_true',default=False)
parser.add_argument('--episode_batch',type=int, default=10)
parser.add_argument('--val_interval',type=int, default=20)
#kd
parser.add_argument('--num_sup_kd', type=int, default=20)
parser.add_argument('--num_qur_kd', type=int, default=20)
parser.add_argument('--inner_update_num_kd', type=int, default=10)
parser.add_argument('--adv', type=float, default=1.0)
parser.add_argument('--bn', type=float, default=0.0)
#data free
parser.add_argument('--way_pretrain', type=int, default=5, help='way')
parser.add_argument('--pre_model_num', type=int, default=100)
parser.add_argument('--num_teacher', type=int, default=4)
parser.add_argument('--pre_backbone', type=str, default='conv4', help='conv4/resnet10/resnet18')
parser.add_argument('--pretrain', action='store_true',default=False)
parser.add_argument('--generate_interval', type=int, default=10)
parser.add_argument('--generate_iterations', type=int, default=5)
parser.add_argument('--Glr', type=float, default=0.001)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.multigpu
setup_seed(42)
device=torch.device('cuda:{}'.format(args.gpu))

########
if args.dataset=='cifar100':
    img_size = 32
    args.img_size =32
    channel = 3
    args.channel=3
    class_num = 64
    args.class_num=64
elif args.dataset=='miniimagenet':
    img_size = 84
    args.img_size=84
    channel = 3
    args.channel=3
    class_num = 64
    args.class_num = 64
elif args.dataset=='cub':
    img_size = 84
    args.img_size=84
    channel = 3
    args.channel=3
    class_num = 100
    args.class_num = 100
elif args.dataset == 'mix':
    img_size = 84
    args.img_size = 84
    channel = 3
    args.channel = 3
    class_num = 228
    args.class_num = None

########
if args.dataset == 'mix':
    model_maml=get_model(args=args,set_maml_value=True,arbitrary_input=True)
else:
    model_maml=get_model(args,'train')
model_maml.cuda(device)
if args.dataset!='mix':
    _, _, test_loader = get_dataloader(args)
elif args.dataset=='mix':
    test_loader_cifar, test_loader_mini, test_loader_cub=get_dataloader(args)
optimizer = torch.optim.Adam(params=model_maml.parameters(), lr=args.outer_lr)
criteria = nn.CrossEntropyLoss()
maml=Maml(args)
mamlkd=MamlKD(args)
loss_all = []
acc_all=[]
max_acc_val = None
best_model_maml = None
##################################################################################################################################
timer = Timer()
feature='{}_{}teacher_{}shot_{}warm_{}numkd_{}Giteration_{}Ginterval'.\
    format(args.dataset,args.pre_model_num,args.num_sup_test,args.warmup,args.num_qur_kd,args.generate_iterations,args.generate_interval)

if not os.path.exists('log'):
    os.makedirs('log')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file_path = os.path.join('log', '{}.log'.format(feature))
handler = logging.FileHandler(log_file_path)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
########
if args.dataset != 'mix':
    pretrained_path=os.path.join(args.pretrained_path_prefix,'{}/{}/{}/{}way/model'.format(args.dataset,args.pre_backbone,'meta_train', args.way_pretrain))
    os.makedirs(pretrained_path, exist_ok=True)

##################################################################################################################################
if args.pretrain:
    pretrains(args,args.pre_model_num,device,pretrained_path)
    print('pretrain end!')
    raise NotImplementedError
##################################################################################################################################

nz = 256
generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=channel).cuda()
transform=get_transform(args)
transform_no_toTensor=get_transform_no_toTensor(args)
if os.path.exists('./datapoolkd/' + feature):
    shutil.rmtree('./datapoolkd/' + feature)
    print('remove')
os.makedirs('./datapoolkd/' + feature,exist_ok=True)
max_batch_per_class=20
synthesizer = Synthesizer(args, None, None, generator,
                          nz=nz, num_classes=class_num,
                          img_size=(channel, img_size, img_size),
                          iterations=args.generate_iterations, lr_g=args.Glr,
                          synthesis_batch_size=30,
                          oh=1.0, adv=args.adv, bn=args.bn,
                          save_dir='./datapoolkd/' + feature,
                          transform=transform, transform_no_toTensor=transform_no_toTensor,
                          device=args.gpu, c_abs_list=None, max_batch_per_class=max_batch_per_class)
##################################################################################################################################
maxAcc=None
max_acc_val=-1
max_acc_val_all=[-1,-1,-1]
max_it_all=[-1,-1,-1]
max_pm_all=[-1,-1,-1]
loss_batch, acc_batch = [], []
start_time = time.time()

def reptile_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(p.data - tar_p.data, alpha=67) # , alpha=40

args.num_node_meta_train = args.pre_model_num
generate_num = 0
time_cost = 0

for epoch in range(args.start_id, args.epochs):
    teachers = []
    specifics = []
    transform_no_toTensors = []
    if args.dataset !='mix':
        if args.pre_backbone == 'mix':
            random_pretrain = random.choice(['conv4', 'resnet10', 'resnet18'])
            teacher = get_premodel(args, random_pretrain).cuda(device)
            for id in range(args.num_teacher):
                node_id = random.randint(0, args.num_node_meta_train - 1)
                pretrained_path = os.path.join(args.pretrained_path_prefix,
                                               '{}/{}/{}/{}way/model'.format(args.dataset, random_pretrain,
                                                                             'meta_train',
                                                                             args.way_pretrain))
                teacher_param_specific = torch.load(
                    os.path.join(pretrained_path, 'model_specific_{}.pth'.format(node_id)))
                teacher.load_state_dict(teacher_param_specific['teacher'])
                specific = teacher_param_specific['specific']
                teachers.append(teacher)
                specifics.append(specific)
        else:
            for id in range(args.num_teacher):
                teacher = get_premodel(args).cuda(device)
                node_id = random.randint(0, args.num_node_meta_train - 1)
                teacher_param_specific=torch.load(os.path.join(pretrained_path,'model_specific_{}.pth'.format(node_id)))
                teacher.load_state_dict(teacher_param_specific['teacher'])
                specific=teacher_param_specific['specific']
                teachers.append(teacher)
                specifics.append(specific)
    elif args.dataset =='mix':
        for id in range(args.num_teacher):
            random_dataset = random.choice(['cifar100', 'miniimagenet', 'cub'])
            if random_dataset == 'cifar100':
                args.img_size = 32
            else:
                args.img_size = 84
            teacher = get_premodel(args).cuda(device)
            node_id = random.randint(0, args.num_node_meta_train - 1)
            pretrained_path = os.path.join(args.pretrained_path_prefix,
                                           '{}/{}/{}/{}way/model'.format(random_dataset, args.pre_backbone, 'meta_train',
                                                                         args.way_pretrain))
            teacher_param_specific=torch.load(os.path.join(pretrained_path,'model_specific_{}.pth'.format(node_id)))
            teacher.load_state_dict(teacher_param_specific['teacher'])
            specific=teacher_param_specific['specific']
            if random_dataset == 'cifar100':
                pass
            elif random_dataset == 'miniimagenet':
                specific = [i + 64 for i in specific]
            elif random_dataset == 'cub':
                specific = [i + 128 for i in specific]
            synthesizer.transform = get_transform(args, dataset=random_dataset)
            synthesizer.transform_no_toTensors[id] = get_transform_no_toTensor(args, dataset=random_dataset)
            transform_no_toTensors.append(get_transform_no_toTensor(args, dataset=random_dataset))
            teachers.append(teacher)
            specifics.append(specific)
    # generate for kd data
    synthesizer.teacher = teachers
    synthesizer.c_abs_list = specifics
    if epoch < args.warmup:
        kd_tensors, _ = synthesizer.synthesize(targets=torch.LongTensor((list(range(len(specific)))) * args.num_qur_kd),
                                            student=None, mode='warmup', c_num=len(specific))
    if epoch >= args.warmup:
        kd_tensors, cost = synthesizer.synthesize(targets=torch.LongTensor((list(range(len(specific)))) * args.num_qur_kd),
                                            student=model_maml, mode='support', c_num=len(specific))
        time_cost += cost
        loss_kd = F.kl_div
        generate_num += args.num_qur_kd * len(specific)

        if args.dataset == 'mix':
            kd_datas = torch.stack(kd_tensors, dim=0)
        elif args.dataset != 'mix':
            kd_datas = transform_no_toTensor(torch.stack(kd_tensors, dim=0))
        label_relative = torch.LongTensor((list(range(len(specific)))) * args.num_qur_kd).cuda(device)

        fast_model = model_maml.clone()
        fast_optimizer = torch.optim.Adam(fast_model.parameters(), lr=args.outer_lr)
        grads = []
        for id in range(args.num_teacher):
            if args.dataset == 'mix':
                loss_outer, train_acc = mamlkd.run_outer(model_maml=fast_model, query=transform_no_toTensors[id](kd_datas[id]),
                                                         query_label=label_relative, criteria=loss_kd, device=device,
                                                         teacher=teachers[id], mode='train')
            elif args.dataset != 'mix':
                loss_outer, train_acc = mamlkd.run_outer(model_maml=fast_model, query=kd_datas[id],
                                                         query_label=label_relative, criteria=loss_kd, device=device,
                                                         teacher=teachers[id], mode='train')
            fast_optimizer.zero_grad()
            loss_outer.backward()
            fast_optimizer.step()

        optimizer.zero_grad()
        reptile_grad(model_maml, fast_model)
        optimizer.step()

        #replay
        e_count = 0
        maxAcc = None
        while e_count < args.generate_interval:

            support_data, support_label_abs, query_data, query_label_abs, specific = synthesizer.get_random_task(
                num_w=args.way_train, num_s=args.num_sup_train, num_q=args.num_qur_train)
            support_label = label_abs2relative(specific, support_label_abs).cuda()
            query_label = label_abs2relative(specific, query_label_abs).cuda()
            support, support_label, query, query_label = support_data.cuda(device), support_label.cuda(
                device), query_data.cuda(device), query_label.cuda(device)
            loss_outer, train_acc = maml.run(model_maml=model_maml, support=support, support_label=support_label,
                                             query=query, query_label=query_label, criteria=criteria, device=device,
                                             mode='train')
            loss_batch.append(loss_outer)
            acc_batch.append(train_acc)

            if len(loss_batch) and len(loss_batch) % args.episode_batch == 0:
                loss = torch.stack(loss_batch).sum(0)
                acc = torch.stack(acc_batch).mean()
                loss_batch, acc_batch = [], []
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if maxAcc == None or acc > maxAcc:
                    maxAcc = acc
                    e_count = 0
                else:
                    e_count = e_count + 1

    #val
    if epoch > args.warmup and epoch % args.val_interval == 0:
        acc_val = []
        if args.dataset != 'mix':
            for test_batch in test_loader:
                data, g_label = test_batch[0].cuda(device), test_batch[1].cuda(device)
                support, support_label_relative, query, query_label_relative = data2supportquery(args, 'test', data)
                _, acc = maml.run(model_maml=model_maml, support=support, support_label=support_label_relative,
                                  query=query, query_label=query_label_relative, criteria=criteria, device=device,
                                  mode='test')
                acc_val.append(acc)
            del _

            acc_val, pm = compute_confidence_interval(acc_val)
            if acc_val > max_acc_val:
                max_acc_val = acc_val
                max_it = epoch
                max_pm = pm
        if args.dataset == 'mix':
            test_loader_all = [test_loader_cifar, test_loader_mini, test_loader_cub]
            acc_val_all = [[], [], []]
            for i, test_loader in enumerate(test_loader_all):
                for test_batch in test_loader:
                    data, g_label = test_batch[0].cuda(device), test_batch[1].cuda(device)
                    support, support_label_relative, query, query_label_relative = data2supportquery(args, 'test', data)
                    _, acc = maml.run(model_maml=model_maml, support=support, support_label=support_label_relative,
                                      query=query, query_label=query_label_relative, criteria=criteria,
                                      device=device,
                                      mode='test')
                    acc_val_all[i].append(acc)
                acc_val, pm = compute_confidence_interval(acc_val_all[i])
                acc_val_all[i] = acc_val
            acc_val = sum(acc_val_all) / len(acc_val_all)
            if acc_val > max_acc_val:
                max_acc_val = acc_val
                max_it = epoch
                max_pm = pm
            for i in range(3):
                if acc_val_all[i] > max_acc_val_all[i]:
                    max_acc_val_all[i] = acc_val_all[i]
                    max_it_all[i] = epoch
                    max_pm_all[i] = pm

        logger.info('task_id:' + str(epoch) + ' test acc: ' + str(acc_val) + '+-'+ str(pm))
        logger.info(str(max_it) + ' best test acc: ' + str(max_acc_val) + '+-' + str(max_pm))
        logger.info('ETA:{}/{}'.format(
            timer.measure(),
            timer.measure((epoch) / (args.epochs))))
        logger.info("Generation Cost: %1.3f" % (time_cost / 3600.))
        print('generate:', generate_num, 'images')
        print(epoch, 'test acc:', acc_val, '+-', pm)
        print(max_it, 'best test acc:', max_acc_val, '+-', max_pm)
        print('ETA:{}/{}'.format(
            timer.measure(),
            timer.measure((epoch) / (args.epochs)))
        )
        print("Generation Cost: %1.3f" % (time_cost / 3600.))