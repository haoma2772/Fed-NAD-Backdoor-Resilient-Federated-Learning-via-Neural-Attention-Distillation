import torch
from utils.grad_utility import grad_clean_train_model
import copy
import wandb
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from utils.grad_utility import grad_clean_train_model
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from utils.plugin import  get_feas_by_hook
from utils.plugin import get_NAD_layers
from utils import base_config
from utils.test_model import test_model
from utils.utility import  get_poison_transform
from utils.utility import vectorize_model
from utils.plugin import AT

def FedNAD(model_list,  self_model, data_loader, device,config_list, client_id, global_round,
             record=None, with_test=False, test_dataset_loader=None, self_test_loader=None, with_save=False):
    
    student: torch.nn.Module = model_list[client_id].to(device)
    # student: torch.nn.Module= average_models(model_list).to(device)
    teacher: torch.nn.Module = self_model.to(device)

    print('this is teacher local train')
    [teacher,
        record[client_id], 
        tea_weight_diff] = grad_clean_train_model(self_flag=True,
                                                    model=teacher, 
                                                    data_loader=data_loader,
                                                    device=device,
                                                    config_list=config_list, 
                                                    client_id=client_id,
                                                    global_round=global_round,
                                                    record=record,
                                                    with_test=True, 
                                                    test_dataset_loader=self_test_loader,
                                                    with_save=False)
    

    teacher.eval()

    lr = config_list['device']['learning_rate']
    num_epochs = config_list['device']['epoch']
    # tea_criterion = nn.KLDivLoss(reduction='batchmean')  # Kullback-Leibler Divergence Loss
    tea_criterion  = AT(config_list['defense']['NAD_p'])

    stu_criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    layer_names, at_weight_list = get_NAD_layers(config_list['model']['name'])
    tea_hook_instances = get_feas_by_hook(teacher, layer_names)
    stu_hook_instances = get_feas_by_hook(student, layer_names)

    pre_model_weight = vectorize_model(student)

    print('this is the student train')
    for ep in range(num_epochs):

        student.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_at_loss = 0.0
        for inputs, labels in tqdm(data_loader, total=len(data_loader), desc=f"Training epoch {ep + 1}/{num_epochs}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                at_loss = 0.0
                with torch.no_grad():
                    tea_output = teacher(inputs)
                stu_output = student(inputs)
                for idx in range(len(layer_names)):
                    at_loss =  at_loss + tea_criterion(stu_hook_instances[idx].fea, tea_hook_instances[idx].fea.detach()) * at_weight_list[idx]

                cls_loss = stu_criterion(stu_output, labels)

                running_cls_loss =  running_cls_loss + cls_loss
                running_at_loss = running_at_loss + at_loss

                loss = cls_loss + at_loss
                running_loss += loss.item()

                loss.backward()
                optimizer.step()

        wandb.log({'All Loss of clean client#{}'.format(client_id): running_loss / len(data_loader), 'epoch': global_round * num_epochs + ep, })
        wandb.log({'AT Loss of clean client#{}'.format(client_id): running_at_loss / len(data_loader), 'epoch': global_round * num_epochs + ep, })
        wandb.log({'CLS Loss of clean client#{}'.format(client_id): running_cls_loss / len(data_loader), 'epoch': global_round * num_epochs + ep, })
        if record is not None:
                record['loss'].append(running_loss / len(data_loader))

        if with_test:
            # need test
            if config_list['backdoor_paras']['backdoor_name'] == 'scaling_attack':
                poison_type = 'badnet'
            elif config_list['backdoor_paras']['backdoor_name'] == 'neurotoxin':
                poison_type = 'badnet'
            else:
                poison_type = config_list['backdoor_paras']['backdoor_name']

            dataset_name = config_list['dataset']['name']
            target_class = base_config.target_class[dataset_name]

            alpha = config_list['backdoor_paras']['alpha']
            trigger = config_list['backdoor_paras']['trigger']
            if trigger == 'None':
                trigger = base_config.trigger_default[dataset_name][poison_type]

            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(224, antialias=True),

            ])
            if dataset_name == 'cifar10':
                num_classes = 10
            elif dataset_name == 'gtsrb':
                num_classes = 43

            elif dataset_name == 'tiny_imagenet':
                num_classes = 200
            elif dataset_name == 'mnist':
                num_classes = 10

            if poison_type == 'badnet_all_to_all':
                all_to_all = True
            else:
                if poison_type == 'TaCT':
                    source_class = base_config.source_class
                    cover_classes = base_config.cover_classes

                all_to_all = False

            poison_transform = get_poison_transform(poison_type=poison_type, dataset_name=dataset_name,
                                                    target_class=target_class
                                                    , source_class=None, cover_classes=None,
                                                    is_normalized_input=True, trigger_transform=trigger_transform,
                                                    alpha=alpha, trigger_name=trigger,device=device)

            CA, ASR = test_model(model=student, data_loader=test_dataset_loader, config_list=config_list, path=None,
                                 poison_test=True,
                                 poison_transform=poison_transform, num_classes=num_classes,
                                 source_classes=None, all_to_all=all_to_all
                                 )

            wandb.log({'Test_accuracy of clean client#{}'.format(client_id): CA, 'epoch': global_round * num_epochs + ep})
            wandb.log({'Attack_success_rate of clean client#{}'.format(client_id): ASR, 'epoch':global_round * num_epochs + ep})

            if record is not None:
                record['test_acc'].append(CA)
                record['attack_acc'].append(ASR)

    cur_model_weight = vectorize_model(student)
    stu_weight_diff = (cur_model_weight - pre_model_weight).cpu()
    print('Finish distillation!')
    return student, teacher, stu_weight_diff, record



















def NormClip(model, norm_bound, stddev,  data_loader,device,config_list, client_id, global_round,
             record=None, with_test=False, test_dataset_loader=None,with_save=False):



    [return_model,
     return_record, 
     weight_diff] = grad_clean_train_model(self_flag=False,
                                                    model=model, 
                                                    data_loader=data_loader,
                                                    device=device,
                                                    config_list=config_list, 
                                                    client_id=client_id,
                                                    global_round=global_round,
                                                    record=record,
                                                    with_test=with_test, 
                                                    test_dataset_loader=test_dataset_loader,
                                                    with_save=with_save)



    weight_diff_norm = torch.norm(weight_diff).item()
    clipped_weight_diff = weight_diff/max(1, weight_diff_norm/norm_bound)

    dp_weight = clipped_weight_diff + torch.randn(
        weight_diff.size(), device=weight_diff.device) * stddev
    
    

    return return_model, return_record, dp_weight










def average_models(model_list: list[torch.nn.Module], weight_list=None):

    if not model_list:
        raise ValueError("模型列表为空")

    length = len(model_list)
    beta = 1.0 / length
    if weight_list is None:
        weight_list = torch.ones(length, dtype=torch.float) * beta
    aggregated_params = {}

    for id, model in enumerate(model_list):
        model_params = model.state_dict()
        for name, param in model_params.items():
            if name not in aggregated_params:
                # 如果参数名称在聚合字典中不存在，初始化为零
                aggregated_params[name] = torch.zeros_like(param.data, dtype=torch.float)
            # 累积参数值
            aggregated_params[name] += weight_list[id] * param.data
            
    averaged_model = copy.deepcopy(model_list[0])
    averaged_model.load_state_dict(aggregated_params)

    return averaged_model






def geometric_median(points, method='auto', options={}):
    """
    Calculates the geometric median of an array of points.
    method specifies which algorithm to use:
        * 'auto' -- uses a heuristic to pick an algorithm
        * 'minimize' -- scipy.optimize the sum of distances
        * 'weiszfeld' -- Weiszfeld's algorithm
    """

    points = np.asarray(points)

    if len(points.shape) == 1:
        # geometric_median((0, 0)) has too much potential for error.
        # Did the user intend a single 2D point or two scalars?
        # Use np.median if you meant the latter.
        raise ValueError("Expected 2D array")

    if method == 'auto':
        if points.shape[1] > 2:
            # weiszfeld tends to converge faster in higher dimensions
            method = 'weiszfeld'
        else:
            method = 'minimize'

    return _methods[method](points, options)


def minimize_method(points, options={}):
    """
    Geometric median as a convex optimization problem.
    """

    # objective function
    def aggregate_distance(x):
        return cdist([x], points).sum()

    # initial guess: centroid
    centroid = points.mean(axis=0)

    optimize_result = minimize(aggregate_distance, centroid, method='COBYLA')

    return optimize_result.x


def weiszfeld_method(points, options={}):
    """
    Weiszfeld's algorithm as described on Wikipedia.
    """

    default_options = {'maxiter': 1000, 'tol': 1e-7}
    default_options.update(options)
    options = default_options

    def distance_func(x):
        return cdist([x], points)

    # initial guess: centroid
    guess = points.mean(axis=0)

    iters = 0

    while iters < options['maxiter']:
        distances = distance_func(guess).T

        # catch divide by zero
        # TODO: Wikipedia cites how to deal with distance 0
        distances = np.where(distances == 0, 1, distances)

        guess_next = (points/distances).sum(axis=0) / (1./distances).sum(axis=0)

        guess_movement = np.sqrt(((guess - guess_next)**2).sum())

        guess = guess_next

        if guess_movement <= options['tol']:
            break

        iters += 1

    return guess


_methods = {
    'minimize': minimize_method,
    'weiszfeld': weiszfeld_method,
}






def Median(model_list: list[torch.nn.Module], config, weight_list=None):



    num_clients = len(model_list)

    if not model_list:
        raise ValueError("模型列表为空")

    zeros_model = copy.deepcopy(model_list[0])

    for param in zeros_model.parameters():
        torch.nn.init.zeros_(param)

    w_zero = zeros_model.state_dict()
    # weight list
    model_weight: list = [copy.deepcopy(model_list[i].state_dict()) for i in range(num_clients)]

    aggregated_params = copy.deepcopy(w_zero)
    for k in w_zero.keys():
        tensors_list = [torch.flatten(model_weight[i][k]) for i in range(num_clients)]
        stacked_tensor = torch.stack(tensors_list)
        median_tensor, _ = torch.median(stacked_tensor, dim=0)
        aggregated_params[k] = torch.reshape(median_tensor, w_zero[k].shape)

    averaged_model = copy.deepcopy(zeros_model)
    averaged_model.load_state_dict(aggregated_params)

    return averaged_model






def trimmed_mean_models(model_list: list[torch.nn.Module], config, weight_list=None):


    num_clients = config['federated_learning']['client_number']
    num_attackers: int = int(config['federated_learning']['backdoor_rate'] * num_clients)

    sta_index = num_attackers
    end_index = num_clients - num_attackers


    if not model_list:
        raise ValueError("模型列表为空")
    if sta_index >= end_index:
        raise ValueError("The number of attacker not satisfy requirement")
    
    zeros_net = copy.deepcopy(model_list[0])

    for param in zeros_net.model.parameters():
        torch.nn.init.zeros_(param)

    w_zero = zeros_net.model.state_dict()
    # weight list
    model_weight: list = [copy.deepcopy(model_list[i].model.state_dict()) for i in range(num_clients)]

    aggregated_params = copy.deepcopy(w_zero)
    for k in w_zero.keys():
        tensors_list = [torch.flatten(model_weight[i][k]) for i in range(num_clients)]
        stacked_tensor = torch.stack(tensors_list)
        sort_tensor, indices = torch.sort(stacked_tensor, dim=0)

        avg_tensor = sort_tensor[sta_index:end_index]
        avg_tensor = torch.mean(avg_tensor.float(), dim=0)
        aggregated_params[k] = torch.reshape(avg_tensor, w_zero[k].shape)

    returned_model = copy.deepcopy(zeros_net)
    returned_model.load_state_dict(aggregated_params)

    return returned_model











import torch

def euclidean_distance(model1, model2):
    distance = 0.0
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        distance += torch.norm(param1 - param2, p='fro')
    return distance.item()


def compute_scores(distances, i, n, f):


    s = [distances[j][i] for j in range(i)] + [
            distances[i][j] for j in range(i + 1, n)
        ]
        
        # 对列表 s 进行排序，并选择前 n - f - 2 个最小的距离的平方
    _s = sorted(s)[: n - f - 2]
        
        # 返回选定距离的平方之和作为节点 i 的 Krum 距离得分
    return sum(_s)


def krum_defense(model_list: list[torch.nn.Module], config, weight_list=None,k=1):
    # for each client select n-f neighbor for compute scores and select k for avg
    # n < 2*f + 2
    # server_list 是模型列表
    num_clients = len(model_list)
    num_attackers = int(config['federated_learning']['backdoor_rate'] * num_clients)

    distances = {}
    for i in range(num_clients-1):
        distances[i] = {}
        for j in range(i + 1, num_clients):
            distances[i][j] = euclidean_distance(model_list[i], model_list[j])
    

    scores = [(i, compute_scores(distances, i, num_clients, num_attackers)) for i in range(num_clients)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    top_k_indices = list(map(lambda x: x[0], sorted_scores))[:k]

    # local_paras = avg_defense(server_list[top_m_indices], )

    return [model_list[i] for i in top_k_indices]
