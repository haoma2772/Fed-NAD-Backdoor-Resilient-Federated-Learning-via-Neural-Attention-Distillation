import yaml
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import numpy as np

import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
from PIL import Image
from torch.utils.data import random_split

import torch
import utils.base_config as base_config
from torchvision import transforms
from PIL import Image
import os


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_dataset(config, trained=True):
    dataset_name = config['dataset']['name']
    data_dir = 'data'

    resize_transform = transforms.Resize((224, 224))

    if dataset_name == 'mnist':
        transform = transforms.Compose([
            resize_transform, 
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = torchvision.datasets.MNIST(root=data_dir, train=trained, transform=transform, download=True)
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            resize_transform, 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=trained, transform=transform, download=True)
    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            resize_transform,  
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = torchvision.datasets.CIFAR100(root=data_dir, train=trained, transform=transform, download=True)
    elif dataset_name == 'imagenet1k':
        transform = transforms.Compose([
            resize_transform, 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))

    return dataset

def complete_graph(n):
 
    adj_matrix = np.ones((n, n))
  
    np.fill_diagonal(adj_matrix, 0)
    return adj_matrix

def cycle_graph(n):

    adj_matrix = np.zeros((n, n))


    for i in range(n):
        adj_matrix[i, (i - 1) % n] = 1 
        adj_matrix[i, (i + 1) % n] = 1 

    return adj_matrix


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    The Dirichlet distribution of argument alpha divides the set of sample indexes into subsets of n_clients
    '''
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]


    client_idcs = [[] for _ in range(n_clients)]

    for c, fracs in zip(class_idcs, label_distribution):

        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs


def dirichlet_distribution(dataset, config):
    total_samples = len(dataset)
    iid = config['federated_learning']['iid']
    n_clients = config['federated_learning']['client_number']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    if iid:
        samples_per_client = total_samples // n_clients
  
        indices = list(range(total_samples))
        np.random.shuffle(indices)
        subsets = [indices[i*samples_per_client: (i+1)*samples_per_client] for i in range(0, n_clients)]
    else:
        DIRICHLET_ALPHA = config['federated_learning']['dirichlet_rate']
        input_sz, num_cls = dataset.data[0].shape[0], len(dataset.classes)

        train_labels = np.array(dataset.targets)

        subsets = dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=n_clients)

    local_dataset_loader = []
    for subset_indices in subsets:


        # sampler = data.sampler.SubsetRandomSampler(subset)
        # ubset_data = [dataset[i] for i in subset]
        # data_loader = data.DataLoader(subset_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        # local_dataset.append(data_loader)
        subset = Subset(dataset, subset_indices)
        local_loader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        local_dataset_loader.append(local_loader)
    return local_dataset_loader


def print_dataset_label(data_loader):
    unique_labels = set() 

    for batch in data_loader:
        _, labels = batch 
        unique_labels.update(labels.tolist()) 

    print("Unique Labels:", unique_labels)


def visualize_class_proportions_bar(dataloader):

    class_counts = defaultdict(int)
    total_samples = len(dataloader.dataset)


    for _, label in dataloader.dataset:
        class_counts[label] += 1


    class_proportions = {label: count / total_samples for label, count in class_counts.items()}


    plt.figure(figsize=(10, 6))
    plt.bar(class_proportions.keys(), class_proportions.values())
    plt.xlabel('Class Label')
    plt.ylabel('Proportion')
    plt.title('Proportion of Each Class in the Dataloader')
    plt.xticks(rotation=45)
    plt.show()


def visualize_class_proportions_pie(dataloader):

    class_counts = defaultdict(int)
    total_samples = len(dataloader.dataset)


    for _, label in dataloader.dataset:
        class_counts[label] += 1

    class_proportions = {label: count / total_samples for label, count in class_counts.items()}


    labels = class_proportions.keys()
    proportions = class_proportions.values()

  
    plt.figure(figsize=(8, 8))
    plt.pie(proportions, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal') 
    plt.title('Proportion of Each Class in the Dataloader')

    # 显示饼状图
    plt.show()



def set_seed(config):
    seed = config['general']['random_seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def split_data_loader(dataloader, train_ratio, config, random_seed=None):
    dataset = dataloader.dataset
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size

    if random_seed is not None:
        torch.manual_seed(random_seed)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return train_dataloader, test_dataloader

def get_neighbors(node_id, adjacency_matrix):

    neighbors = []
    for i in range(len(adjacency_matrix[node_id])):
        if adjacency_matrix[node_id][i] == 1 and i != node_id:
            neighbors.append(i)

    return neighbors


def restricted_find_top_k_rewards(reward_list, select_list, k):

    indexed_values = [(index, value) for index, value in enumerate(reward_list) if index in select_list]


    indexed_values.sort(key=lambda x: x[1], reverse=True)

    restricted_top_k = [index for index, _ in indexed_values[:k]]

    return restricted_top_k


def vectorize_model(model: torch.nn.Module):
    # return one dim vector
    return torch.cat([p.clone().detach().view(-1) for p in model.parameters()])



def vectorize_model_dict(model: torch.nn.Module):
    # return a dict which
    # name: vector
    weight_accumulator = dict()
    for name, data in model.named_parameters():

        weight_accumulator[name] = data

    return weight_accumulator

def calculate_norm(model_weight_dict, p=2):
    # return a dict
    # naem l2, value
    l2_norm_dict = {}
    for layer_name, weights in model_weight_dict.items():
        l2_norm = torch.norm(weights,p=2)
        l2_norm_dict[layer_name] = l2_norm.item()
    return l2_norm_dict

def select_top_k_norms(l2_norm_dict:dict, top_k:int):
    # just return a list
    # each element is a layer name
    sorted_norms = sorted(l2_norm_dict.items(), key=lambda x: x[1], reverse=True)
    top_k_norms_layers = [layer_name for layer_name, _ in sorted_norms[:top_k]]
    return top_k_norms_layers


def load_model_weight(model, weight):
    index_bias = 0
    for p_index, p in enumerate(model.parameters()):
        # Check if there are enough elements in the weight array
        if index_bias + p.numel() > len(weight):
            raise ValueError("Not enough elements in weight array to load into model")
        
        # Load the weight into the model parameter
        p.data = weight[index_bias:index_bias+p.numel()].view(p.size())
        index_bias += p.numel()


def load_model_weight_diff(net, weight_diff, global_weight):
    """
    load rule: w_t + clipped(w^{local}_t - w_t)
    """
    listed_global_weight = list(global_weight.parameters())
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data =  weight_diff[index_bias:index_bias+p.numel()].view(p.size()) + listed_global_weight[p_index]
        index_bias += p.numel()




def load_model_weight_diff(model, weight_diff, global_weight):
    """
    load rule: w_t + clipped(w^{local}_t - w_t)
    """
    
    listed_global_weight = list(global_weight.parameters())
    index_bias = 0
    for p_index, p in enumerate(model.parameters()):
        p.data =  weight_diff[index_bias:index_bias+p.numel()].view(p.size()) + listed_global_weight[p_index]
        index_bias += p.numel()





def get_poison_transform(poison_type, dataset_name, target_class, source_class=1, cover_classes=[5, 7],
                         is_normalized_input=False, trigger_transform=None,
                         alpha=0.2, trigger_name=None, device='cpu'):
    # source class will be used for TaCT poison

    if trigger_name is None:
        if dataset_name != 'tiny_imagenet':
            trigger_name = base_config.trigger_default[dataset_name][poison_type]
        else:
            if poison_type == 'badnet':
                trigger_name = 'badnet_high_res.png'
            else:
                raise NotImplementedError('%s not implemented for tiny_imagenet' % poison_type)

    # if dataset_name in ['gtsrb', 'cifar10', 'cifar100']:
    #     img_size = 32
    # elif dataset_name == 'tiny_imagenet' or dataset_name == 'imagenet':
    #     img_size = 224
    # else:
    #     raise NotImplementedError('<Undefined> Dataset = %s' % dataset_name)
    img_size = 224
    
    resize_transform = transforms.Resize(224, antialias=True)
    if dataset_name == "cifar10":
        img_size = 224
        num_classes = 10
        data_transform = transforms.Compose([

            transforms.ToTensor(),
            resize_transform,
        ])

        normalize_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        denormalize_transform = transforms.Normalize(
            mean=(-0.4940, -0.4850, -0.4504),
            std=(1/0.2467, 1/0.2429, 1/0.2616)
        )
    elif dataset_name == 'mnist':
        img_size = 224
        num_classes = 10
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            resize_transform,
        ])
        normalize_transform = transforms.Compose([
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
        denormalize_transform = transforms.Compose([
                transforms.Normalize(mean=[-0.5, -0.5, -0.5],std=[1/0.5, 1/0.5, 1/0.5])
            ])

    elif dataset_name == 'emnist':
        img_size = 224
        num_classes = 47
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            resize_transform,
        ])
        normalize_transform = transforms.Compose([
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
        denormalize_transform = transforms.Compose([
                transforms.Normalize(mean=[-0.5, -0.5, -0.5],std=[1/0.5, 1/0.5, 1/0.5])
            ])
        
    elif dataset_name == 'gtsrb':
        img_size = 224
        num_classes = 43
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            resize_transform,
        ])
        normalize_transform = transforms.Compose([
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        denormalize_transform = transforms.Compose([
                transforms.Normalize([-0.485 / 0.229, -0.456/0.224, -0.406/0.225],[1/0.229, 1/0.224, 1/0.225])
            ])

    elif dataset_name == 'tiny_imagenet':

        img_size = 224
        num_classes = 10
        normalize_transform =transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        denormalize_transform = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            resize_transform,
        ])
    else:
        raise NotImplementedError('Undefined Dataset')

    poison_transform = None
    trigger = None
    trigger_mask = None

    if poison_type in ['basic', 'badnet', 'blend', 'clean_label', 'refool',
                       'adaptive_blend', 'adaptive_patch', 'adaptive_k_way',
                       'SIG', 'TaCT', 'WaNet', 'SleeperAgent', 'none',
                       'badnet_all_to_all', 'trojan', 'SRA', 'bpp', 'DBA']:

        if trigger_transform is None:
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(224, antialias=True),
            ])

        # trigger mask transform; remove `Normalize`!
        trigger_mask_transform_list = []
        for t in trigger_transform.transforms:
            if "Normalize" not in t.__class__.__name__:
                trigger_mask_transform_list.append(t)
        trigger_mask_transform = transforms.Compose(trigger_mask_transform_list)

        if trigger_name != "none":
            # none for SIG
            trigger_path = os.path.join(base_config.triggers_dir, trigger_name)
            trigger = Image.open(trigger_path).convert("RGB")
            trigger = trigger_transform(trigger)

            trigger_mask_path = os.path.join(base_config.triggers_dir, 'mask_%s' % trigger_name)
            if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)

                # print('trigger_mask_path:', trigger_mask_path)
                trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                # trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
                trigger_mask = trigger_transform(trigger_mask)
            else:  # by default, all black pixels are masked with 0's
                # print('No trigger mask found! By default masking all black pixels...')
                trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0),
                                                trigger[2] > 0).float()
                # trigger_mask = trigger_transform(trigger_mask)

        normalizer = transforms.Compose([normalize_transform])
        denormalizer = transforms.Compose([denormalize_transform])

        if poison_type == 'basic':
            from poison_tool_box import basic
            poison_transform = basic.poison_transform(img_size=img_size, trigger_mark=trigger,
                                                      trigger_mask=trigger_mask,
                                                      target_class=target_class, alpha=alpha)

        elif poison_type == 'badnet':
            from poison_tool_box import badnet
            poison_transform = badnet.poison_transform(img_size=img_size, trigger_mark=trigger,
                                                       trigger_mask=trigger_mask, target_class=target_class)

        elif poison_type == 'badnet_all_to_all':
            from poison_tool_box import badnet_all_to_all
            poison_transform = badnet_all_to_all.poison_transform(img_size=img_size, trigger_mark=trigger,
                                                                  trigger_mask=trigger_mask, num_classes=num_classes)

        elif poison_type == 'trojan':
            from poison_tool_box import trojan
            poison_transform = trojan.poison_transform(img_size=img_size, trigger_mark=trigger,
                                                       trigger_mask=trigger_mask, target_class=target_class)

        elif poison_type == 'blend':
            from poison_tool_box import blend
            poison_transform = blend.poison_transform(img_size=img_size, trigger=trigger,
                                                      target_class=target_class, alpha=alpha)
        elif poison_type == 'DBA':
            poison_patterns= []
            for i in range(0,base_config.DBA_poison_patterns_num):
                poison_patterns = poison_patterns + base_config.DBA_poison_patterns[dataset_name][str(i) + '_poison_pattern']
            # poison_patterns= base_config.DBA_poison_patterns[dataset_name][str(0) + '_poison_pattern']

            from poison_tool_box import DBA
            poison_transform = DBA.poison_transform(img_size=img_size,
                                                    poison_patterns= poison_patterns,
                                                      target_class=target_class, alpha=alpha)

        elif poison_type == 'refool':
            from poison_tool_box import refool
            poison_transform = refool.poison_transform(img_size=img_size, target_class=target_class,
                                                       denormalizer=denormalizer, normalizer=normalizer,
                                                       max_image_size=224)

        elif poison_type == 'clean_label':
            from poison_tool_box import clean_label
            poison_transform = clean_label.poison_transform(img_size=img_size, trigger_mark=trigger,
                                                            trigger_mask=trigger_mask,
                                                            target_class=target_class)

        elif poison_type == 'WaNet':
            s = 0.5
            k = 4
            grid_rescale = 1
            ins = torch.rand(1, 2, k, k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            noise_grid = (
                torch.nn.functional.upsample(ins, size=img_size, mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
            )
            array1d = torch.linspace(-1, 1, steps=img_size)
            x, y = torch.meshgrid(array1d, array1d)
            identity_grid = torch.stack((y, x), 2)[None, ...]

            from poison_tool_box import WaNet
            poison_transform = WaNet.poison_transform(img_size=img_size, denormalizer=denormalizer,
                                                      identity_grid=identity_grid, noise_grid=noise_grid, s=s, k=k,
                                                      grid_rescale=grid_rescale, normalizer=normalizer,
                                                      target_class=target_class)

        elif poison_type == 'adaptive_blend':

            from poison_tool_box import adaptive_blend
            poison_transform = adaptive_blend.poison_transform(img_size=img_size, trigger=trigger,
                                                               target_class=target_class, alpha=alpha)

        elif poison_type == 'adaptive_patch':
            from poison_tool_box import adaptive_patch
            poison_transform = adaptive_patch.poison_transform(img_size=img_size, test_trigger_names=
            base_config.adaptive_patch_test_trigger_names[dataset_name],
                                                               test_alphas=
                                                               base_config.adaptive_patch_test_trigger_alphas[
                                                                   dataset_name], target_class=target_class,
                                                               denormalizer=denormalizer, normalizer=normalizer,
                                                                device=device )

        elif poison_type == 'adaptive_k_way':
            from poison_tool_box import adaptive_k_way
            poison_transform = adaptive_k_way.poison_transform(img_size=img_size, target_class=target_class,
                                                               denormalizer=denormalizer, normalizer=normalizer, )

        elif poison_type == 'SIG':

            from poison_tool_box import SIG
            poison_transform = SIG.poison_transform(img_size=img_size, denormalizer=denormalizer, normalizer=normalizer,
                                                    target_class=target_class, delta=30 / 255, f=6,
                                                    has_normalized=is_normalized_input)

        elif poison_type == 'TaCT':
            from poison_tool_box import TaCT
            poison_transform = TaCT.poison_transform(img_size=img_size, trigger=trigger, mask=trigger_mask,
                                                     target_class=target_class)

        elif poison_type == 'SleeperAgent':
            from poison_tool_box import SleeperAgent
            poison_transform = SleeperAgent.poison_transform(random_patch=False, img_size=img_size,
                                                             target_class=target_class, denormalizer=denormalizer,
                                                             normalizer=normalizer)

        return poison_transform


    elif poison_type == 'dynamic':

        normalizer = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
        denormalizer = transforms.Compose([transforms.Normalize(mean=(-0.5 / 0.5), std=(1.0 / 0.5)), ])

        if dataset_name == 'cifar10':
            channel_init = 224
            steps = 3
            input_channel = 3
            ckpt_path = './models/all2one_cifar10_ckpt.pth.tar'

            require_normalization = True

        elif dataset_name == 'gtsrb':
            # the situation for gtsrb is inverese
            # the original implementation of generator does not require normalization
            channel_init = 224
            steps = 3
            input_channel = 3
            ckpt_path = './models/all2one_gtsrb_ckpt.pth.tar'

            require_normalization = False

        else:
            raise Exception("Invalid Dataset")

        if not os.path.exists(ckpt_path):
            raise NotImplementedError(
                '[Dynamic Attack] Download pretrained generator first: https://github.com/VinAIResearch/input-aware-backdoor-attack-release')

        from poison_tool_box import dynamic
        poison_transform = dynamic.poison_transform(ckpt_path=ckpt_path, channel_init=channel_init, steps=steps,
                                                    input_channel=input_channel, normalizer=normalizer,
                                                    denormalizer=denormalizer, target_class=target_class,
                                                    has_normalized=is_normalized_input,
                                                    require_normalization=require_normalization)
        return poison_transform



    else:
        raise NotImplementedError('<Undefined> Poison_Type = %s' % poison_type)
    







    
def init_masks(params:dict, sparsities):
    # params 是一个dict

    masks = {}
    for name in params:
        masks[name] = torch.zeros_like(params[name])
        dense_numel = int((1 - sparsities[name]) * torch.numel(masks[name]))
        if dense_numel > 0:
            temp = masks[name].view(-1)
            perm = torch.randperm(len(temp))
            perm = perm[:dense_numel]
            temp[perm] = 1
        masks[name] = masks[name].to("cpu")
    return masks


def calculate_sparsities(config, params, tabu=[], distribution="ERK"):
    spasities = {}
    if distribution == "uniform":
        for name in params:
            if name not in tabu:
                spasities[name] = 1 - config['defense']['Lockdown_dense_ratio']
            else:
                spasities[name] = 0
    elif distribution == "ERK":
        # logging.info('initialize by ERK')
        total_params = 0
        for name in params:
            total_params += params[name].numel()
        is_epsilon_valid = False
        # # The following loop will terminate worst case when all masks are in the
        # custom_sparsity_map. This should probably never happen though, since once
        # we have a single variable or more with the same constant, we have a valid
        # epsilon. Note that for each iteration we add at least one variable to the
        # custom_sparsity_map and therefore this while loop should terminate.
        dense_layers = set()

        density = config['defense']['Lockdown_dense_ratio']
        while not is_epsilon_valid:
            # We will start with all layers and try to find right epsilon. However if
            # any probablity exceeds 1, we will make that layer dense and repeat the
            # process (finding epsilon) with the non-dense layers.
            # We want the total number of connections to be the same. Let say we have
            # for layers with N_1, ..., N_4 parameters each. Let say after some
            # iterations probability of some dense layers (3, 4) exceeded 1 and
            # therefore we added them to the dense_layers set. Those layers will not
            # scale with erdos_renyi, however we need to count them so that target
            # paratemeter count is achieved. See below.
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name in params:
                if name in tabu or "running" in name or "track" in name :
                    dense_layers.add(name)
                n_param = np.prod(params[name].shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (
                                                      np.sum(params[name].shape) / np.prod(params[name].shape)
                                              ) ** 1
                    divisor += raw_probabilities[name] * n_param
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name in params:
            if name in dense_layers:
                spasities[name] = 0
            else:
                spasities[name] = (1 - epsilon * raw_probabilities[name])
    return spasities




def get_dataset_num_class(dataset_name):
    if dataset_name == 'cifar10':
        return 10
    elif dataset_name == 'mnist':
        return 10
    elif dataset_name == 'cifar100':
        return 100
    elif dataset_name == 'gtsrb':
        return 43
    elif dataset_name == 'emnist':
        return 47
    elif dataset_name == 'tiny_imagenet':
        return 200