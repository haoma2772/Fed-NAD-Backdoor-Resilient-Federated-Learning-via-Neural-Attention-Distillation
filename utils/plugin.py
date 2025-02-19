import torch
import numpy as np
from torch.utils.data import random_split, Subset
import torchvision.transforms as transforms
from utils.inject_backdoor import create_poisoned_set
import copy
import pickle
import os
import torchvision
from torch.utils.data import Dataset
from utils.utility import vectorize_model
from PIL import Image


class CustomGTSRBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.images = []
        self.labels = []
        self.training_dir = os.path.join(data_dir,'Training')
        # self.test_dir = os.path.join(data_dir,'Final_Test')

        for class_id in range(43):
            class_dir = os.path.join(self.training_dir, f'{class_id:05d}')

            for file_name in os.listdir(class_dir):
                if file_name.endswith('.ppm'):
                    image_path = os.path.join(class_dir, file_name)
                    self.image_paths.append(image_path)
                    self.labels.append(class_id)
                    #image = Image.open(image_path).convert('RGB')
                    #image = transform(image)
                    #self.images.append(image)




    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # image = Image.open(image_path).convert('RGB')
        # image = self.transform(image)
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform != None:
            image = self.transform(image)

        return image, label
def load_dataset(dataset_name, data_dir, trained=True):

    resize_transform = transforms.Resize((224, 224), antialias=True)

    if dataset_name == 'mnist':
        transform = transforms.Compose([
            resize_transform ,
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),

            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = torchvision.datasets.MNIST(root=data_dir, train=trained, transform=transform, download=True)
        # test_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transform, download=True)

        # combined_dataset = train_dataset
    elif dataset_name == 'emnist':
            transform = transforms.Compose([
                resize_transform ,
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),

                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            train_dataset = torchvision.datasets.EMNIST(root=data_dir, split='balanced', train=trained, download=False, transform=transform)
            # test_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transform, download=True)

            # combined_dataset = train_dataset
    elif dataset_name == 'cifar10':
        if trained:
            transform = transforms.Compose([
            resize_transform ,
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # 训练集
        ])
        else:
            transform = transforms.Compose([
            resize_transform ,
            transforms.ToTensor(),
            transforms.Normalize((0.4940, 0.4850, 0.4504), (0.2467, 0.2429, 0.2616))  # 测试集
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=trained, transform=transform, download=True)
        # test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)
        # combined_dataset = train_dataset

    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
             resize_transform ,
            transforms.ToTensor(),

            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=trained, transform=transform, download=True)
        # test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, transform=transform, download=True)
        # combined_dataset = train_dataset + test_dataset

    elif dataset_name == 'imagenet1k':

        transform = transforms.Compose([
             resize_transform ,
            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
        # test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
        # combined_dataset = train_dataset + test_dataset

    elif dataset_name == 'gtsrb':
        
        mean_nums = [0.485, 0.456, 0.406]
        std_nums = [0.229, 0.224, 0.225]
        if trained == True:
            transform = transforms.Compose([
                resize_transform,

                transforms.ToTensor(), # 转换为tensor
                transforms.Normalize(mean_nums, std_nums),
            ])
            train_dataset = torchvision.datasets.GTSRB(root=data_dir, split='train',transform=transform, download=True)
        else:
            transform =transforms.Compose([
                
                resize_transform,
                # transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean_nums, std_nums)
            ])
            train_dataset = torchvision.datasets.GTSRB(root=data_dir, split='test', transform=transform, download=True)


    elif dataset_name == 'tiny_imagenet':
        transform = transforms.Compose([
            resize_transform ,
            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        from CustomDataset import TinyImageNet
        data_dir = './data/tiny-imagenet-200/'
        train_dataset = TinyImageNet(data_dir, transform=transform, train=trained)
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))

    return train_dataset


def get_dataset_transform(dataset_name, trained=True):

    pass

def load_dataset_plus(config, trained=True):
    dataset_name = config['dataset']['name']
    data_dir = 'data'



    if dataset_name == 'mnist':

        train_dataset = torchvision.datasets.MNIST(root='data', train=trained, download=True)
        # test_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transform, download=True)

        # combined_dataset = train_dataset

    elif dataset_name == 'cifar10':

        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=trained, download=True)
        # test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)
        # combined_dataset = train_dataset

    elif dataset_name == 'cifar100':

        train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=trained, download=True)
        # test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, transform=transform, download=True)
        # combined_dataset = train_dataset + test_dataset

    elif dataset_name == 'imagenet1k':

        train_dataset = torchvision.datasets.ImageFolder(root=data_dir,)
        # test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
        # combined_dataset = train_dataset + test_dataset

    elif dataset_name == 'gtsrb':

        data_path = "data/gtsrb/GTSRB"
        if trained == True:
            train_dataset = torchvision.datasets.GTSRB(root=data_dir, split='train', download=True)
        else:
            train_dataset = torchvision.datasets.GTSRB(root=data_dir, split='test', download=True)
        # combined_dataset = train_dataset + test_dataset
        
        # train_dataset = CustomGTSRBDataset(data_path, transform)

    elif dataset_name == 'tiny_imagenet':

        from CustomDataset import TinyImageNet
        data_dir = './data/tiny-imagenet-200/'
        train_dataset = TinyImageNet(data_dir,  train=True)
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))

    return train_dataset



def split_data(dataset, train_ratio, config, random_seed=None):
    # dataset = dataloader.dataset

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size

    if random_seed is not None:
        torch.manual_seed(random_seed)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    return train_dataset, test_dataset



def dirichlet_split_noniid(train_labels, num_class, alpha, n_clients, seed, config):

    # The Dirichlet distribution of argument alpha 
    # divides the set of sample indexes into subsets of n_clients

    np.random.seed(seed)
    torch.manual_seed(seed)

    n_classes = num_class
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # (K, N) Category label distribution matrix X, which records the proportion of each category to each client

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    # (K, ...) Records the sample index set corresponding to K categories

    client_idcs = [[] for _ in range(n_clients)]
    # Record the sample index set corresponding to N clients
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split divides the sample of class k into N subsets according to proportion
        #  i indicates the i th client, and idcs indicates its corresponding sample index set idcs
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs


def get_dataset_labels(dataset ,dataset_name):
    
    labels = []
 

    # for sample in dataset:
    #     print(sample)
    
    labels = [sample[1] for sample in dataset]
    # if dataset_name == 'gtsrb':
    #     labels = [target for _, target in dataset._samples]
    # elif dataset_name == 'cifar10':
    #     labels = [sample[1] for sample in dataset]
    # elif dataset_name == 'tiny_imagenet':
    #     labels = [target for _, target in dataset.images]

    # else:
    #     ValueError("Unsupported dataset. Please implement the 'get_labels' method for the dataset.")
    return labels


def dirichlet_distribution(dataset, config, clinets, seed=0):
    total_samples = len(dataset)
    iid = config['federated_learning']['iid']
    n_clients = clinets
    dataset_name = config['dataset']['name']

    np.random.seed(seed)
    torch.manual_seed(seed)

    if iid:
        samples_per_client = total_samples // n_clients
        indices = list(range(total_samples))
        np.random.shuffle(indices)
        subsets = [indices[i*samples_per_client: (i+1)*samples_per_client] for i in range(0, n_clients)]

    else:

        if dataset_name == 'cifar10':
            num_cls = 10
        elif dataset_name == 'mnist':
            num_cls = 43
        elif dataset_name == 'emnist':
            num_cls = 47
        elif dataset_name == 'gtsrb':
            num_cls = 43
        elif dataset_name == 'tiny_imagenet':
            num_cls = 200
        
        DIRICHLET_ALPHA = config['federated_learning']['dirichlet_rate']

        # train_labels = dataset.labels
        # num_cls = dataset.num_class
        train_labels = np.array(get_dataset_labels(dataset, dataset_name))
        
        subsets = dirichlet_split_noniid(train_labels, num_class=num_cls,
                                         alpha=DIRICHLET_ALPHA, n_clients=n_clients,seed=seed, config=config)

    local_dataset = []
    for subset_indices in subsets:
    
        subset = Subset(dataset, subset_indices)
        local_dataset.append(subset)

    return local_dataset


def get_dataset(backdoor_index, config):



    dataset_name = config['dataset']['name']
    Dalpha = config['federated_learning']['dirichlet_rate']
    poisoned_rate =config['backdoor_paras']['poisoned_rate']
    num_client = config['federated_learning']['client_number']
    cover_rate = config['backdoor_paras']['cover_rate']
    # read dataset from the path
    file_path = os.path.join('data','distribution', dataset_name)
    tmp = 'dalpha=' + str(Dalpha)
    file_path = os.path.join(file_path, tmp)
    backdoor_name = config['backdoor_paras']['backdoor_name']

    os.makedirs(file_path, exist_ok=True)
    # Save the dataset list as a.pkl file
    dataset_train_path = os.path.join(file_path, 'dataset_train_list.pkl')
    dataset_test_path = os.path.join(file_path, 'dataset_test_list.pkl')



    
    # Load the dataset list
    with open(dataset_train_path, 'rb') as f:
        dataset_train_list = pickle.load(f)

    with open(dataset_test_path, 'rb') as f:
        dataset_test_list = pickle.load(f)


    # global_dataset_test_path =  os.path.join(file_path, 'global_dataset_test.pkl')
    # with open(global_dataset_test_path, 'rb') as f:
    #     global_dataset_test = pickle.load(f)



    if config['federated_learning']['iid'] == True:
        train_dataset_list = []
        test_dataset_list = []
        

        if backdoor_name == 'scaling_attack':

            poisoned_type = 'badnet'

        elif backdoor_name == 'neurotoxin':

            poisoned_type = 'badnet'
        else:
            poisoned_type = backdoor_name

        benign_train_dataset = dataset_train_list[0]
        benign_test_dataset = dataset_test_list[0]


        for client_id in range(num_client):

            if client_id in backdoor_index:

                poisoned_train_dataset = copy.deepcopy(benign_train_dataset)

                poisoned_train_dataset = create_poisoned_set(dataset=poisoned_train_dataset, 
                                                            poison_rate=poisoned_rate, cover_rate=cover_rate,
                                                            backdoor_name = poisoned_type,
                                                            config_list=config, client_idx=client_id)
                
                train_dataset_list.append(poisoned_train_dataset)

            else:
                train_dataset_list.append(copy.deepcopy(benign_train_dataset))
        
            test_dataset_list.append(benign_test_dataset)
    else:
    
        train_dataset_list = []
        test_dataset_list = []



        if backdoor_name == 'scaling_attack':
            poisoned_type = 'badnet'
        elif backdoor_name == 'neurotoxin':
            poisoned_type = 'badnet'
        else:
            poisoned_type = backdoor_name


        for client_id in range(num_client):

            if client_id in backdoor_index:

                poisoned_train_dataset = copy.deepcopy(dataset_train_list[client_id])

                poisoned_train_dataset = create_poisoned_set(dataset=poisoned_train_dataset, 
                                                            poison_rate=poisoned_rate, cover_rate=cover_rate,
                                                            backdoor_name = poisoned_type,
                                                            config_list=config, client_idx=client_id)
                
                train_dataset_list.append(poisoned_train_dataset)


            else:
                train_dataset_list.append(copy.deepcopy(dataset_train_list[client_id]))

            test_dataset_list.append(dataset_test_list[client_id])


    return train_dataset_list, test_dataset_list
   #  return train_dataset_list, test_dataset_list, global_dataset_test



class HookTool: 
    def __init__(self):
        self.hook = None
        self.fea = None 

    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out

    def register_hook(self, module):
        self.hook = module.register_forward_hook(self.hook_fun)

    def remove(self):
        if self.hook is not None:
            self.hook.remove()


# the second step: register hook, I will tell model in which layer to extract the feature #
def get_feas_by_hook(model, layer_names: list):
    # To extract the feature after the specified name layer, we need to traverse the module of the model, 
    # and then find the specified name layer, register the hook function to these layers;
    # This is equivalent to telling the model that I'm going to use hook_fun on the layer 
    # with the specified name to process the feature output from that layer.
    # Since there may be more than one named layer in a model, we use hook_feas to store 
    # the features after each named layer
    fea_hooks = []
    for name in layer_names:
        module = model
        for attr in name.split('.'):
            module = getattr(module, attr)
        hook = HookTool()
        hook.register_hook(module=module)
        # module.register_forward_hook(hook.hook_fun)

        fea_hooks.append(hook)
    return fea_hooks


import torch.nn as nn
import torch.nn.functional as F

'''
AT with sum of absolute values with power p
code from: https://github.com/AberHu/Knowledge-Distillation-Zoo
'''
class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p):
		super(AT, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am


def get_NAD_layers(model_name):

    if model_name == 'resnet18':
        layers =  ['layer1', 'layer2', 'layer3', 'layer4']
        weight_list = [1000,2000,2000,1]
        # weight_list = [10,20,20,50]
        # weight_list = [50,100,100,250]
    

    return layers, weight_list

def grad_replace(backdoor_weight, num_clients, scale_factor:None, device): 



    if scale_factor is None:
        gamma = num_clients
    else:
        gamma = grad_compute_gamma(backdoor_weight,
                                scale_factor_s=scale_factor)
        
    weight_diff = gamma * backdoor_weight

    return weight_diff


def grad_compute_gamma(backdoor_weight, scale_factor_s, device='cpu'):


    gamma = scale_factor_s / torch.norm(backdoor_weight, p=2)

    return gamma

def model_replace(global_model:torch.nn.Module, backdoor_model:torch.nn.Module, num_clients, scale_factor:None, device): 


    if scale_factor is None:
        gamma = num_clients
    else:
        gamma = model_compute_gamma(global_model, backdoor_model,scale_factor_s=scale_factor)
    aggregated_params = {}

    
    new_model:torch.nn.Module = copy.deepcopy(global_model)

    for name, param in new_model.state_dict().items():

        aggregated_params[name] = torch.zeros_like(param.data, dtype=torch.float)

        aggregated_params[name] = (gamma * (backdoor_model.state_dict()[name] - global_model.state_dict()[name]) + global_model.state_dict()[name]).clone().detach().float().to(device)



    new_model.load_state_dict(aggregated_params)
    
    
    return new_model






def model_compute_gamma(global_model, original_client_model, scale_factor_s, device='cpu'):


    malicious_client_model_vec = vectorize_model(original_client_model)
    global_model_vec = vectorize_model(global_model)
    
    gamma = scale_factor_s / (torch.norm((global_model_vec-malicious_client_model_vec), p=2))

    return gamma




def get_test_weight(config):


    dataset_name = config['dataset']['name']
    Dalpha = config['federated_learning']['dirichlet_rate']
    backdoor_rate = config['federated_learning']['backdoor_rate']
    client_number = config['federated_learning']['client_number']
    attacker_number = int(backdoor_rate * client_number)


    file_path = os.path.join('data','distribution', dataset_name)
    tmp = 'dalpha=' + str(Dalpha)
    file_path = os.path.join(file_path, tmp)

    os.makedirs(file_path, exist_ok=True)

    dataset_test_path = os.path.join(file_path, 'dataset_test_list.pkl')

    with open(dataset_test_path, 'rb') as f:
        dataset_test_list = pickle.load(f)

    test_sample_list = []

    for idx in range(len(dataset_test_list)):

        if idx < attacker_number:
            pass
        else:
            person_id = idx
            person_dataset = dataset_test_list[person_id]

            val = len(person_dataset)
            test_sample_list.append(val)

    # print('the length of test_dataset is: ', len(test_sample_list))
    # print('the test sample is:', test_sample_list)
    total_num = sum(test_sample_list)
    for idx in range(len(test_sample_list)):
        test_sample_list[idx] = test_sample_list[idx] / total_num
    # print('the total number of sample is:', total_num)
    print('the ratio of each client is:', test_sample_list)

    return test_sample_list



