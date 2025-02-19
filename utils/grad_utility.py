import torch
from tqdm import tqdm

from utils.test_model import test_model
import wandb
from utils.utility import vectorize_model
from utils.test_model import test_model
from utils.utility import  get_poison_transform
from utils import base_config
from torchvision import transforms
from utils.utility import vectorize_model_dict, calculate_norm, select_top_k_norms
import math



def grad_clean_train_model(self_flag, model, data_loader, device, config_list, client_id, global_round, 
                      record=None, with_test=False, test_dataset_loader=None, with_save=False):
    

    lr = config_list['device']['learning_rate']
    dataset_name = config_list['dataset']['name']
    num_epochs = config_list['device']['epoch']
    loss = config_list['device']['criterion']
 

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr
        )
    model.to(device)

    pre_model_weight = vectorize_model(model)

    for epoch in range(num_epochs):
        
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(data_loader, total=len(data_loader), desc=f"Training epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device) 
            labels = labels.to(device)  
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(data_loader)}')
        

        if self_flag == True:
            wandb.log({'Local teacher loss of clean client#{}'.format(client_id): running_loss / len(data_loader), 'epoch':global_round*num_epochs+epoch})
        else:
             wandb.log({'Loss of clean client#{}'.format(client_id): running_loss / len(data_loader), 'epoch':global_round*num_epochs+epoch})
        if record is not None:
            record['loss'].append(running_loss / len(data_loader))
        # wandb.log({'Loss of backdoor client#{}'.format(client_id): running_loss / len(data_loader)}, step=global_round * num_epochs + epoch)
       #  writer.add_scalar('Loss', running_loss / len(data_loader), epoch+1)

        if with_test:
            if config_list['backdoor_paras']['backdoor_name'] == 'scaling_attack':
                poison_type = 'badnet'
            elif config_list['backdoor_paras']['backdoor_name'] == 'neurotoxin':
                poison_type = 'badnet'
            else:
                poison_type = config_list['backdoor_paras']['backdoor_name']
            # need test

            dataset_name = config_list['dataset']['name']
            target_class = base_config.target_class[dataset_name]
            source_class = base_config.source_class
            cover_classes = base_config.cover_classes
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
            elif dataset_name == 'emnist':
                num_classes = 47

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

            CA, ASR = test_model(model=model, data_loader=test_dataset_loader, config_list=config_list, path=None,
                                 poison_test=True,
                                 poison_transform=poison_transform, num_classes=num_classes,
                                 source_classes=None, all_to_all=all_to_all
                                 )
            if self_flag == True:
                wandb.log(
                    {'Local teacher test_accuracy of clean client#{}'.format(client_id): CA, 'epoch': global_round * num_epochs + epoch})
                wandb.log({'Local teacher attack_success_rate of clean client#{}'.format(client_id): ASR,
                        'epoch': global_round * num_epochs + epoch})
            else:
                wandb.log(
                    {'Test_accuracy of clean client#{}'.format(client_id): CA, 'epoch': global_round * num_epochs + epoch})
                wandb.log({'Attack_success_rate of clean client#{}'.format(client_id): ASR,
                        'epoch': global_round * num_epochs + epoch})

            if record is not None:
                record['test_acc'].append(CA)
                record['attack_acc'].append(ASR)

    # return gt = w(t) - w(t+1)
    cur_model_weight = vectorize_model(model)
    weight_diff = (cur_model_weight - pre_model_weight).cpu()
    # weight_diff = (cur_model_weight - pre_model_weight)
    print('Finish training')

    return model, record, weight_diff



def grad_backdoor_train_model(model, data_loader, device, config_list, client_id, global_round, 
                 record=None, with_test=False, test_dataset_loader=None,
               with_save=False):
    
    dataset_name = config_list['dataset']['name']
    lr = config_list['device']['learning_rate']
    num_epochs = config_list['device']['epoch']
    loss = config_list['device']['criterion']
    model.to(device)


    loss_mapping = {
        'CrossEntropy': torch.nn.CrossEntropyLoss(),
    }
    criterion = loss_mapping[loss]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    pre_model_weight = vectorize_model(model)

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        for images, labels in tqdm(data_loader, total=len(data_loader), desc=f"Training epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device) 
            labels = labels.to(device)  
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(data_loader)}')

        wandb.log({'Loss of backdoor client#{}'.format(client_id): running_loss / len(data_loader), 'epoch':global_round*num_epochs+epoch})

        if record is not None:
            record['loss'].append(running_loss / len(data_loader))
        # wandb.log({'Loss of backdoor client#{}'.format(client_id): running_loss / len(data_loader)}, step=global_round * num_epochs + epoch)
       #  writer.add_scalar('Loss', running_loss / len(data_loader), epoch+1)

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

            num_classes = 0
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
            elif dataset_name == 'emnist':
                num_classes = 47

            if poison_type == 'badnet_all_to_all':
                all_to_all = True
            else:
                all_to_all = False
                if poison_type == 'TaCT':
                    source_class = base_config.source_class
                    cover_classes = base_config.cover_classes

            poison_transform = get_poison_transform(poison_type=poison_type, dataset_name=dataset_name,
                                                    target_class=target_class
                                                    , source_class=None, cover_classes=None,
                                                    is_normalized_input=True, trigger_transform=trigger_transform,
                                                    alpha=alpha, trigger_name=trigger,device=device)

            CA, ASR = test_model(model=model, data_loader=test_dataset_loader, config_list=config_list, path=None,
                                 poison_test=True,
                                 poison_transform=poison_transform, num_classes=num_classes,
                                 source_classes=None, all_to_all=all_to_all
                                 )

            wandb.log(
                {'Test_accuracy of backdoor client#{}'.format(client_id): CA, 'epoch': global_round * num_epochs + epoch})

            wandb.log({'Attack_success_rate of backdoor client#{}'.format(client_id): ASR,
                       'epoch': global_round * num_epochs + epoch})
            if record is not None:
                record['test_acc'].append(CA)
                record['attack_acc'].append(ASR)

    print('Finsh training')
    
    cur_model_weight = vectorize_model(model)
    weight_diff = (cur_model_weight - pre_model_weight).cpu()
    # weight_diff = (cur_model_weight - pre_model_weight)
    return model, record, weight_diff



def compute_bakcdoor_weight(all_weight_diff_list):

    if len(all_weight_diff_list) == 0:
        return None

    total_sum = sum(all_weight_diff_list)


    average_tensor = total_sum / len(all_weight_diff_list)

    return average_tensor


def Neurotoxin_backdoor_train(model:torch.nn.Module, data_loader, device, config_list, client_id, global_round, 
                record=None, with_test=False, test_dataset_loader=None,
               with_save=False):

    dataset_name = config_list['dataset']['name']
    lr = config_list['device']['learning_rate']
    num_epochs = config_list['device']['epoch']
    loss = config_list['device']['criterion']
    top_k_ratio = config_list['attack']['top_k_ratio']

    model.to(device)

    model_weight_dict = vectorize_model_dict(model)
    top_k = math.ceil(top_k_ratio * len(model_weight_dict))
    norm_dict = calculate_norm(model_weight_dict)

    # select_layers is a list
    select_layers:list = select_top_k_norms(norm_dict, top_k)
    

    for name, params in model.named_parameters():
        if name in select_layers:
            params.requires_grad = True
        else:
            params.requires_grad = False


    loss_mapping = {
        'CrossEntropy': torch.nn.CrossEntropyLoss(),
    }
    criterion = loss_mapping[loss]

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr,
    )
    
    pre_model_weight = vectorize_model(model)

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        for images, labels in tqdm(data_loader, total=len(data_loader), desc=f"Training epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device) 
            labels = labels.to(device)  
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(data_loader)}')

        wandb.log({'Loss of backdoor client#{}'.format(client_id): running_loss / len(data_loader), 'epoch':global_round*num_epochs+epoch})

        if record is not None:
            record['loss'].append(running_loss / len(data_loader))
        # wandb.log({'Loss of backdoor client#{}'.format(client_id): running_loss / len(data_loader)}, step=global_round * num_epochs + epoch)
       #  writer.add_scalar('Loss', running_loss / len(data_loader), epoch+1)

        if with_test:
            # need test
            poison_type = 'badnet'
            dataset_name = config_list['dataset']['name']
            target_class = base_config.target_class[dataset_name]

            alpha = config_list['backdoor_paras']['alpha']
            trigger = config_list['backdoor_paras']['trigger']

            num_classes = 0
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
            elif dataset_name == 'emnist':
                num_classes = 47

            if poison_type == 'badnet_all_to_all':
                all_to_all = True
            else:
                all_to_all = False
                if poison_type == 'TaCT':
                    source_class = base_config.source_class
                    cover_classes = base_config.cover_classes

            poison_transform = get_poison_transform(poison_type=poison_type, dataset_name=dataset_name,
                                                    target_class=target_class
                                                    , source_class=None, cover_classes=None,
                                                    is_normalized_input=True, trigger_transform=trigger_transform,
                                                    alpha=alpha, trigger_name=trigger,device=device)

            CA, ASR = test_model(model=model, data_loader=test_dataset_loader, config_list=config_list, path=None,
                                 poison_test=True,
                                 poison_transform=poison_transform, num_classes=num_classes,
                                 source_classes=None, all_to_all=all_to_all
                                 )

            wandb.log(
                {'Test_accuracy of backdoor client#{}'.format(client_id): CA, 'epoch': global_round * num_epochs + epoch})

            wandb.log({'Attack_success_rate of backdoor client#{}'.format(client_id): ASR,
                       'epoch': global_round * num_epochs + epoch})
            if record is not None:
                record['test_acc'].append(CA)
                record['attack_acc'].append(ASR)
    

    print('Finish training')
    
    cur_model_weight = vectorize_model(model)

    weight_diff = (cur_model_weight - pre_model_weight).cpu()
    # weight_diff = (cur_model_weight - pre_model_weight)
    return model, record, weight_diff

