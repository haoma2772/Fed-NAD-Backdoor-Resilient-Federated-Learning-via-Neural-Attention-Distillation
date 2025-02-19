import torch
from tqdm import tqdm

from utils.utility import load_config, load_dataset

from utils.model import Net
from test_model import test_model

import wandb
from torch.utils.data import DataLoader

from test_model import test_model
from utils.utility import  get_poison_transform
import base_config
from torchvision import transforms





def backdoor_train_model(model, data_loader, device, config_list, client_id, global_round, 
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
            poison_type = config_list['poisoned_paras']['poisoned_type']
            dataset_name = config_list['dataset']['name']
            target_class = base_config.target_class[dataset_name]

            alpha = config_list['poisoned_paras']['alpha']
            trigger = config_list['poisoned_paras']['trigger']

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

    print('Finished training')

    return model,  record


def clean_train_model(self_flag, model, data_loader, device, config_list, client_id, global_round, 
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
    # wandb.watch(model, log_freq=batch_size)
    
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
            wandb.log({'Local loss of clean client#{}'.format(client_id): running_loss / len(data_loader), 'epoch':global_round*num_epochs+epoch})
        else:
             wandb.log({'Loss of clean client#{}'.format(client_id): running_loss / len(data_loader), 'epoch':global_round*num_epochs+epoch})
        if record is not None:
            record['loss'].append(running_loss / len(data_loader))
        # wandb.log({'Loss of backdoor client#{}'.format(client_id): running_loss / len(data_loader)}, step=global_round * num_epochs + epoch)
       #  writer.add_scalar('Loss', running_loss / len(data_loader), epoch+1)

        if with_test:
            # need test
            poison_type = config_list['poisoned_paras']['poisoned_type']
            dataset_name = config_list['dataset']['name']
            target_class = base_config.target_class[dataset_name]
            source_class = base_config.source_class
            cover_classes = base_config.cover_classes
            alpha = config_list['poisoned_paras']['alpha']
            trigger = config_list['poisoned_paras']['trigger']
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

            CA, ASR = test_model(model=model, data_loader=test_dataset_loader, config_list=config_list, path=None,
                                 poison_test=True,
                                 poison_transform=poison_transform, num_classes=num_classes,
                                 source_classes=None, all_to_all=all_to_all
                                 )
            if self_flag == True:
                wandb.log(
                    {'Local test_accuracy of clean client#{}'.format(client_id): CA, 'epoch': global_round * num_epochs + epoch})
                wandb.log({'Local attack_success_rate of clean client#{}'.format(client_id): ASR,
                        'epoch': global_round * num_epochs + epoch})
            else:
                wandb.log(
                    {'Test_accuracy of clean client#{}'.format(client_id): CA, 'epoch': global_round * num_epochs + epoch})
                wandb.log({'Attack_success_rate of clean client#{}'.format(client_id): ASR,
                        'epoch': global_round * num_epochs + epoch})

            if record is not None:
                record['test_acc'].append(CA)
                record['attack_acc'].append(ASR)

    print('Finished training')

    return model, record


if __name__ == '__main__':
    pass




