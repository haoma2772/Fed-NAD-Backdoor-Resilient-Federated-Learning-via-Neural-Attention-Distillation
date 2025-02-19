
from torchvision import transforms

import torch
import os
from PIL import Image
from utils import base_config
import numpy as np
from utils.CustomDataset import CustomPoisonedDataset


def create_poisoned_set(dataset, poison_rate, backdoor_name, cover_rate, config_list, client_idx):

    seed = config_list['general']['random_seed']

    resize_transform = transforms.Resize(224, antialias=True)


    poisoned_rate = poison_rate
    cover_rate = cover_rate
    alpha = config_list['backdoor_paras']['alpha']
    trigger = config_list['backdoor_paras']['trigger']
    dataset_name = config_list['dataset']['name']

    # according to different dataset and poisoned type select trigger
    if trigger == 'None':
        trigger = base_config.trigger_default[dataset_name][backdoor_name]

    # the same as the target class

    # dynamic method
    if backdoor_name == 'dynamic':
        if dataset_name == 'cifar10':
            img_size = 224
            num_classes = 10
            channel_init = 224
            steps = 3
            input_channel = 3
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                resize_transform,
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])

            normalizer = transforms.Compose([

                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])

            denormalizer = transforms.Compose([
                transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261],
                                     [1 / 0.247, 1 / 0.243, 1 / 0.261])
            ])
            ckpt_path = './models/all2one_cifar10_ckpt.pth.tar'


        elif dataset_name == 'gtsrb':
            img_size = 224
            num_classes = 43
            channel_init = 224
            steps = 3
            input_channel = 3

            data_transform = transforms.Compose([

                transforms.ToTensor(),
                resize_transform,
            ])
            normalizer = transforms.Compose([

                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
            denormalizer = transforms.Compose([

                transforms.Normalize([-0.485 / 0.229, -0.456/0.224, -0.406/0.225],[1/0.229, 1/0.224, 1/0.225])
            ])
            ckpt_path = './models/all2one_gtsrb_ckpt.pth.tar'

        elif dataset_name == 'tiny_imagenet':
            raise NotImplementedError('tiny_imagenet unsupported for dynamic!')
        else:
            raise NotImplementedError('Undefined Dataset')

    elif backdoor_name == 'ISSBA':
        if dataset_name =='cifar10':
            img_size = 224
            num_classes = 10
            input_channel = 3

            data_transform = transforms.Compose([

                transforms.ToTensor(),
                resize_transform,
            ])
            ckpt_path = './models/ISSBA_cifar10.pth'
        elif dataset_name == 'gtsrb':
            img_size = 224
            num_classes = 43
            input_channel = 3
            ckpt_path = './models/ISSBA_gtsrb.pth'


            data_transform = transforms.Compose([

                transforms.ToTensor(),
                resize_transform,
            ])
        elif dataset_name == 'tiny_imagenet':
            raise NotImplementedError('tiny_imagenet unsupported!')
        else:
            raise NotImplementedError('Undefined Dataset')

    else:
        if dataset_name == "cifar10":
            img_size = 224
            num_classes = 10
            data_transform = transforms.Compose([

                transforms.ToTensor(),
                resize_transform,
            ])
        elif dataset_name == 'mnist':
            img_size = 224
            num_classes = 10
            data_transform = transforms.Compose([

                transforms.ToTensor(),
                resize_transform,
            ])
        elif dataset_name == 'emnist':
            img_size = 224
            num_classes = 47
            data_transform = transforms.Compose([

                transforms.ToTensor(),
                resize_transform,
            ])
        elif dataset_name == 'gtsrb':
            img_size = 224
            num_classes = 43
            data_transform = transforms.Compose([

                transforms.ToTensor(),
                resize_transform,
            ])
            
        elif dataset_name == 'tiny_imagenet':
            # 这个也可以得到的直接
            img_size = 224
            num_classes = 200
            data_transform = transforms.Compose([

                transforms.ToTensor(),
                resize_transform,
            ])
        else:
            raise NotImplementedError('Undefined Dataset')


    # obtain trigger
    trigger_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, antialias=True),

    ])

    poison_set_dir = 'poisoned_set'

    if backdoor_name in ['basic', 'badnet', 'blend', 'clean_label', 'refool',
                        'adaptive_blend', 'adaptive_patch', 'adaptive_k_way',
                        'SIG', 'TaCT', 'WaNet', 'SleeperAgent', 'none',
                        'badnet_all_to_all', 'trojan', 'DBA']:
        trigger_name = trigger
        trigger_path = os.path.join(base_config.triggers_dir, trigger_name)

        trigger = None
        trigger_mask = None

        if trigger_name != "none":
            # none for SIG
            trigger_path = os.path.join(base_config.triggers_dir, trigger_name)
            trigger = Image.open(trigger_path).convert("RGB")
            trigger = trigger_transform(trigger)

            trigger_mask_path = os.path.join(base_config.triggers_dir, 'mask_%s' % trigger_name)
            if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)

                #print('trigger_mask_path:', trigger_mask_path)
                trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                # trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
                trigger_mask = trigger_transform(trigger_mask)
            else:  # by default, all black pixels are masked with 0's
                # print('No trigger mask found! By default masking all black pixels...')
                trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0),
                                                trigger[2] > 0).float()
                # trigger_mask = trigger_transform(trigger_mask)

        if backdoor_name == 'basic':

            from poison_tool_box import basic
            poison_generator = basic.poison_generator(img_size=img_size, dataset=dataset,
                                                      poison_rate=poisoned_rate,
                                                      path=poison_set_dir,
                                                      trigger_mark=trigger, trigger_mask=trigger_mask,
                                                      target_class=base_config.target_class[dataset_name], alpha=alpha)

        elif backdoor_name == 'badnet':

            from poison_tool_box import badnet
            poison_generator = badnet.poison_generator(img_size=img_size, dataset=dataset,
                                                       poison_rate=poisoned_rate, trigger_mark=trigger,
                                                       trigger_mask=trigger_mask,
                                                       path=poison_set_dir,
                                                       target_class=base_config.target_class[dataset_name],
                                                       num_classes=num_classes)

        elif backdoor_name == 'badnet_all_to_all':

            from poison_tool_box import badnet_all_to_all
            poison_generator = badnet_all_to_all.poison_generator(img_size=img_size, dataset=dataset,
                                                                  poison_rate=poisoned_rate, trigger_mark=trigger,
                                                                  trigger_mask=trigger_mask,
                                                                  path=poison_set_dir, num_classes=num_classes)

        elif backdoor_name == 'trojan':

            from poison_tool_box import trojan
            poison_generator = trojan.poison_generator(img_size=img_size, dataset=dataset,
                                                       poison_rate=poisoned_rate, trigger_mark=trigger,
                                                       trigger_mask=trigger_mask,
                                                       path=poison_set_dir,
                                                       target_class=base_config.target_class[dataset_name],
                                                       num_classes=num_classes)

        elif backdoor_name == 'blend':

            from poison_tool_box import blend
            poison_generator = blend.poison_generator(img_size=img_size, dataset=dataset,
                                                      poison_rate=poisoned_rate, trigger=trigger,
                                                      path=poison_set_dir,
                                                      target_class=base_config.target_class[dataset_name],
                                                      alpha=alpha,
                                                      num_classes=num_classes)
        
        elif backdoor_name == 'DBA':

            pattern_name = str(client_idx% base_config.DBA_poison_patterns_num) + '_poison_pattern'
            poison_patterns = base_config.DBA_poison_patterns[dataset_name][pattern_name]

            from poison_tool_box import DBA
            poison_generator = DBA.poison_generator(img_size=img_size, dataset=dataset,
                                                      poison_rate=poisoned_rate,
                                                      poison_patterns=poison_patterns,
                                                      path=poison_set_dir,
                                                      target_class=base_config.target_class[dataset_name],
                                                      alpha=alpha,
                                                      seed=seed)


        elif backdoor_name == 'refool':
   
            from poison_tool_box import refool
            poison_generator = refool.poison_generator(img_size=img_size, dataset=dataset,
                                                       poison_rate=poisoned_rate,
                                                       path=poison_set_dir,
                                                       target_class=base_config.target_class[dataset_name],
                                                       max_image_size=224)

        elif backdoor_name == 'TaCT':
            # this method has source and target class

            # source_
            from poison_tool_box import TaCT
            poison_generator = TaCT.poison_generator(img_size=img_size, dataset=dataset,
                                                     poison_rate=poisoned_rate, cover_rate=cover_rate,
                                                     trigger=trigger, mask=trigger_mask,
                                                     path=poison_set_dir,
                                                     target_class=base_config.target_class[dataset_name],
                                                     source_class=base_config.source_class,
                                                     cover_classes=base_config.cover_classes)
        

        elif backdoor_name == 'WaNet':
            # Prepare grid
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


            # path = os.path.join(poison_set_dir, 'identity_grid')
            # torch.save(identity_grid, path)
            # path = os.path.join(poison_set_dir, 'noise_grid')
            # torch.save(noise_grid, path)


            from poison_tool_box import WaNet
            poison_generator = WaNet.poison_generator(img_size=img_size, dataset=dataset,
                                                      poison_rate=poisoned_rate, cover_rate=cover_rate,
                                                      path=poison_set_dir,
                                                      identity_grid=identity_grid, noise_grid=noise_grid,
                                                      s=s, k=k, grid_rescale=grid_rescale,
                                                      target_class=base_config.target_class[dataset_name])

        # elif backdoor_name == 'adaptive':
        #     # no this method
        #     from poison_tool_box import adaptive
        #     poison_generator = adaptive.poison_generator(img_size=img_size, dataset=dataset,
        #                                                  poison_rate=poisoned_rate,
        #                                                  path=poison_set_dir,
        #                                                  trigger_mark=trigger, trigger_mask=trigger_mask,
        #                                                  target_class=base_config.target_class[dataset_name], alpha=alpha,
        #                                                  cover_rate=cover_rate)

        elif backdoor_name == 'adaptive_blend':

            from poison_tool_box import adaptive_blend
            poison_generator = adaptive_blend.poison_generator(img_size=img_size, dataset=dataset,
                                                               poison_rate=poisoned_rate,
                                                               path=poison_set_dir, trigger=trigger,
                                                               pieces=16, mask_rate=0.5,
                                                               target_class=base_config.target_class[dataset_name],
                                                               alpha=alpha,
                                                               cover_rate=cover_rate)

        elif backdoor_name == 'adaptive_patch':

            from poison_tool_box import adaptive_patch
            poison_generator = adaptive_patch.poison_generator(img_size=img_size, dataset=dataset,
                                                               poison_rate=poisoned_rate,
                                                               path=poison_set_dir,
                                                               trigger_names=base_config.adaptive_patch_train_trigger_names[
                                                                   dataset_name],
                                                               alphas=base_config.adaptive_patch_train_trigger_alphas[
                                                                   dataset_name],
                                                               target_class=base_config.target_class[dataset_name],
                                                               cover_rate=cover_rate,
                                                               random_seed=config_list['general']['random_seed'],
                                                               num_classes=num_classes)

        elif backdoor_name == 'adaptive_k_way':

            from poison_tool_box import adaptive_k_way
            poison_generator = adaptive_k_way.poison_generator(img_size=img_size, dataset=dataset,
                                                               poison_rate=poisoned_rate,
                                                               path=poison_set_dir,
                                                               target_class=base_config.target_class[dataset_name],
                                                               cover_rate=cover_rate)

        elif backdoor_name == 'SIG':

            from poison_tool_box import SIG
            poison_generator = SIG.poison_generator(img_size=img_size, dataset=dataset,
                                                    poison_rate=poisoned_rate,
                                                    path=poison_set_dir, target_class=base_config.target_class[dataset_name],
                                                    delta=30 / 255, f=6, num_classes=num_classes)

        elif backdoor_name == 'clean_label':

            if dataset_name == 'cifar10':
                adv_imgs_path = "/home/ubuntu/backdoor_toolbox/data/cifar10/" \
                                "clean_label/fully_poisoned_training_datasets/two_600.npy"

                if not os.path.exists("/home/ubuntu/backdoor_toolbox/data/cifar10/clean_label/"
                                      "fully_poisoned_training_datasets/two_600.npy"):
                    raise NotImplementedError(
                        "Run './data/cifar10/clean_label/setup.sh' first to launch clean label attack!")
                adv_imgs_src = np.load("/home/ubuntu/backdoor_toolbox/data/cifar10/clean_label/"
                                       "fully_poisoned_training_datasets/two_600.npy").astype(
                    np.uint8)
                adv_imgs = []
                for i in range(adv_imgs_src.shape[0]):
                    adv_imgs.append(data_transform(adv_imgs_src[i]).unsqueeze(0))
                adv_imgs = torch.cat(adv_imgs, dim=0)
                # assert adv_imgs.shape[0] == len(dataset)
            else:
                raise NotImplementedError('Clean Label Attack is not implemented for %s' % dataset_name)

            # Init Attacker
            from poison_tool_box import clean_label
            poison_generator = clean_label.poison_generator(img_size=img_size, dataset=dataset, adv_imgs=adv_imgs,
                                                            poison_rate=poisoned_rate,
                                                            trigger_mark=trigger, trigger_mask=trigger_mask,
                                                            path=poison_set_dir,
                                                            target_class=base_config.target_class[dataset_name])



        else:  # 'none'
            from poison_tool_box import none
            poison_generator = none.poison_generator(img_size=img_size, dataset=dataset,
                                                         path=poison_set_dir)


        # the above is to get a generator, the following is to get the poisoned data

        if backdoor_name not in ['TaCT', 'WaNet', 'adaptive_blend', 'adaptive_patch', 'adaptive_k_way']:
            img_set, poison_indices, label_set = poison_generator.generate_poisoned_training_set()
            # print('[Generate Poisoned Set] Save %d Images' % len(label_set))
            custom_poisoned_dataset = CustomPoisonedDataset(img_set, poison_indices, label_set)


        # elif backdoor_name == 'DBA':
            
            
        #     img_set, poison_indices, label_set = poison_generator.generate_poisoned_training_set()
        #     # print('[Generate Poisoned Set] Save %d Images' % len(label_set))
        #     custom_poisoned_dataset = CustomPoisonedDataset(img_set, poison_indices, label_set)

        else:
            img_set, poison_indices, cover_indices, label_set = poison_generator.generate_poisoned_training_set()
            # print('[Generate Poisoned Set] Save %d Images' % len(label_set))
            custom_poisoned_dataset = CustomPoisonedDataset(img_set, poison_indices, label_set, cover_indices)
            # cover_indices_path = os.path.join(poison_set_dir, 'cover_indices')
            # torch.save(cover_indices, cover_indices_path)
            # print('[Generate Poisoned Set] Save %s' % cover_indices_path)
        

        # img_path = os.path.join(poison_set_dir, 'imgs')
        # torch.save(img_set, img_path)
        # print('[Generate Poisoned Set] Save %s' % img_path)
        #
        # label_path = os.path.join(poison_set_dir, 'labels')
        # torch.save(label_set, label_path)
        # print('[Generate Poisoned Set] Save %s' % label_path)
        #
        # poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
        # torch.save(poison_indices, poison_indices_path)
        # print('[Generate Poisoned Set] Save %s' % poison_indices_path)
        #
        # print('poison_indices : ', poison_indices)
    elif backdoor_name == 'dynamic':
        """
                Since we will use the pretrained model by the original paper, here we use normalized data following 
                the original implementation.
                Download Pretrained Generator from https://github.com/VinAIResearch/input-aware-backdoor-attack-release
            """


        if not os.path.exists(ckpt_path):
            raise NotImplementedError(
                '[Dynamic Attack] Download pretrained generator first : https://github.com/VinAIResearch/input-aware-backdoor-attack-release')
        # Init Attacker
        from poison_tool_box import dynamic
        poison_generator = dynamic.poison_generator(ckpt_path=ckpt_path, channel_init=channel_init, steps=steps,
                                                    input_channel=input_channel, normalizer=normalizer,
                                                    denormalizer=denormalizer, dataset=dataset,
                                                    poison_rate=poisoned_rate, path=poison_set_dir,
                                                    target_class=base_config.target_class[dataset_name])

        # Generate Poison Data
        img_set, poison_indices, label_set = poison_generator.generate_poisoned_training_set()
        # print('[Generate Poisoned Set] Save %d Images' % len(label_set))
        custom_poisoned_dataset = CustomPoisonedDataset(img_set, poison_indices, label_set)


    else:
        raise NotImplementedError('%s not defined' % backdoor_name)


    return custom_poisoned_dataset



