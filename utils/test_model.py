import torch
from tqdm import tqdm

from utils.utility import load_config, load_dataset
import os
import torch.nn as nn
import numpy as np




def test_model(model, data_loader, config_list, path=None,
               poison_test=False, poison_transform=None,
               num_classes=10, source_classes=None, all_to_all=False):
    # test_loader


    if path is not None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
    model.eval()

    dataset_name = config_list['dataset']['name']
    device_state = config_list['general']['device']
    cuda_num = config_list['general']['cuda_number']
    device = torch.device("cuda:{}".format(cuda_num) if torch.cuda.is_available() and device_state == 'gpu' else "cpu")

    model.to(device)
    clean_correct = 0
    poison_correct = 0
    non_source_classified_as_target = 0
    tot = 0
    num_non_target_class = 0
    criterion = nn.CrossEntropyLoss()
    tot_loss = 0
    poison_acc = 0

    class_dist = np.zeros((num_classes))
    # target_class = base_config.target_class[dataset_name]
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(data_loader):

            # clean data, clean target
            data, target = data.to(device), target.to(device)
            clean_output = model(data)
            clean_pred = clean_output.argmax(dim=1)
            clean_correct += clean_pred.eq(target).sum().item()

            tot += len(target)
            this_batch_size = len(target)
            tot_loss += criterion(clean_output, target) * this_batch_size

            # add to corresponding class
            for bid in range(this_batch_size):
                if clean_pred[bid] == target[bid]:
                    class_dist[target[bid]] += 1

            # need poison_test
            if poison_test:
                clean_target = target
                # poisoned data, poisoned target
                data, target = poison_transform.transform(data, target)

                poison_output = model(data)
                poison_pred = poison_output.argmax(dim=1, keepdim=True)

                if not all_to_all:

                    target_class = target[0].item()
                    for bid in range(this_batch_size):
                        if clean_target[bid] != target_class:
                            # previous class not equal poisoned target class we need to add
                            if source_classes is None:
                                num_non_target_class += 1
                                if poison_pred[bid] == target_class:
                                    poison_correct += 1
                                    
                            else:  # for source-specific attack
                                if clean_target[bid] in source_classes:
                                    num_non_target_class += 1
                                    if poison_pred[bid] == target_class:
                                        poison_correct += 1

                else:

                    for bid in range(this_batch_size):
                        num_non_target_class += 1
                        if poison_pred[bid] == target[bid]:
                            poison_correct += 1

                poison_acc += poison_pred.eq((clean_target.view_as(poison_pred))).sum().item()

    print('Clean ACC: {}/{} = {:.6f}, Loss: {}'.format(
        clean_correct, tot,
        100 * clean_correct / tot, tot_loss / tot
    ))
    if poison_test:
        print('ASR: %d/%d = %.6f' % (poison_correct, num_non_target_class, 100*poison_correct / num_non_target_class))
        # print('Attack ACC: %d/%d = %.6f' % (poison_acc, tot, poison_acc/tot) )
    # print('Class_Dist: ', class_dist)
    print("")

    if poison_test:
        return 100 * clean_correct / tot, 100 * poison_correct / num_non_target_class
    return 100 *clean_correct / tot, None




