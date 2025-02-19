from utils.utility import load_config
import matplotlib.pyplot as plt
import numpy as np
from utils.plugin import load_dataset
from utils.inject_backdoor import create_poisoned_set
import copy

if __name__ == "__main__":
    print()


    path = './utils/config.yaml'
    config = load_config(path)

    dataset = load_dataset(dataset_name=config['dataset']['name'], data_dir=config['dataset']['save_path'], trained=True)
    # img, label = dataset[790]
    # print(label)
    {'airplane': 0, 'automobile': 1, 'automobile': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
     'truck': 9}

    class_id_to_label = {0:'airplane', 1:'automobile', 2:'automobile', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
    # 911
    select_idx = 902
    attack_method = ['none', 'badnet_all_to_all', 'blend', 'adaptive_patch', 'trojan' ,'DBA']
    # attack_method = ['none','adaptive_patch']
    for poisoned_name in attack_method:
       if poisoned_name == 'none':
           path_name = 'airplane_' + poisoned_name  + '.pdf'
           attack_sample = []
           data = copy.deepcopy(dataset[select_idx])
           plt.imshow(np.transpose(data[0].numpy(), (1, 2, 0)))
           # plt.title(f'Label: {poisoned_data[0][1]}')
           print('label is :', class_id_to_label[int(data[1])])
           plt.axis('off')
           plt.xticks([])
           plt.yticks([])
           plt.gca().set_aspect('equal', adjustable='box')
           plt.tight_layout()  
           plt.savefig(path_name, bbox_inches='tight')
           plt.close()
       else:
  
           path_name = 'airplane_' + poisoned_name + '.pdf'
           attack_sample = []
           data = copy.deepcopy(dataset[  select_idx])
           attack_sample.append(data)
           config['backdoor_paras']['backdoor_name'] = poisoned_name
           poisoned_data = create_poisoned_set(attack_sample, backdoor_name=poisoned_name ,poison_rate=1, cover_rate=0, config_list=config, client_idx=2)

           plt.imshow(np.transpose(poisoned_data[0][0].numpy(), (1, 2, 0)))
           # plt.title(f'Label: {poisoned_data[0][1]}')

           config['backdoor_paras']['alpha'] = 0.4
           plt.axis('off')
           plt.xticks([])
           plt.yticks([])
           plt.gca().set_aspect('equal', adjustable='box')
           plt.tight_layout()  
           plt.savefig(path_name, bbox_inches='tight')
           plt.close()




