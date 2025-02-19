from utils.utility import load_config
from  utils.plugin import split_data
from utils.plugin import dirichlet_distribution
import copy
import pickle
import os
from utils.plugin import load_dataset

def generate_noniid_distribution(config):
    
    Dalpha = config['federated_learning']['dirichlet_rate']
    client_number = config['federated_learning']['client_number']

    trainset = load_dataset(dataset_name=config['dataset']['name'], data_dir=config['dataset']['save_path'],trained=True)
    testset = load_dataset(dataset_name=config['dataset']['name'], data_dir=config['dataset']['save_path'], trained=False)
    seed= config['general']['random_seed']
    dataset_name = config['dataset']['name']

    all_train_dataset_list = dirichlet_distribution(trainset, config, clinets=client_number, seed=seed)
    file_path = os.path.join('data','distribution', dataset_name)
    tmp = 'dalpha=' + str(Dalpha)
    
    file_path = os.path.join(file_path, tmp)
    # print(file_path)

    os.makedirs(file_path, exist_ok=True)
    dataset_train_path = os.path.join(file_path, 'dataset_train_list.pkl')
    dataset_test_path = os.path.join(file_path, 'dataset_test.pkl')

    with open(dataset_train_path, 'wb') as f:
        pickle.dump(all_train_dataset_list, f)

    with open(dataset_test_path, 'wb') as f:
        pickle.dump(testset, f)



def generate_pfl_noniid_distribution(config):
    import torch.utils.data

    Dalpha = config['federated_learning']['dirichlet_rate']
    client_number = config['federated_learning']['client_number']
    split_rate = config['federated_learning']['split_rate']
    trainset = load_dataset(dataset_name=config['dataset']['name'], data_dir=config['dataset']['save_path'],trained=True)
    testset = load_dataset(dataset_name=config['dataset']['name'], data_dir=config['dataset']['save_path'], trained=False)
    seed= config['general']['random_seed']

    num_group = config['federated_learning']['number_group']
    dataset_name = config['dataset']['name']
    # print(len(trainset))
    # print(len(testset))
    combined_data = torch.utils.data.ConcatDataset([trainset, testset])
    # print(len(combined_data))
    # print(combined_data[0])
    expanded_data_list = [copy.deepcopy(combined_data) for _ in range(num_group)]
    expanded_data = torch.utils.data.ConcatDataset(expanded_data_list)


    all_dataset_list = dirichlet_distribution(expanded_data, config, clinets=client_number, seed=seed)

    # all_dataset_list = []
    # for idx in range(num_group):
    #     all_dataset_list.extend(dataset_list)
    dataset_train_list = []
    dataset_test_list = []
    for i in range(client_number):
        train_data, test_data = split_data(all_dataset_list[i], split_rate, config)
        # from collections import Counter
        # train_label_counts = Counter(item[1] for item in train_data) 
        # test_label_counts = Counter(item[1] for item in test_data) 

        dataset_train_list.append(train_data)
        dataset_test_list.append(test_data)

    file_path = os.path.join('data','distribution', dataset_name)
    tmp = 'dalpha=' + str(Dalpha)
    
    file_path = os.path.join(file_path, tmp)
    # print(file_path)

    os.makedirs(file_path, exist_ok=True)

    dataset_train_path = os.path.join(file_path, 'dataset_train_list.pkl')
    dataset_test_path = os.path.join(file_path, 'dataset_test_list.pkl')
    global_dataset_test_path =  os.path.join(file_path, 'global_dataset_test.pkl')

    global_test_data = []
    for test_data in  dataset_test_list:
        global_test_data.extend(test_data)

    with open(dataset_train_path, 'wb') as f:
        pickle.dump(dataset_train_list, f)

    with open(dataset_test_path, 'wb') as f:
        pickle.dump(dataset_test_list, f)

    with open(global_dataset_test_path, 'wb') as f:
        pickle.dump(global_test_data, f)



if __name__ == '__main__':

    config_list = load_config('./utils/config.yaml')
    generate_pfl_noniid_distribution(config=config_list)
    #generate_noniid_distribution(config=config_list)




