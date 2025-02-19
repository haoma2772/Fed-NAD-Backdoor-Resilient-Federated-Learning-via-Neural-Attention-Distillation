import copy,math,os,pickle,torch,wandb
from datetime import datetime
from torch.utils.data import DataLoader
from wandb import AlertLevel
from utils.utility import load_config, vectorize_model, load_model_weight_diff, complete_graph
from utils.plugin import get_dataset
from utils.grad_utility import Neurotoxin_backdoor_train, grad_clean_train_model, grad_backdoor_train_model, compute_bakcdoor_weight
from utils.model_defense import FedNAD, NormClip
from utils.model import get_model
from utils.grad_defense import average, TrimmedMean

if __name__ == '__main__':


    project_name = 'Your project name'
    
    # Read configuration information
    path = './utils/config.yaml'
    config = load_config(path)

    # Extract various parameters for training
    global_round = config['federated_learning']['round']
    client_number = config['federated_learning']['client_number']
    dirichlet_rate = config['federated_learning']['dirichlet_rate']
    backdoor_rate = config['federated_learning']['backdoor_rate']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    lr = config['device']['learning_rate']
    model_name = config['model']['name']
    dataset_name = config['dataset']['name']
    defense_method = config['general']['denfense']
    backdoor_name = config['backdoor_paras']['backdoor_name']
    local_epoch = config['device']['epoch']
    cuda_num = config['general']['cuda_number']
    scale_factor = config['attack']['scale_factor']

    dl_rate = "noniid={}".format(dirichlet_rate)
    device = torch.device("cuda:{}".format(cuda_num) if torch.cuda.is_available() else "cpu")
    
    backdoor_client_number: int = max(0, math.ceil(client_number * backdoor_rate))
    backdoor_client_index = list(range(backdoor_client_number))
    benign_client_number = client_number - backdoor_client_number

    # Load datasets for training and testing
    train_dataset_list, clean_test_dataset_list = get_dataset(
        backdoor_index=backdoor_client_index, config=config)

    dataset_train_loader_list = [
        DataLoader(train_dataset_list[client_id], batch_size=batch_size, num_workers=num_workers, shuffle=True) for
        client_id in range(client_number)]
    
    dataset_test_loader_list = [
        DataLoader(clean_test_dataset_list[client_id], batch_size=batch_size, num_workers=num_workers, shuffle=True) for
        client_id in range(client_number)]

    now_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    runs_name = (model_name + '_' + dataset_name + '_' + backdoor_name + '_' + defense_method + '({date})'.format(
                date=now_time) + 'dalpha=' +
                str(dirichlet_rate))  + 'backdoor_rate:' + str(backdoor_rate)

    # Initialize wandb logging
    wandb.init(project=project_name, name=runs_name, config=config)

    wandb.alert(
        title="Code execution started!",
        text="Code execution has started!",
        level=AlertLevel.WARN,
        wait_duration=1, )
    
    # Initialize model for each client
    base_model = get_model(model_name=model_name, dataset_name=dataset_name)
    model_list = [copy.deepcopy(base_model) for _ in range(client_number)]

    if defense_method == 'Fed-NAD':
        self_model_list = copy.deepcopy(model_list)

    base_weight_diff  = vectorize_model(base_model)
    base_weight_diff = torch.zeros_like(base_weight_diff)
    all_weight_diff_list = []

    save_name = dataset_name + '_' + model_name + '_' + backdoor_name + '_' + defense_method + '_' + 'dalpha=' + str(dirichlet_rate) + '_' + 'backdoor_rate:' + str(backdoor_rate)
    save_path = os.path.join('supply_result', save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    if defense_method == 'NormClip':
        norm_bound = config['defense']['norm_bound']
        stddev = config['defense']['stddev']


    # Save models at specific rounds
    save_round = [0, 10, 20, 30] 

    for idx in range(client_number):
        all_weight_diff_list.append(base_weight_diff)

    adj = complete_graph(client_number)

    init_record_list = {'test_acc': [], 'attack_acc': [], 'loss': []}
    whole_init_record_list: list = [copy.deepcopy(init_record_list) for _ in range(client_number)]

    # Main loop for global rounds
    for each_round in range(global_round):
        model_list_old = copy.deepcopy(model_list)

        for client_id in range(client_number):
            # Local update
            if client_id in backdoor_client_index:
                print('Backdoor Client #{} training!'.format(client_id))
                if backdoor_name == 'Neurotoxin':
                    [model_list[client_id],  whole_init_record_list[client_id], all_weight_diff_list[client_id]] = Neurotoxin_backdoor_train(
                        model=model_list[client_id],
                        data_loader=dataset_train_loader_list[client_id],
                        device=device,
                        config_list=config,
                        client_id=client_id,
                        global_round=each_round,
                        record=whole_init_record_list[client_id],
                        with_test=True,
                        test_dataset_loader=dataset_test_loader_list[client_id],
                    )
                else:
                    [model_list[client_id],  whole_init_record_list[client_id], all_weight_diff_list[client_id]] = grad_backdoor_train_model(
                        model=model_list[client_id],
                        data_loader=dataset_train_loader_list[client_id],
                        device=device,
                        config_list=config,
                        client_id=client_id,
                        global_round=each_round,
                        record=whole_init_record_list[client_id],
                        with_test=True,
                        test_dataset_loader=dataset_test_loader_list[client_id],
                    )
            else:
                print('Clean Client #{} Distillation!'.format(client_id))

                if defense_method == 'Fed-NAD':
                    [model_list[client_id], 
                     self_model_list[client_id],
                     all_weight_diff_list[client_id],
                     whole_init_record_list[client_id]] = FedNAD(
                        model_list=model_list,
                        self_model=self_model_list[client_id],
                        device=device,
                        config_list=config, 
                        client_id=client_id,
                        data_loader=dataset_train_loader_list[client_id],
                        global_round=each_round,
                        record=whole_init_record_list[client_id],
                        with_test=True, 
                        test_dataset_loader=dataset_test_loader_list[client_id],
                        self_test_loader=dataset_test_loader_list[client_id],
                        with_save=False
                    )

                elif defense_method == 'NormCliping':
                    [model_list[client_id], 
                     whole_init_record_list[client_id],
                     all_weight_diff_list[client_id]] = NormClip(
                        model=model_list[client_id],
                        norm_bound=norm_bound,
                        stddev=stddev,
                        device=device,
                        config_list=config, 
                        client_id=client_id,
                        data_loader=dataset_train_loader_list[client_id],
                        global_round=each_round,
                        record=whole_init_record_list[client_id],
                        with_test=True, 
                        test_dataset_loader=dataset_test_loader_list[client_id],
                        with_save=False
                    )
                else:
                    # other methods
                    [model_list[client_id],
                     whole_init_record_list[client_id], 
                     all_weight_diff_list[client_id]] = grad_clean_train_model(
                        self_flag=False,
                        model=model_list[client_id], 
                        data_loader=dataset_train_loader_list[client_id],
                        device=device,
                        config_list=config, 
                        client_id=client_id,
                        global_round=each_round,
                        record=whole_init_record_list[client_id],
                        with_test=True, 
                        test_dataset_loader=dataset_test_loader_list[client_id],
                        with_save=False)

            # Handle backdoor attack after training
            # Poisoning attacks
            backdoor_avg_weight = compute_bakcdoor_weight(all_weight_diff_list)
            for backdoor_idx in range(backdoor_client_number):
                    load_model_weight_diff(
                        model=model_list[backdoor_idx],
                        weight_diff=backdoor_avg_weight,
                        global_weight=model_list_old[backdoor_idx])

            # Aggregation with defense methods
            if defense_method == 'Fed-Avg':
                cur_weight = average(all_weight_diff_list)
            elif defense_method == 'TrimmedMean':
                cur_weight = TrimmedMean(all_weight_diff_list, client_num=client_number, attacker_num=backdoor_client_number)
            elif defense_method == 'NormCliping':
                cur_weight = average(all_weight_diff_list)
            elif defense_method == 'Fed-NAD':
                cur_weight = average(all_weight_diff_list)


            # Save models at specific rounds
            if each_round in save_round:
                model_name1 = os.path.join(save_path, defense_method + '_2_' + str(each_round))
                model_name2 = os.path.join(save_path, defense_method + '_3_' + str(each_round))
                torch.save(model_list[2].state_dict(), model_name1 + '.pt')
                torch.save(model_list[3].state_dict(), model_name2 + '.pt')

            for idx in range(benign_client_number):
                benign_client_id = idx + backdoor_client_number
                load_model_weight_diff(
                    model=model_list[benign_client_id],
                    weight_diff=cur_weight,
                    global_weight=model_list_old[benign_client_id])

    # Save results
    file_path = os.path.join(save_path, 'result.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(whole_init_record_list, f)

    # Final alert in wandb
    wandb.alert(
        title="IDS code execution finished!",
        text="IDS code execution has finished",
        level=AlertLevel.WARN,
        wait_duration=1, )
