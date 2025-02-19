import math

from utils.plugin import get_dataset
from utils.utility import load_config
from torch.utils.data import DataLoader
import numpy as np
from wandb import AlertLevel


if __name__ == '__main__':

    # Load configuration settings from the YAML file
    path = 'config.yaml'
    config = load_config(path)

    # Extract various parameters for training
    global_round = config['federated_learning']['round']  # Number of global rounds
    client_number = config['federated_learning']['client_number']  # Total number of clients
    split_rate = config['federated_learning']['split_rate']  # Data split rate
    poisoned_rate = config['general']['poisoned_rate']  # Rate of poisoned data
    select_number = config['general']['select_num']  # Number of clients selected for each round
    dirichlet_rate = config['federated_learning']['dirichlet_rate']  # Dirichlet distribution rate (for non-IID data)
    backdoor_rate = config['federated_learning']['backdoor_rate']  # Rate of backdoor clients
    batch_size = config['dataset']['batch_size']  # Batch size for training and testing
    num_workers = config['dataset']['num_workers']  # Number of workers for data loading
    lr = config['device']['learning_rate']  # Learning rate
    runs_time = config['general']['runs_time']  # Number of repeated runs
    dataset_name = config['model']['name']  # Model name
    model_name = config['dataset']['name']  # Dataset name
    backdoor_type = config['poisoned_paras']['poisoned_type']  # Type of backdoor attack
    defense_method = config['general']['denfense']  # Defense method to be used
    repeated_times = config['general']['runs_time']  # Number of repetitions
    poisoned_method = config['poisoned_paras']['poisoned_type']  # Poisoning method used
    epoch_time = config['device']['epoch']  # Number of epochs per client

    tag_name = poisoned_method + '_' + "{}*{}".format(global_round, epoch_time)  # Tag to track the experiment
    dl_rate = "noniid={}".format(dirichlet_rate)  # Label to indicate whether data is IID or non-IID
    tags = []
    tags.append(tag_name)
    tags.append(dl_rate)

    # Define a set of colors for plotting
    color1 = (203, 180, 123)
    color2 = (91, 183, 205)
    color3 = (71, 120, 185)
    color4 = (42, 157, 142)
    color5 = (197, 86, 89)
    color6 = (255, 158, 2)
    color7 = (68, 4, 90)
    color8 = (183, 181, 160)
    color9 = (107, 112, 92)
    color10 = (254, 179, 174)

    from matplotlib.colors import to_rgba

    # Convert RGB colors to RGBA format and add them to the color list
    color = []
    color.append(to_rgba((color1[0] / 255, color1[1] / 255, color1[2] / 255), alpha=1))
    color.append(to_rgba((color2[0] / 255, color2[1] / 255, color2[2] / 255), alpha=1))
    color.append(to_rgba((color3[0] / 255, color3[1] / 255, color3[2] / 255), alpha=1))
    color.append(to_rgba((color4[0] / 255, color4[1] / 255, color4[2] / 255), alpha=1))
    color.append(to_rgba((color5[0] / 255, color5[1] / 255, color5[2] / 255), alpha=1))
    color.append(to_rgba((color6[0] / 255, color6[1] / 255, color6[2] / 255), alpha=1))
    color.append(to_rgba((color7[0] / 255, color7[1] / 255, color7[2] / 255), alpha=1))
    color.append(to_rgba((color8[0] / 255, color8[1] / 255, color8[2] / 255), alpha=1))
    color.append(to_rgba((color9[0] / 255, color9[1] / 255, color9[2] / 255), alpha=1))
    color.append(to_rgba((color10[0] / 255, color10[1] / 255, color10[2] / 255), alpha=1))

    # Loop over the number of repeated experiments
    for each_repeat in range(repeated_times):

        # Calculate the number of backdoor clients
        backdoor_client_number: int = max(1, math.floor(client_number * backdoor_rate))
        # Create a list of indices for the backdoor clients
        backdoor_client_index: list = list(range(0, backdoor_client_number))

        # Load the datasets (training and clean test datasets)
        train_dataset_list, clean_test_dataset_list = get_dataset(
            backdoor_index=backdoor_client_index, config=config)

        # Create DataLoader objects for each client
        dataset_train_loader_list = [
             DataLoader(train_dataset_list[client_id], batch_size=batch_size, num_workers=num_workers, shuffle=True) for
             client_id in range(client_number)]
        dataset_test_loader_list = [
             DataLoader(clean_test_dataset_list[client_id], batch_size=batch_size, num_workers=num_workers,
                        shuffle=True)
             for
             client_id in range(client_number)]

        # IDs of two specific clients to visualize their data distribution
        first_id = 5
        second_id = 8

        import matplotlib.pyplot as plt
        # Get the labels of the training and testing data for the first client
        first_id_train_labels = np.concatenate([y.numpy() for _, y in dataset_train_loader_list[first_id]])
        first_id_test_labels = np.concatenate([y.numpy() for _, y in dataset_test_loader_list[first_id]])

        # Get the labels of the training and testing data for the second client
        second_id_train_labels = np.concatenate([y.numpy() for _, y in dataset_train_loader_list[second_id]])
        second_id_test_labels = np.concatenate([y.numpy() for _, y in dataset_test_loader_list[second_id]])

        # Class names (CIFAR-10 dataset)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # Calculate the number of samples in each class for the first client
        first_id_class_counts = [np.sum(first_id_train_labels == i) + np.sum(first_id_test_labels == i) for i in range(10)]
        # Calculate the number of samples in each class for the second client
        second_id_class_counts = [np.sum(second_id_train_labels == i) + np.sum(second_id_test_labels == i) for i in
                                  range(10)]


        # Function to format the percentage labels on the pie chart
        def my_autopct(pct):
            if pct < 0.01:
                return ''
            else:
                return f'{pct:.1f}%'

        # Plot the first pie chart for the first client
        plt.style.use('ggplot')
        plt.figure(figsize=(5, 4))
        plt.subplots_adjust(left=0, right=0.8, top=1, bottom=0)
        plt.pie(first_id_class_counts, autopct=my_autopct, startangle=90, colors=color)
        plt.legend(class_names, title="Classes", loc="center left", bbox_to_anchor=(1, 0.5))
        # Save the pie chart as a PDF
        plt.savefig(f'client_{first_id}_class_distribution.pdf', bbox_inches='tight')
        plt.close()  # Close the figure to free memory

        # Plot the second pie chart for the second client
        plt.style.use('ggplot')
        plt.figure(figsize=(5, 4))
        plt.subplots_adjust(left=0, right=0.8, top=1, bottom=0)
        plt.pie(second_id_class_counts, autopct=my_autopct, startangle=90, colors=color)
        plt.legend(class_names, title="Classes", loc="center left", bbox_to_anchor=(1, 0.5))
        # Save the pie chart as a PDF
        plt.savefig(f'client_{second_id}_class_distribution.pdf', bbox_inches='tight')
        plt.close()  # Close the figure to free memory
