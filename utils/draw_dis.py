from utils.utility import load_config
import math

from utils.plugin import get_dataset
from utils.utility import load_config
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

if __name__ == '__main__':

    path = 'config.yaml'
    config = load_config(path)

    global_round = config['federated_learning']['round']  # Number of global rounds
    client_number = config['federated_learning']['client_number']  # Total number of clients
    split_rate = config['federated_learning']['split_rate']  # Data split rate
    poisoned_rate = config['general']['poisoned_rate']  # Rate of poisoned data
    select_number = config['general']['select_num']  # Number of selected clients
    dirichlet_rate = config['federated_learning']['dirichlet_rate']  # Dirichlet distribution rate for non-IID
    backdoor_rate = config['federated_learning']['backdoor_rate']  # Backdoor client rate
    batch_size = config['dataset']['batch_size']  # Batch size for data loading
    num_workers = config['dataset']['num_workers']  # Number of workers for data loading
    lr = config['device']['learning_rate']  # Learning rate
    runs_time = config['general']['runs_time']  # Number of runs
    dataset_name = config['model']['name']  # Model name
    model_name = config['dataset']['name']  # Dataset name
    backdoor_type = config['poisoned_paras']['poisoned_type']  # Type of backdoor attack
    defense_method = config['general']['denfense']  # Defense method
    repeated_times = config['general']['runs_time']  # Number of times to repeat the experiment
    poisoned_method = config['poisoned_paras']['poisoned_type']  # Poisoned data method
    epoch_time = config['device']['epoch']  # Number of epochs per client

    tag_name = poisoned_method + '_' + "{}*{}".format(global_round, epoch_time)  # Tag for experiment
    dl_rate = "noniid={}".format(dirichlet_rate)  # Non-IID data distribution rate

    for each_repeat in range(repeated_times):

        backdoor_client_number: int = max(0, math.floor(client_number * backdoor_rate))  # Calculate number of backdoor clients
        backdoor_client_index: list = list(range(0, backdoor_client_number))  # Indexes of backdoor clients

        # Load datasets for training and testing
        train_dataset_list, clean_test_dataset_list, global_dataset = get_dataset(
            backdoor_index=backdoor_client_index, config=config)

        # Create DataLoader for each client for training and testing
        dataset_train_loader_list = [DataLoader(train_dataset_list[idx], 
                                                batch_size=batch_size, shuffle=True, num_workers=num_workers) for idx in range(client_number)]
        dataset_test_loader_list = [DataLoader(clean_test_dataset_list[idx], 
                                               batch_size=batch_size, shuffle=False, num_workers=num_workers) for idx in range(client_number)]

        num_classes = 10  # Number of classes in the dataset
        client_labels = []  # List to store label distribution for each client

        # Count the label distribution for each client in the training data
        for i, classes in enumerate(dataset_train_loader_list):
            label_counts = [0] * num_classes  
            for batch_idx, (_, lab) in enumerate(dataset_train_loader_list[i]):
                for label in lab.tolist():
                    label_counts[label] += 1
            client_labels.append(label_counts)

        # Count the label distribution for each client in the testing data
        for i, classes in enumerate(dataset_test_loader_list):
            label_counts = [0] * num_classes 
            for batch_idx, (_, lab) in enumerate(dataset_test_loader_list[i]):
                for label in lab.tolist():
                    label_counts[label] += 1
            client_labels.append(label_counts)

        import matplotlib.pyplot as plt
        # Create a new plot
        # fig, ax = plt.subplots(figsize=(12,8))  # Optional for a larger figure size

        # Color map for each label class
        colors = plt.cm.tab10.colors
        plt.style.use('ggplot')  # Use 'ggplot' style for the plot

        # Create a figure to visualize the label distribution across clients
        fig, ax = plt.subplots(figsize=(12,8))
        num_clients = 10  # Total number of clients
        for i in range(num_clients):
            drawn_length = 0  # Track the length already drawn for each client
            # Iterate over the label counts for each client
            for j, label_count in enumerate(client_labels[i]):
                # Draw a horizontal line for each label's count, using different colors
                plt.hlines(y=i, xmin=drawn_length, xmax=drawn_length + label_count, linewidth=15, color=colors[j],
                           alpha=0.7)
                drawn_length += label_count  # Update the drawn length

        ax.set_xlabel('Sample Count', color='black', fontsize=20)  # X-axis label
        ax.set_ylabel('Client Index', color='black', fontsize=20)  # Y-axis label
        ax.set_xticks(np.arange(0, 12000, 1000))  # Adjust the x-axis ticks as needed
        ax.set_yticks(np.arange(num_clients))  # Set Y-axis ticks to correspond to the number of clients
        ax.set_xticklabels(ax.get_xticks(), color='black', fontsize=16)  # Set x-tick label styles
        ax.set_yticklabels(ax.get_yticks(), color='black', fontsize=16)  # Set y-tick label styles
        
        # Create legend labels for each class
        legend_labels = ['Class {}'.format(i) for i in range(len(client_labels[0]))]
        ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 0.7), fontsize=16)

        # Adjust the layout to fit the legend and labels
        plt.tight_layout()
        pic_name = 'distribution_alpha=' + str(dirichlet_rate) + '.pdf'  # Save the plot as a PDF file
        plt.savefig(pic_name, bbox_inches='tight')  # Save the plot
        plt.show()  # Show the plot

        # plt.show()  # Optional if you want to display the plot interactively
        # # Generate sample image (not needed if using plt.savefig)
        # image = plt.imread(pic_name)  # Read and display the saved image
