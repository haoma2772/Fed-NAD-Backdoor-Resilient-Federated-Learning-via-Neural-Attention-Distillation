import torch
import numpy as np


def average(all_weight_diff_list):
        # Check if the list is empty
    if len(all_weight_diff_list) == 0:
        return None

    # Sum all tensors in the list
    total_sum = sum(all_weight_diff_list)

    # Calculate the average of all tensors in the list
    average_tensor = total_sum / len(all_weight_diff_list)

    return average_tensor


def TrimmedMean(all_weight_diff_list, client_num, attacker_num):

    sta_idx = attacker_num
    end_idx = client_num - attacker_num

    if sta_idx >= end_idx:
        raise ValueError("The number of attackers does not satisfy the requirement")

    # Stack all tensors to create a single tensor
    stacked_tensor = torch.stack(all_weight_diff_list)
    
    # Sort the tensor along the specified dimension
    sort_tensor, indices = torch.sort(stacked_tensor, dim=0)

    # Select the part of the tensor that is not affected by attackers
    avg_tensor = sort_tensor[sta_idx:end_idx]
    
    # Compute the mean of the remaining tensors
    avg_tensor = torch.mean(avg_tensor.float(), dim=0)

    return avg_tensor


def euclidean_distance(vector1, vector2):
    
    # Compute the Euclidean distance between two vectors
    return torch.norm(vector1 - vector2, p=2)

def compute_scores(distances, i, n, f):

    # Collect the distances of node i to all other nodes
    s = [distances[j][i] for j in range(i)] + [
            distances[i][j] for j in range(i + 1, n)
        ]
        
    # Sort the list s and select the smallest n - f - 2 distances
    _s = sorted(s)[:n - f - 2]
        
    # Return the sum of the squared selected distances as node i's Krum score
    return sum(_s)


def KrumMean(all_weight_diff_list, client_num, attacker_num, m):

    distances = {}
    
    # Compute pairwise distances between all clients
    for i in range(client_num-1):
        distances[i] = {}
        for j in range(i + 1, client_num):
            distances[i][j] = euclidean_distance(all_weight_diff_list[i], all_weight_diff_list[j])
        
    # Compute scores for all clients based on the distances
    scores = [(i, compute_scores(distances, i, client_num, attacker_num)) for i in range(client_num)]
    
    # Sort clients based on their scores
    sorted_scores = sorted(scores, key=lambda x: x[1])
    
    # Select the top m clients based on the smallest scores
    top_k_indices = list(map(lambda x: x[0], sorted_scores))[:m]

    # Select the weight differences from the top m clients
    sele_weight_diff_list = [all_weight_diff_list[i] for i in top_k_indices]
    
    # Calculate the average weight difference from the selected clients
    res_weight_diff = average(sele_weight_diff_list)

    return res_weight_diff


def Median(all_weight_diff_list):

    # Stack all tensors and compute the median
    stacked_tensor = torch.stack(all_weight_diff_list)
    median_tensor, _ = torch.median(stacked_tensor, dim=0)

    return median_tensor


def GeoMed(all_weight_diff_list):

    from model_defense import geometric_median
    # Convert the weight differences to a numpy array
    vectors = np.array(all_weight_diff_list).astype(np.float32)

    # Compute the geometric median of the weight differences
    median = geometric_median(vectors)
    
    # Convert the result back to a tensor
    res_tensor = torch.from_numpy(median.astype(np.float32))

    return res_tensor
