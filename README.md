# Fed-NAD: Backdoor-Resilient Federated Learning via Neural Attention Distillation

## 📚 Abstract
Federated Learning (FL) has emerged as a distributed machine learning paradigm that enables training a global model across multiple devices while preserving data privacy. However, its decentralized nature introduces backdoor vulnerabilities, where malicious participants can collaboratively poison the global model by carefully scaling their shared models. 

In this paper, we propose **Fed-NAD**, a backdoor-resilient FL framework. Fed-NAD leverages **Neural Attention Distillation (NAD)** to enable benign clients to purify the backdoored global model during local training. The process unfolds in two stages: 
1. Benign clients first train a **teacher network** locally on clean datasets to capture benign input features.
2. The teacher network is then used for **neural attention distillation** on the aggregated backdoored global model, effectively filtering out malicious updates.

This approach ensures that benign clients can collaboratively obtain clean global models, free from backdoors. We evaluate **Fed-NAD** using extensive experiments on the CIFAR-10 dataset with a ResNet-18 architecture, demonstrating a significant decrease in attack success rates (from 30% to 60%), while incurring no more than a 2% reduction in accuracy compared to other defense baselines.

---

## 🚀 Features
- **Backdoor Attack Mitigation**: Protects FL systems from backdoor attacks by filtering out malicious updates.
- **Neural Attention Distillation**: Purifies the global model during aggregation using attention mechanisms.
- **Scalability**: Designed for large-scale Federated Learning systems with multiple clients and large datasets.
- **Open Source**: Fully open-source and available for academic and practical use.
- **Seamless Integration**: Can be easily integrated with existing FL frameworks.

---

## 🗂 File Descriptions

### 🗂 `poison_tool_box/`
This directory contains tools for generating and managing poisoned data used for simulating backdoor attacks.

### 🗂 `poisoned_set/`
Includes scripts and utilities for creating and handling poisoned data sets to evaluate backdoor attacks.


### 🗂 `triggers/`
Contains files related to trigger generation used in backdoor attacks within the federated learning framework. You can design specific triggers according to your purpose.

### 🗂 `utils/`
This folder includes utility scripts and helper functions supporting the core functionality of **Fed-NAD**.

### 📄 `README.md`
Main documentation file that provides an overview of the **Fed-NAD** project, installation steps, and the defense approach.

### 📄 `create_data.py`
Script for generating the required datasets, including both clean and poisoned datasets for experimentation.

### 📄 `draw_poisoned.py`
Visualizes the effect of backdoor attacks, plotting poisoned data and analyzing their impact on model performance.

### 📄 `main.py`
Main entry script to run the **Fed-NAD** experiments, orchestrating the federated learning process with the backdoor defense mechanism.

### 📄 `utils/config.yaml`
Configuration file that specifies hyperparameters, model configurations, defense methods, and other settings for the experiment.

---

## ⚙️ Installation

1. **Install Required Packages**
   
   Run the following command to install the necessary Python dependencies:  
   
  ```bash
     pip install -r requirements.txt
  ```

   
2. **Generate the Dataset**
   
   Use the following command to generate the required datasets:  
   
```bash
   python create_data.py
```

4. **Run the Experiments**  

   - **Using Python Directly**  
     You can directly run the experiments using Python with the desired configuration file:  
  ```bash
    python main.py
  ```
       
5. **Real-Time Result Monitoring with Weights and Biases (WandB)**
   This project leverages WandB for real-time experiment tracking and visualization
  ```bash
      wandb login
```
   
## Citation
   If you use BR-FEEL in your research, please cite our paper:
      
      @INPROCEEDINGS{10590622,
      author={Ma, Hao and Qi, Senmao and Yao, Jiayue and Yuan, Yuan and Zou, Yifei and Yu, Dongxiao},
      booktitle={2024 10th IEEE International Conference on Intelligent Data and Security (IDS)}, 
      title={Fed-NAD: Backdoor Resilient Federated Learning via Neural Attention Distillation}, 
      year={2024},
      volume={},
      number={},
      pages={7-13},
      keywords={Training;Data privacy;Toxicology;Federated learning;Distributed databases;Distance measurement;Data models;Federated Learning;Backdoor Attack;Neural Attention Distillation},
      doi={10.1109/IDS62739.2024.00009}}
    


## Acknowledgements
  
    This repository was developed as part of a research project focused on enhancing the resilience of federated edge learning systems against backdoor attacks. We extend our gratitude to all contributors and the community for their invaluable support and feedback.
 
