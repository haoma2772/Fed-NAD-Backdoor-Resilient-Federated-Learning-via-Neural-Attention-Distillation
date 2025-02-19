
import torch
import random
import copy

# code is modified for  ICLR 2020 paper DBA: Distributed Backdoor Attacks against Federated Learning
# because of our img size = 224, we achieve the simple poison patterns in base_config
# you can modify this poison patters according your self use

def add_pixel_pattern(ori_image, poison_patterns):

        # 这是一个样本单张照片

        image = copy.deepcopy(ori_image)
        for i in range(0,len(poison_patterns)):


            # 将指定位置的三个通道的像素值设置为 0
            image[:, poison_patterns[i][0], poison_patterns[i][1]] = 0

        return image

class poison_generator():

    def __init__(self, img_size, dataset,poison_patterns, poison_rate, path,target_class=0, alpha=0.2, seed=0, ):
        # 反正没保存 随便写了个路径
        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0

        self.alpha = alpha
        self.seed = seed
        self.num_img = len(dataset)
        self.poison_patterns = poison_patterns





    def generate_poisoned_training_set(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        id_set = list(range(0,self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort() # increasing order
        img_set = []
        label_set = []
        pt = 0
        cnt = 0
        poison_id = []


        for i in range(self.num_img):
            img, gt = self.dataset[i]

            # poisoned image
            if pt < num_poison and poison_indices[pt] == i:
                
                poison_id.append(cnt)
                gt = self.target_class
                img = add_pixel_pattern(img, self.poison_patterns)
                pt+=1

            # img_file_name = '%d.png' % cnt
            # img_file_path = os.path.join(self.path, img_file_name)
            # save_image(img, img_file_path)
            # print('[Generate Poisoned Set] Save %s' % img_file_path)
            
            img_set.append(img.unsqueeze(0))
            label_set.append(gt)
            cnt+=1
            
        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)
        poison_indices = poison_id
        # print("Poison indices:", poison_indices)
        return img_set, poison_indices, label_set
    





class poison_transform():

    def __init__(self, img_size, poison_patterns, target_class=0, alpha=0.2):


        self.img_size = img_size
        self.target_class = target_class
        self.alpha = alpha
        self.poison_patterns = poison_patterns


    def transform(self, data, labels):
        # this is a batch
        data, labels = data.clone(), labels.clone()
        
        for idx in range(len(data)):
            data[idx] = add_pixel_pattern(data[idx], self.poison_patterns)

        labels[:] = self.target_class



        return data, labels