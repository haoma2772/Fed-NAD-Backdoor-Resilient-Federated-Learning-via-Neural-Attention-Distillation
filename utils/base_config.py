




triggers_dir = './triggers' # default triggers directory

# default target class (without loss of generality)
# 投毒的源类别
# source_class = [0,1,2,3,4,5,6,7,8,9]          #||| default source class for TaCT
# cover_classes = [0,1,2,3,4,5,6,7,8,9]      #||| default cover classes for TaCT
source_class = [1,2,5,9]          #||| default source class for TaCT
cover_classes = [6,7]      #||| default cover classes for TaCT

record_poison_seed = True
record_model_arch = False


target_class = {
    'mnist' : 5,
    'emnist' : 5,
    'cifar10' : 5,
    'gtsrb' : 2,
    # 'gtsrb' : 12, # BadEncoder
    'tiny_imagenet': 0,
    'imagenet' : 0,
}



trigger_default = {
    'mnist': {
        'none' : 'none',
        'adaptive':'none',
        'adaptive_blend': 'hellokitty_32.png',
        'adaptive_patch': 'none',
        'adaptive_k_way': 'none',
        'clean_label' : 'badnet_patch4_dup_32.png',
        'basic' : 'badnet_patch_32.png',
        'badnet' : 'badnet_patch_32.png',
        'blend' : 'hellokitty_32.png',
        'refool': 'none',
        'TaCT' : 'trojan_square_32.png',
        'SIG' : 'none',
        'WaNet': 'none',
        'dynamic' : 'none',
        'ISSBA': 'none',
        'SleeperAgent': 'none',
        'badnet_all_to_all' : 'badnet_patch_32.png',
        'trojannn': 'none',
        'BadEncoder': 'none',
        'SRA': 'phoenix_corner_32.png',
        'trojan': 'trojan_square_32.png',
        'bpp': 'none',
        'WB': 'none',
        'DBA': 'none',
    },
    'emnist': {
        'none' : 'none',
        'adaptive':'none',
        'adaptive_blend': 'hellokitty_32.png',
        'adaptive_patch': 'none',
        'adaptive_k_way': 'none',
        'clean_label' : 'badnet_patch4_dup_32.png',
        'basic' : 'badnet_patch_32.png',
        'badnet' : 'badnet_patch_32.png',
        'blend' : 'hellokitty_32.png',
        'refool': 'none',
        'TaCT' : 'trojan_square_32.png',
        'SIG' : 'none',
        'WaNet': 'none',
        'dynamic' : 'none',
        'ISSBA': 'none',
        'SleeperAgent': 'none',
        'badnet_all_to_all' : 'badnet_patch_32.png',
        'trojannn': 'none',
        'BadEncoder': 'none',
        'SRA': 'phoenix_corner_32.png',
        'trojan': 'trojan_square_32.png',
        'bpp': 'none',
        'WB': 'none',
        'DBA': 'none',
    },
    'cifar10': {
        'none' : 'none',
        'adaptive':'none',
        'adaptive_blend': 'hellokitty_32.png',
        'adaptive_patch': 'none',
        'adaptive_k_way': 'none',
        'clean_label' : 'badnet_patch4_dup_32.png',
        'basic' : 'badnet_patch_32.png',
        'badnet' : 'badnet_patch_32.png',
        'blend' : 'hellokitty_32.png',
        'refool': 'none',
        'TaCT' : 'trojan_square_32.png',
        'SIG' : 'none',
        'WaNet': 'none',
        'dynamic' : 'none',
        'ISSBA': 'none',
        'SleeperAgent': 'none',
        'badnet_all_to_all' : 'badnet_patch_32.png',
        'trojannn': 'none',
        'BadEncoder': 'none',
        'SRA': 'phoenix_corner_32.png',
        'trojan': 'trojan_square_32.png',
        'bpp': 'none',
        'WB': 'none',
        'DBA': 'none',
    },
    'gtsrb': {
        'none' : 'none',
        'adaptive_blend': 'hellokitty_32.png',
        'adaptive_patch': 'none',
        'adaptive_k_way': 'none',
        'clean_label' : 'badnet_patch4_dup_32.png',
        'basic' : 'badnet_patch_32.png',
        'badnet' : 'badnet_patch_32.png',
        'blend' : 'hellokitty_32.png',
        'refool': 'none',
        'TaCT' : 'trojan_square_32.png',
        'SIG' : 'none',
        'WaNet': 'none',
        'dynamic' : 'none',
        'ISSBA': 'none',
        'SleeperAgent': 'none',
        'badnet_all_to_all' : 'badnet_patch_32.png',
        'trojannn': 'none',
        'BadEncoder': 'none',
        'SRA': 'phoenix_corner_32.png',
        'trojan': 'trojan_square_32.png',
        'DBA': 'none',
    },
    'imagenet': {
        'none': 'none',
        'badnet': 'badnet_patch_256.png',
        'blend' : 'hellokitty_224.png',
        'trojan' : 'trojan_watermark_224.png',
        'SRA': 'phoenix_corner_256.png',
        'DBA': 'none',
    },
    'tiny_imagenet': {
        'none': 'none',
        'badnet': 'badnet_patch_256.png',
        'blend' : 'hellokitty_224.png',
        'trojan' : 'trojan_watermark_224.png',
        'SRA': 'phoenix_corner_256.png',
        'DBA': 'none',
    }
}




# adapitve-patch triggers for different datasets
adaptive_patch_train_trigger_names = {
    'mnist': [
        'phoenix_corner_32.png',
        'firefox_corner_32.png',
        'badnet_patch4_32.png',
        'trojan_square_32.png',
    ],
    'emnist': [
        'phoenix_corner_32.png',
        'firefox_corner_32.png',
        'badnet_patch4_32.png',
        'trojan_square_32.png',
    ],
    'cifar10': [
        'phoenix_corner_32.png',
        'firefox_corner_32.png',
        'badnet_patch4_32.png',
        'trojan_square_32.png',
    ],
    'gtsrb': [
        'phoenix_corner_32.png',
        'firefox_corner_32.png',
        'badnet_patch4_32.png',
        'trojan_square_32.png',
    ],
}

adaptive_patch_train_trigger_alphas = {
    'mnist': [
        0.5,
        0.2,
        0.5,
        0.3,
    ],
    'emnist': [
        0.5,
        0.2,
        0.5,
        0.3,
    ],
    'cifar10': [
        0.5,
        0.2,
        0.5,
        0.3,
    ],
    'gtsrb': [
        0.5,
        0.2,
        0.5,
        0.3,
    ],
}

adaptive_patch_test_trigger_names = {
    'mnist': [
        'phoenix_corner2_32.png',
        'badnet_patch4_32.png',
    ],
    'emnist': [
        'phoenix_corner2_32.png',
        'badnet_patch4_32.png',
    ],
    'cifar10': [
        'phoenix_corner2_32.png',
        'badnet_patch4_32.png',
    ],
    'gtsrb': [
        'firefox_corner_32.png',
        'trojan_square_32.png',
    ],
}

adaptive_patch_test_trigger_alphas = {
    'mnist': [
        1,
        1,
    ],
    'emnist': [
        1,
        1,
    ],
    'cifar10': [
        1,
        1,
    ],
    'gtsrb': [
        1,
        1,
    ],
}

DBA_poison_patterns_num = 4
DBA_poison_patterns = {
    # 'cifar10':{
    #         "0_poison_pattern": [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
    #         "1_poison_pattern": [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]],
    #         "2_poison_pattern": [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]],
    #         "3_poison_pattern": [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]],
    #         },
    'cifar10':
            {
                '0_poison_pattern': [[x, y] for x in range(21) for y in range(21)],
                '1_poison_pattern': [[x, y] for x in range(30,51) for y in range(30,51)],
                '2_poison_pattern': [[x, y] for x in range(30,51) for y in range(21)],
                '3_poison_pattern': [[x, y] for x in range(21) for y in range(30,51)],
            },
    'tiny_imagenet':
            {   
                '0_poison_pattern': [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2], [0, 3], [1, 3], [0, 4], [1, 4], [0, 5], [1, 5], [0, 6], [1, 6], [0, 7], [1, 7], [0, 8], [1, 8], [0, 9], [1, 9]],
                '1_poison_pattern': [[0, 12], [1, 12], [0, 13], [1, 13], [0, 14], [1, 14], [0, 15], [1, 15], [0, 16], [1, 16], [0, 17], [1, 17], [0, 18], [1, 18], [0, 19], [1, 19], [0, 20], [1, 20], [0, 21], [1, 21]],
                '2_poison_pattern': [[4, 0], [5, 0], [4, 1], [5, 1], [4, 2], [5, 2], [4, 3], [5, 3], [4, 4], [5, 4], [4, 5], [5, 5], [4, 6], [5, 6], [4, 7], [5, 7], [4, 8], [5, 8], [4, 9], [5, 9]],
                '3_poison_pattern': [[4, 12], [5, 12], [4, 13], [5, 13], [4, 14], [5, 14], [4, 15], [5, 15], [4, 16], [5, 16], [4, 17], [5, 17], [4, 18], [5, 18], [4, 19], [5, 19], [4, 20], [5, 20], [4, 21], [5, 21]],
            },

    'mnist':{
            "0_poison_pattern": [[0, 0], [0, 1], [0, 2], [0, 3]],
            "1_poison_pattern": [[0, 6], [0, 7], [0, 8], [0, 9]],
            "2_poison_pattern": [[3, 0], [3, 1], [3, 2], [3, 3]],
            "3_poison_pattern": [[3, 6], [3, 7], [3, 8], [3, 9]],
        },
    'emnist':{
            "0_poison_pattern": [[0, 0], [0, 1], [0, 2], [0, 3]],
            "1_poison_pattern": [[0, 6], [0, 7], [0, 8], [0, 9]],
            "2_poison_pattern": [[3, 0], [3, 1], [3, 2], [3, 3]],
            "3_poison_pattern": [[3, 6], [3, 7], [3, 8], [3, 9]],
        },
    'gtsrb':{
            "0_poison_pattern": [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]],
            "1_poison_pattern": [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]],
            "2_poison_pattern": [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]],
            "3_poison_pattern": [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]],
            },
}


