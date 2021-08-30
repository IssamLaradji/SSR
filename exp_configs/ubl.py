from haven import haven_utils as hu
import copy
EXP_GROUPS = {}

def get_base(ddict):
    base = {"batch_size": 64,
        
        "dataset_directory": "/mnt/public/datasets2/softras/mesh_reconstruction",
        "image_size": 64,
       
        "lambda_flatten": 0.0005,
        "lambda_laplacian": 0.005,
        "learning_rate": 0.0001,
        "lr_type": "step",
        "model_directory": "/mnt/public/datasets2/softras/results/models",
        "reset": 1,
        "seed": 0,
        "sigma_val": 0.0001,
        }
    base.update(ddict)
    return base

######

EXP_GROUPS['upperbound'] =  hu.cartesian_exp_group(get_base(
        {"loss":[
               [{"name":'base'}],
               
                
                ],
        #  "n_train_ratio":[None
        # ],

        # "mem":[True],
         "n_val_ratio":[0.10],
         'epochs':1000,
        "version":[4],
        # "pretrained":[None],
        "classes": [['Airplane'],
       ['Car'], ['Chair'], 
       ['Cabinet'],
       ['Bench'], ['Display'], ['Lamp'],
       ['Loudspeaker'], ['Rifle'],
       ['Sofa'], ['Table'], ['Telephone'],
       ['Watercraft']
        ],
        }), remove_none=True)

ubl_dict = get_base(
        {"loss":[

                #  [{'name':'ubl_cheat', 'p':'auto'}],
                 [{"name":'vp_cons'}, {'name':'base_vp_cons'}],
                # [{"name":'vp_cons'}, {'name':'base'}],
            #    [{"name":'base_ubl', 'cheat':True}],
               [{"name":'base'}],
               
            
            ],
        "n_train_objects":[
            2,
         5, 
          20
    #  0.15
    ],
        "train_ubl":[None],
'augment_loss':[True, None],
    # "mem":[True],
        "n_val_ratio":[0.10],
        'epochs':1000,
    "version":[ None,     1],
    # "pretrained":[None],
    "classes": [
    ['Airplane'],
    ['Chair'], 
    ['Car'],        
    ['Display'],
    ['Cabinet'],
    ['Bench'],  
    ['Lamp'],
    ['Loudspeaker'], 
    ['Rifle'],
    ['Sofa'], 
    ['Table'],
     ['Telephone'],
    ['Watercraft']
    ],
    })

EXP_GROUPS['ubl'] =  hu.cartesian_exp_group(ubl_dict, remove_none=True)
ubl_real_val = copy.deepcopy(ubl_dict)
ubl_real_val['n_val_ratio'] = None
ubl_real_val['augment_loss'] = True
ubl_real_val['version'] = None

EXP_GROUPS['ubl_real'] =  hu.cartesian_exp_group(ubl_real_val, remove_none=True)
qual_dict = copy.deepcopy(ubl_dict)
qual_dict['loss'] = [
                 [{"name":'vp_cons'}, {'name':'base_vp_cons'}],
                # [{"name":'base'}]
                ]
qual_dict['classes'] = [['Airplane'],['Car'],['Chair']]
qual_dict['n_train_objects'] = [2]
EXP_GROUPS['qualitative'] =  hu.cartesian_exp_group(qual_dict, remove_none=True)




EXP_GROUPS['ablation'] =  hu.cartesian_exp_group(get_base(
        {"loss":[
                 [{"name":'vp_cons'}, {'name':'base_vp_cons'}],
                
                ],
         "n_train_objects":[2,
        ],
        ############################
        # "augment_pred":[False, None],
        # "augment_loss":[False, None],
        # "train_ubl":[False, None],
        # # "always_ubl":[True, None],
        # "predict_prob":[0., None],
        # "pseudo_cycle":[None, 1000, ],
        "pseudo_batch_size":[8, 10, 16, None, 48],
        ############################

        # "mem":[True],
         "n_val_ratio":[0.10],
         'epochs':1000,
       
        # "pretrained":[None],
        # "classes": [['Airplane'], ['Chair'], 
        "classes":[
    #    ['Car'], ['Chair'], 
       ['Airplane'],
        ],
    #    ['Cabinet'],
    #    ['Bench'], ['Display'], ['Lamp'],
    #    ['Loudspeaker'], ['Rifle'],
    #    ['Sofa'], ['Table'], ['Telephone'],
    #    ['Watercraft']
        # ],
        }), remove_none=True)