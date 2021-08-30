import argparse
from haven import haven_utils as hu
from haven import haven_wizard as hw
import torch
import numpy as np
import pandas as pd
from argparse import Namespace
import soft_renderer as sr
import copy
from src import losses
from src import datasets
from src import models
from pytorch3d.structures import Meshes
import time, tqdm
import os



def trainval(exp_dict, savedir, args):
    exp_args = Namespace(**exp_dict)
    # randomness
    # torch.backends.cudnn.deterministic = True
    torch.manual_seed(exp_args.seed)
    torch.cuda.manual_seed_all(exp_args.seed)
    np.random.seed(exp_args.seed)

    # load dataset and model
    dataset_train = datasets.ShapeNet(args.datadir, 'train', exp_dict)
    dataset_val = datasets.ShapeNet(args.datadir, 'val', exp_dict)
    dataset_test = datasets.ShapeNet(args.datadir, 'test', exp_dict)
    dataset_unlabeled = datasets.ShapeNet(args.datadir, 'unlabeled', exp_dict)

    model = models.get_model(exp_dict, train_set=dataset_train, 
                             unlabeled_set=dataset_unlabeled)

    # load checkpoint
    chk_dict = hw.get_checkpoint(savedir, return_model_state_dict=True)

    if len(chk_dict['model_state_dict']):
        # resume checkpoint
        model.set_state_dict(chk_dict['model_state_dict'])
        print(f'Resuming from checkpoint at epoch: {chk_dict["epoch"]}')

    

    if args.debug:
        for i in range(50):
            batch = hu.collate_fn([model.unlabeled_set.__getitem__(im, int(v1), int(v2)) 
                        for (im, v1, v2) in [(i,0,0)]], mode='default')
            images_a = batch['images_a']
            pred_a = model.get_out(images_a.cuda(), 
                            model.train_set.possible_viewpoints[0].cuda())[0]
            path = os.path.join(f'test_results',
                                f'{exp_dict["classes"][0]}_{exp_dict["n_train_objects"]}')
            
            hu.save_image(os.path.join(path, f'{i}.png'), images_a[[0]])                 
            hu.save_image(
                os.path.join(path, 
                   f'{i}_{exp_dict["loss"][0]["name"]}.png'), 
                           pred_a[[0]])

        return

    if exp_dict.get('pretrained'):
        exp_dict_pre = copy.deepcopy(exp_dict)
        del exp_dict_pre['pretrained']
        exp_dict_pre['loss'] = [{'name':'base'}]
        savedir_pre = os.path.join(os.path.split(savedir)[0].replace('debug', 'toolkit'), hu.hash_dict(exp_dict_pre))
        score_list_pre = hu.load_pkl(os.path.join(savedir_pre, 'score_list_best.pkl'))
        model_pre = torch.load(os.path.join(savedir_pre, 'model_best.pth'))
        model.set_state_dict(model_pre)
        print(f'pretrained score: {score_list_pre[-1]["test_iou"]:.2f}')


    model.args = args
    
    # Go through epochs
    for e in range(chk_dict['epoch'], exp_dict['epochs']):
        # setup metrics
        score_dict = {}
        score_dict['epoch'] = e
        
        # Train one epoch
        train_dict = model.train_on_dataset(dataset_train, num_workers=args.num_workers)
        score_dict.update(train_dict)
        score_dict['pseudo_ratio'] = model.meta.get('pseudo_ratio', 0)
        score_dict['pseudo_acc'] = model.meta.get('pseudo_acc', -1)
        score_dict['n_train'] = len(dataset_train)

        # Val on batch
        val_dict = model.val_on_dataset(dataset_val, savedir_images=os.path.join(savedir, 'images'))
        for k in val_dict:
            score_dict[f'val_{k}'] = val_dict[k]

        if (e % exp_dict['epochs'] == 0) or (val_dict['iou'] > pd.DataFrame(chk_dict['score_list'])['val_iou'].max()):
            test_dict = model.val_on_dataset(dataset_test, savedir_images=os.path.join(savedir, 'images'))
            for k in test_dict:
                score_dict[f'test_{k}'] = test_dict[k]

            chk_dict['score_list'] += [score_dict]
            # save best checkpoint
            hw.save_checkpoint(savedir, 
                            fname_suffix='_best',
                            model_state_dict=model.get_state_dict(),
                            score_list=chk_dict['score_list'])

        else:
            # Save metrics
            chk_dict['score_list'] += [score_dict]

        # Report and save results
        hw.save_checkpoint(savedir, 
                            model_state_dict=model.get_state_dict(),
                            score_list=chk_dict['score_list'])



if __name__ == "__main__":
    import exp_configs

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', default=None)
    parser.add_argument('-nw', '--num_workers', default=0, type=int)
    parser.add_argument("-db", "--debug",  default=0, type=int)
    parser.add_argument("-d", "--datadir",  default=None)
    parser.add_argument("-r", "--reset",  default=0, type=int,
                        help='Reset or resume the experiment.')
    args, others = parser.parse_known_args()


    if not os.path.exists('job_configs.py'):
        job_config = None 
    else:
        import job_configs
        job_config = job_configs.JOB_CONFIG

    # 9. Launch experiments using magic command
    if args.exp_group_list is not None:
        # Get List of experiments
        exp_list = [e for group in args.exp_group_list for e in exp_configs.EXP_GROUPS[group]]
    else:
        exp_list = None 

    hw.run_wizard(func=trainval,
                 exp_list=exp_list,
                 python_binary_path='nvidia-smi && /mnt/home/miniconda37/bin/python',
                 job_config=job_config, use_threads=1, args=args,
                 results_fname='results.ipynb')
