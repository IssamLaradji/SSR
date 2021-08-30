import torch, tqdm
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
from . import utils as ut            
from haven import haven_utils as hu
from skimage.transform import warp, AffineTransform

import numpy as np
import random
from matplotlib import pyplot as plt

from skimage import data
from skimage.util import img_as_float
from skimage.feature import (corner_harris, corner_subpix, corner_peaks,
                             plot_matches)
from skimage.transform import warp, AffineTransform
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage.measure import ransac
from kornia.geometry.transform import rotate

def iou(predict, target, eps=1e-6, reduction='mean'):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + eps
    if reduction is None:
        return (intersect / union)
    return (intersect / union).sum() / intersect.nelement()

def iou_loss(predict, target, reduction='mean'):
    return 1 - iou(predict, target, reduction=reduction)

def multiview_iou_loss(predicts, targets_a, targets_b):
    loss = (iou_loss(predicts[0][:, 3], targets_a[:, 3]) + \
            iou_loss(predicts[1][:, 3], targets_a[:, 3]) + \
            iou_loss(predicts[2][:, 3], targets_b[:, 3]) + \
            iou_loss(predicts[3][:, 3], targets_b[:, 3])) / 4
    return loss

# my losses
# ----------------------------------------------------------------
def cons_features_loss(self, y_a, y_b):
        # compute representations
        z_a = self.encoder(y_a) # NxD
        z_b = self.encoder(y_b) # NxD
       
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / (1e-5 + z_a.std(0)) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / (1e-5 + z_b.std(0)) # NxD
        # cross-correlation matrix
        c = torch.mm(z_a.T, z_b) / y_a.shape[0] # DxD
        # loss
        d = c.shape[0]
        c_diff = (c - torch.eye(d).cuda()).pow(2) 
        # DxD
        # multiply off-diagonal elems of c_diff by lambda
        ind = (1-torch.eye(d)).bool()
        c_diff[ind] = c_diff[ind] * 0.001
        loss = c_diff.sum()

        loss = (z_a - z_b).pow(2).mean()
        return loss

def match(img_orig, img_warped):
    img_orig = (img_orig[:3].transpose(0,2).cpu().numpy() > 0).astype('float32')
    img_warped = (img_warped[:3].transpose(0,2).cpu().numpy() > 0).astype('float32')
    img_orig_gray = rgb2gray(img_orig)
    img_warped_gray = rgb2gray(img_warped)
    # extract corners using Harris' corner measure
    coords_orig = corner_peaks(corner_harris(img_orig_gray), threshold_rel=0.001,
                            min_distance=5)
    coords_warped = corner_peaks(corner_harris(img_warped_gray),
                                threshold_rel=0.001, min_distance=5)

    # determine sub-pixel corner position
    coords_orig_subpix = coords_orig
    #corner_subpix(img_orig_gray, coords_orig, window_size=9)
    coords_warped_subpix = coords_warped
    #corner_subpix(img_warped_gray, coords_warped,
    #                                    window_size=9)


    def gaussian_weights(window_ext, sigma=1):
        y, x = np.mgrid[-window_ext:window_ext+1, -window_ext:window_ext+1]
        g = np.zeros(y.shape, dtype=np.double)
        g[:] = np.exp(-0.5 * (x**2 / sigma**2 + y**2 / sigma**2))
        g /= 2 * np.pi * sigma * sigma
        return g


    def match_corner(coord, window_ext=5):
        r, c = np.round(coord).astype(np.intp)
        window_orig = img_orig[r-window_ext:r+window_ext+1,
                                c-window_ext:c+window_ext+1, :]

        # weight pixels depending on distance to center pixel
        weights = gaussian_weights(window_ext, 3)
        weights = np.dstack((weights, weights, weights))

        # compute sum of squared differences to all corners in warped image
        SSDs = []
        for cr, cc in coords_warped:
            window_warped = img_warped[cr-window_ext:cr+window_ext+1,
                                    cc-window_ext:cc+window_ext+1, :]
            SSD = np.sum(weights * (window_orig - window_warped)**2)
            SSDs.append(SSD)

        # use corner with minimum SSD as correspondence
        min_idx = np.argmin(SSDs)
        return coords_warped_subpix[min_idx]


    # find correspondences using simple weighted sum of squared differences
    src = []
    dst = []
    for coord in coords_orig_subpix:
        if np.isnan(coord).sum() > 0:
            continue
        src.append(coord)
        dst.append(match_corner(coord))
    src = np.array(src)
    dst = np.array(dst)


    # estimate affine transform model using all coordinates
    model = AffineTransform()
    model.estimate(src, dst)

    # robustly estimate affine transform model with RANSAC
    model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=1,
                                residual_threshold=2, max_trials=100)
    outliers = inliers == False


    # compare "true" and estimated transform parameters
    return model.scale, model.translation, model.rotation


def opt_single(model, b_ubl, n_iters=1000, vis=True, vp='gt', lr=1e-3):
    if 1:
        self = copy.deepcopy(model)
        param_list = (
            list(self.decoder.fc_centroid.parameters()) +
        list(self.decoder.fc_bias.parameters())
                    )
        opt = torch.optim.Adam(param_list, lr=lr)
        for p in self.encoder.parameters():
                p.requires_grad = False
        for p in self.decoder.parameters():
                p.requires_grad = False
        for p in self.decoder.fc_centroid.parameters():
            p.requires_grad = True
        for p in self.decoder.fc_bias.parameters():
            p.requires_grad = True

        
        # labeled
        import itertools
        b_lbl = hu.collate_fn([self.train_set.__getitem__(i, j) for i, j in 
                    itertools.product((0, 1), range(24))], mode='default')

        images_lbl = b_lbl['images_a'].cuda()
        vp_lbl =  b_lbl['viewpoints_a'].cuda()

        # unlabeled
        
        im_ubl = torch.cat([b_ubl['images_a'].cuda(), images_lbl.cuda()], dim=0)
        #vp_ubl = torch.cat([self.train_set.possible_viewpoints.cuda()[:n], vp_lbl], dim=0)
        if isinstance(vp, str) and vp == 'gt':
            vp = int(b_ubl['viewpoint_id_a'])
        vp_ubl = torch.cat([self.train_set.possible_viewpoints.cuda()[[vp]], vp_lbl], dim=0)
        
    pbar = tqdm.tqdm(range(n_iters))
    for i in pbar:
        features = self.encoder(im_ubl)
        vertices, faces = self.decoder(features.detach())
        self.renderer.transform.set_eyes(vp_ubl)
        out = self.renderer(vertices, faces)
        if i == 0:
            out_init = out[0].detach()
        if vis and (i % 20) == 0:
            vn = 3
            # print(match(out_init, im_ubl[0].detach()))
            # print(match(out[0].detach(), im_ubl[0].detach()))
            hu.save_image('tmp.png', np.vstack([im_ubl[:vn].cpu().numpy(),
                                                out[:vn].detach().cpu().numpy(), 
                                            ]), nrow=3)

        loss = self.lambda_laplacian * self.laplacian_loss(vertices).mean()
        loss += self.lambda_flatten *self.flatten_loss(vertices).mean()
        
        loss += iou_loss(out[:, 3], im_ubl[:, 3], reduction='mean')
        our_loss = iou_loss(out[:1, 3], im_ubl[:1, 3], reduction='mean')
        
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_description(f'{i} - loss: {float(our_loss):.3f} - '
                f'{int(vp)}, {int(b_ubl["viewpoint_id_a"])}')

    return our_loss, int(vp), int(b_ubl['viewpoint_id_a'])

def out_sig(self, im_ubl, im_lbl):
    n_samples = im_ubl.shape[0] // 24
    f_ubl = self.vp_siam.forward_one(im_ubl) 
    f_lbl = self.vp_siam.forward_one(im_lbl) 
    d_vp1 = torch.sigmoid(self.vp_siam.out(torch.abs(f_ubl[:, None] - 
                            f_lbl[None]))).squeeze()
    pred_vp_vals, pred_vp = d_vp1.max(dim=1)
    pred_vp_vals = pred_vp_vals.view(n_samples ,24)
    pred_vp = pred_vp.view(n_samples ,24)
    
    return pred_vp, pred_vp_vals

def predict_vps(self, images_lbl, n_start=0, n_samples=100, prob=0.5):
    with torch.no_grad():
        n_samples = min(n_samples, len(self.unlabeled_set) - n_start)
        im_ubl = self.unlabeled_set.images[n_start:n_start+n_samples].view(n_samples* 24, 4, 64, 64).cuda()
        im_lbl = images_lbl.cuda()

        # non rot
        pred_vp, pred_vp_vals = out_sig(self, im_ubl, im_lbl)
        ind = pred_vp_vals>prob
        preds = pred_vp[ind]
        img_list, vp_list = n_start+torch.where(ind)[0], torch.where(ind)[1]
        

        # rot
        rot_pred_vp, rot_pred_vp_vals = out_sig(self, rotate(im_ubl, torch.as_tensor(45.).cuda(),
                     align_corners=True), 
                     rotate(im_lbl, torch.as_tensor(45.).cuda(),
                     align_corners=True))
        ind = rot_pred_vp_vals>prob
        rot_preds = rot_pred_vp[ind]


        # gt_vp = torch.arange(24).repeat(n_samples).cuda().view(n_samples ,24)
        
        # acc = float((gt_vp[ind] == preds).float().mean())
        # print(f'{n_start} - vp acc: {acc:.2f} - {gt_vp[ind].shape[0]}/{gt_vp.shape[0]*24}', )
        
        return (img_list.cpu().numpy(), vp_list.cpu().numpy(), preds, rot_preds)
        


def add_pseudo_labels(self, predict_proba=0.5):
    b_lbl = hu.collate_fn([self.train_set.__getitem__(0, i) 
                        for i in range(24)], mode='default')
    n_samples = 100
    for n_start in range(0, len(self.unlabeled_set), n_samples):
        pred_tuple = predict_vps(self, b_lbl['images_a'], 
                                                n_start=n_start, n_samples=n_samples,
                                                prob=predict_proba)
        for (i, v, p, pr) in zip(*pred_tuple):
            if self.exp_dict.get('augment_pred', False):
                if p == pr:
                    self.pseudo_vps.data[int(i), int(v)] = p
            else:
                self.pseudo_vps.data[int(i), int(v)] = p
    lbl_ind = self.pseudo_vps != -1
    vp_acc = (self.pseudo_vps[lbl_ind] == 
            torch.arange(24).repeat(self.pseudo_vps.shape[0], 1).cuda()[lbl_ind]).float().mean()
    lbl_ratio = lbl_ind.float().mean()
    self.meta = {'pseudo_ratio':float(lbl_ratio), 'pseudo_acc':float(vp_acc)}
    return self.meta 

def compute_siamvp_loss(self, images, viewpoints_ids):
    f = self.vp_siam.forward_one(images) 
    d = self.vp_siam.out(torch.abs(f[:, None] - f[None]))
    m = (torch.abs(viewpoints_ids[:, None] - viewpoints_ids[None]) == 0).float().cuda()
    idx_upper = torch.triu(torch.ones(d.shape[0], d.shape[0]),  diagonal=1) == 1
    # hu.save_image('tmp.png', images[m[6]==1])
    m = m[idx_upper]

    d_upper = torch.sigmoid(d[idx_upper])

    loss = 0.
    # negs
    idx = m == 0
    preds = d_upper[idx]
    neg = (preds < 0.5).float().mean()
    bs = 32

    loss += F.binary_cross_entropy(preds.max(), torch.as_tensor(0.).cuda()) / bs
    loss += F.binary_cross_entropy(preds[np.random.choice(preds.shape[0],bs,replace=True)].squeeze(), 
                                        torch.ones(bs).cuda()*0)
    
    # positives
    idx = m == 1
    preds = d_upper[idx]
    pos = (preds > 0.5).float().mean()
    loss += F.binary_cross_entropy(preds.min(),  torch.as_tensor(1.).cuda()) / bs
    loss += F.binary_cross_entropy(preds[np.random.choice(preds.shape[0],bs,replace=True)].squeeze(),  
                                    torch.ones(bs).cuda())

    acc = (pos + neg)/2.
    return d_upper, m, loss, acc

def compute_loss_list(self, batch, ldict):
        images_a = batch['images_a']
        images_b = batch['images_b']
        viewpoints_a = batch['viewpoints_a']
        viewpoints_b = batch['viewpoints_b']

        name = ldict['name']
        if name == 'base':
            loss = compute_loss(self, images_a, images_b, viewpoints_a, viewpoints_b)
        
        elif name == 'base_vp_cons':
            i_list, _ = torch.where(self.pseudo_vps != -1)
            u_list, c_list = i_list.unique(return_counts=True)
            u_list = u_list[c_list >= 2]
            pbs = self.exp_dict.get('pseudo_batch_size', 32)
            if len(u_list):
                im_ind = np.random.choice(u_list.cpu().numpy(), pbs, replace=True)
                pv = self.pseudo_vps.detach().cpu().numpy()
                flags = (self.pseudo_vps[im_ind]!= -1).detach()
               
                ind_list = []
                vp1_list = []
                vp2_list = []
                vp_poss = torch.arange(24)
                for i in range(len(im_ind)):
                    pv_i = pv[im_ind[i]] 
                    ind_bool = np.where(pv_i != -1)[0]
                    v1_gt, v2_gt = np.random.choice(ind_bool, 2, replace=False)
                    assert int(vp_poss[v1_gt]) == v1_gt
                    assert int(vp_poss[v2_gt])  == v2_gt
                    v1_pred, v2_pred = int(pv_i[v1_gt]), int(pv_i[v2_gt])
                    assert v1_pred != -1 and v2_pred != -1
                    ind_list += [(im_ind[i], v1_gt, v2_gt)]
                    vp1_list += [self.train_set.possible_viewpoints[int(v1_pred)]]
                    vp2_list += [self.train_set.possible_viewpoints[int(v2_pred)]]

                b_ubl = hu.collate_fn([self.unlabeled_set.__getitem__(im, int(v1), int(v2)) 
                        for (im, v1, v2) in ind_list], mode='default')
                
                lbs = 64 - pbs
                loss = compute_loss(self, torch.cat([images_a[:lbs], b_ubl['images_a'].to('cuda')], dim=0), 
                                          torch.cat([images_b[:lbs], b_ubl['images_b'].to('cuda')], dim=0), 
                                          torch.cat([viewpoints_a[:lbs], torch.stack(vp1_list, dim=0).to('cuda')], dim=0), 
                                          torch.cat([viewpoints_b[:lbs], torch.stack(vp2_list, dim=0).to('cuda')], dim=0))
                """
                if 1:
                    acc_all = float((self.pseudo_vps[im_ind][flags] == 
                              torch.arange(24).repeat(flags.shape[0], 1).cuda()[flags]).float().mean())
                    acc_batch = float((torch.norm(torch.stack(vp1_list, dim=0) - b_ubl['viewpoints_a'], dim=1)<1e-5).float().mean())
                    print(f'acc all: {acc_all}, acc batch: {acc_batch}', )
                """
            else:
                loss = compute_loss(self, images_a, images_b, viewpoints_a, viewpoints_b)

        elif name == 'vp_cons':
            

            ###
            # b,v,c,h,w = self.unlabeled_set.images.shape
            
            d, m, loss, acc = compute_siamvp_loss(self, images_a[:,:3], batch['viewpoint_id_a'])
            self.vp_siam.opt.zero_grad()
            loss.backward()
            self.vp_siam.opt.step()
            
            if self.exp_dict.get('augment_loss', False):
                rotated = rotate(images_a[:,:3], torch.as_tensor(np.random.randint(360)).float().cuda(), 
                                    align_corners=True)
                d, m, loss_rot, acc_rot = compute_siamvp_loss(self, rotated, batch['viewpoint_id_a'])
                self.vp_siam.opt.zero_grad()
                loss_rot.backward()
                self.vp_siam.opt.step()

            # self.meta['vp_acc'] = acc
            loss = 0.

            # Get new images
            if acc > 0.99 and self.iteration % self.exp_dict.get('pseudo_cycle', 400) == 0 and self.iteration != 0:
                add_pseudo_labels(self, predict_proba=0.8)

        else:
            raise ValueError(f'{name} existe pas')

        return loss

def single_loss(self, images, vps, iou_only=False):
    # get mesh features from images
    vertices, faces = self.reconstruct(images)


    # [Raa, Rba, Rab, Rbb], cross render multiview images
    self.renderer.transform.set_eyes(vps)
    silhouettes = self.renderer(vertices, faces)

    # compute losses
    loss = self.lambda_laplacian * self.laplacian_loss(vertices).mean()
    loss += self.lambda_flatten *self.flatten_loss(vertices).mean()
    
    loss_iou = iou_loss(silhouettes[:, 3], images[:, 3], reduction='mean')
    if iou_only:
        return loss_iou
    loss += loss_iou
    return loss 

def get_triplets(embeddings, y):
  from itertools import combinations
  margin = 1
  D = pdist(embeddings)
  D = D.cpu()

  y = y.cpu().data.numpy().ravel()
  trip = []

  for label in set(y):
      label_mask = (y == label)
      label_indices = np.where(label_mask)[0]
      if len(label_indices) < 2:
          continue
      neg_ind = np.where(np.logical_not(label_mask))[0]
      
      ap = list(combinations(label_indices, 2))  # All anchor-positive pairs
      ap = np.array(ap)

      ap_D = D[ap[:, 0], ap[:, 1]]
      
      # # GET HARD NEGATIVE
      # if np.random.rand() < 0.5:
      #   trip += get_neg_hard(neg_ind, hardest_negative,
      #                D, ap, ap_D, margin)
      # else:
      trip += get_neg_hard(neg_ind, random_neg,
                 D, ap, ap_D, margin)

  if len(trip) == 0:
      ap = ap[0]
      trip.append([ap[0], ap[1], neg_ind[0]])

  trip = np.array(trip)

  return torch.LongTensor(trip)

def pdist(vectors):
    D = -2 * vectors.mm(torch.t(vectors)) 
    D += vectors.pow(2).sum(dim=1).view(1, -1) 
    D += vectors.pow(2).sum(dim=1).view(-1, 1)

    return D

def random_neg(loss_values):
    neg_hards = np.where(loss_values > 0)[0]
    return np.random.choice(neg_hards) if len(neg_hards) > 0 else None
    
def get_neg_hard(neg_ind, 
                      select_func,
                      D, ap, ap_D, margin):
    trip = []

    for ap_i, ap_di in zip(ap, ap_D):
        loss_values = (ap_di - 
               D[torch.LongTensor(np.array([ap_i[0]])), 
                torch.LongTensor(neg_ind)] + margin)

        loss_values = loss_values.data.cpu().numpy()
        neg_hard = select_func(loss_values)

        if neg_hard is not None:
            neg_hard = neg_ind[neg_hard]
            trip.append([ap_i[0], ap_i[1], neg_hard])

    return trip

def compute_loss(self, images_a, images_b, viewpoints_a, viewpoints_b):
    self.train()

    # predict multi view
    images = torch.cat((images_a, images_b), dim=0)
    viewpoints = torch.cat((viewpoints_a, viewpoints_a, viewpoints_b, viewpoints_b), dim=0)
    

    # get mesh features from images
    vertices, faces = self.reconstruct(images)

    # [Ma, Mb, Ma, Mb]
    vertices = torch.cat((vertices, vertices), dim=0)
    faces = torch.cat((faces, faces), dim=0)

    # [Raa, Rba, Rab, Rbb], cross render multiview images
    self.renderer.transform.set_eyes(viewpoints)
    silhouettes = self.renderer(vertices, faces).chunk(4, dim=0)

    # compute losses
    loss = self.lambda_laplacian * self.laplacian_loss(vertices).mean()
    loss += self.lambda_flatten *self.flatten_loss(vertices).mean()
    loss += multiview_iou_loss(silhouettes, images_a, images_b)
    
    return loss


def compute_unlabeled_loss(self, cheating=False):
    images_ubl = self.get_ubl_batch()['images_a'].cuda()
    out_list = self.get_augmentations(images_ubl, scale=self.exp_dict['loss'].get('scale', 1))
    im_a, v_a = out_list[0]['image'], out_list[0]['view']
    im_b, v_b = out_list[1]['image'], out_list[1]['view']
    
    loss = self.compute_loss(self, im_a, im_b, v_a, v_b)
    return loss

def compute_vpt_loss(self, im, vp):
    logits = self.viewpoint_out(self.encoder(im))
    if self.vpt_clf:
        
        tgt = torch.as_tensor(self.neigh.predict(vp.cpu().numpy()))
        return F.cross_entropy(logits, tgt.cuda())
    else:
        return F.mse_loss(logits, vp)

