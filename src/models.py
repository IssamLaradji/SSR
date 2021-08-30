import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time
import pandas as pd
import soft_renderer as sr
import soft_renderer.functional as srf
import math
from haven import haven_utils as hu
from src.losses import multiview_iou_loss, iou_loss
import soft_renderer.functional as srf
from soft_renderer.functional.voxelization import voxelize_sub1, voxelize_sub2, voxelize_sub3
from src.utils import AverageMeter, img_cvt
import imageio, tqdm
import numpy as np 
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
import pylab as plt 
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.io import load_obj, save_obj
from . import losses
import os
import torch
import numpy as np
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)


def get_model(exp_dict, train_set, unlabeled_set):
    model_name = exp_dict.get('model')
    if model_name is None:
        return Model(train_set, unlabeled_set, exp_dict)

class Model(nn.Module):
    def __init__(self, train_set, unlabeled_set, exp_dict):
        super(Model, self).__init__()
        self.train_set = train_set
        self.unlabeled_set = unlabeled_set
        self.exp_dict = exp_dict
        self.init_model()
        self.n_iters_per_epoch = 200
        self.meta = {}

    def init_model(self):
        exp_dict = self.exp_dict
        self.param_list = []
        self.encoder = Encoder(im_size=exp_dict['image_size'], use_dropout=exp_dict.get('use_dropout', False))
        self.decoder = Decoder('src/sphere_642.obj')
        self.renderer = sr.SoftRenderer(image_size=exp_dict['image_size'], sigma_val=exp_dict['sigma_val'], 
                                        aggr_func_rgb='hard', camera_mode='look_at', viewing_angle=15,
                                        dist_eps=1e-10)
        self.laplacian_loss = sr.LaplacianLoss(self.decoder.vertices_base, self.decoder.faces)
        self.flatten_loss = sr.FlattenLoss(self.decoder.faces)
        self.lambda_laplacian = exp_dict['lambda_laplacian']
        self.lambda_flatten = exp_dict['lambda_flatten']
        self.iteration = 0
        self.param_list += list(self.encoder.parameters()) 
        self.param_list += list(self.decoder.parameters())
        vpt_list = [ldict for ldict in self.exp_dict['loss'] if ldict['name']=='vpt']

        self.viewpoint_out = nn.Linear(512, 3)
        self.vp_clf = None
        if exp_dict.get('vp_clf'):
            self.vp_clf = torch.nn.Sequential(Encoder(im_size=exp_dict['image_size'],
                                 use_dropout=exp_dict.get('use_dropout', False)),
                                 torch.nn.Linear(512, 3)).cuda()
            self.param_list += list(self.vp_clf.parameters())
        if len(vpt_list):
            if vpt_list[0].get('clf'):
                self.viewpoint_out = nn.Linear(512, 100)
                self.vpt_clf = True
                p_vp = self.train_set.possible_viewpoints.numpy()
                from sklearn.neighbors import KNeighborsClassifier
                self.neigh = KNeighborsClassifier(n_neighbors=1)
                y = np.arange(p_vp.shape[0])
                self.neigh.fit(p_vp, y)
                assert (self.neigh.predict(p_vp) -  y).sum() == 0
            else:
                self.vpt_clf = False
                self.viewpoint_out = nn.Linear(512, 3)
        self.pseudo_vps = nn.Parameter(torch.ones(len(self.unlabeled_set), 24)*-1)
                
        
        # self.opt = torch.optim.Adam(self.model_param(), exp_dict['learning_rate'])
        if 'vp_cons' in [ldict['name'] for ldict in self.exp_dict['loss']]:
            self.vp_siam = Siamese(exp_dict)
            # self.param_list += list(self.vp_siam.parameters())

        self.cuda()

        self.opt = torch.optim.Adam(self.param_list , exp_dict['learning_rate'])

    def set_dropout_train(self):
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.Dropout2d):
                module.train()

    @torch.no_grad()
    def score_on_batch(self, batch, n_mcmc=10):
        self.eval()
        self.set_dropout_train()

        # put images to cuda
        images = batch['images_a'].cuda()
        input_shape = images.shape
        # forward on n_mcmc batch      
        images_stacked = torch.stack([images] * n_mcmc)
        images_stacked = images_stacked.view(input_shape[0] * n_mcmc, *input_shape[1:])
        logits, faces = self.reconstruct(images_stacked)
            

        logits = logits.view([n_mcmc, input_shape[0], *logits.size()[1:]])
      
        left = logits.mean(dim=0).sum(dim=1)
        right = logits.sum(dim=2).mean(0)
        bald = torch.abs(left - right)
        score_map = float(bald.mean())


        return score_map 

    def train_on_batch(self, batch):
        self.train()
        loss_dict = {}
        # adjust learning rate and sigma_val (decay after 150k iter)
        lr = adjust_learning_rate([self.opt], self.exp_dict['learning_rate'], self.iteration, method=self.exp_dict['lr_type'])
        self.set_sigma(adjust_sigma(self.exp_dict['sigma_val'], self.iteration))

        batch['images_a'] = batch['images_a'].cuda()
        batch['images_b'] = batch['images_b'].cuda()
        batch['viewpoints_a'] = batch['viewpoints_a'].cuda()
        batch['viewpoints_b'] = batch['viewpoints_b'].cuda()

        # Main loss
        self.opt.zero_grad()

        loss = 0.
        n_terms =  len(self.exp_dict['loss'])
        for ldict in  self.exp_dict['loss']:
            loss += losses.compute_loss_list(self, batch, ldict) / n_terms
        
        
        # compute gradient and optimize
        
        loss.backward()
        self.opt.step()

        # vis
        # if self.args.debug:
        #     if 1:
        #         out_list = self.get_augmentations(images_a, scale=self.exp_dict['loss'].get('scale', 1))
        #         im1, im2 = out_list[0]['image'], out_list[1]['image']
        #         a1 = np.array(im1[:1,:3].detach().cpu())
        #         a2 = np.array(im2[:1,:3].detach().cpu())
        #         i1 = np.vstack([a1, a2])
        #         hu.save_image('.tmp/tmp.png', np.vstack([i1]), nrow=2)
        self.iteration  += 1
        loss_dict['train_loss'] = float(loss)
        return loss_dict

    

    def predict_viewpoint(self, im):
        logits = self.viewpoint_out(self.encoder(im))
        if self.vpt_clf:
            return self.train_set.possible_viewpoints[logits.argmax(dim=1)]
            
        return logits


    def get_lbl_batch(self):
        ind_list = np.random.choice(len(self.train_set), self.exp_dict['batch_size'])
        b_lbl = hu.collate_fn([self.train_set[i] for i in ind_list], mode='default')
        return b_lbl

    def get_ubl_batch(self, ind_list=None):
        if ind_list is None:
            ind_list = np.random.choice(len(self.unlabeled_set), self.exp_dict['batch_size'])

        b_ulb = hu.collate_fn([self.unlabeled_set[i] for i in ind_list], mode='default')
        return b_ulb

    def get_out(self, imgs, views):
        vertices, faces = self.reconstruct(imgs)
        #  meshes = Meshes(verts=[vertices], faces=[faces])
        # save_obj('tmp.obj', vertices[0], faces[0])
        self.renderer.transform.set_eyes(views)
        out = self.renderer(vertices, faces)

        return out, vertices, faces
    @torch.no_grad()
    def get_augs(self, batch, scale=1):  
        images_a = batch['images_a'].cuda()
        images_b = batch['images_b'].cuda()
        out_list = self.get_augmentations([images_a, images_b], scale=0)
                
        batch_new = {} 
        batch_new['images_a'] = out_list[0]['image']
        batch_new['images_b'] = out_list[1]['image']

        batch_new['viewpoints_a'] = out_list[0]['view']
        batch_new['viewpoints_b'] = out_list[1]['view']

        return batch_new

    def get_augmentations(self, images_list, scale=1):  
        v = self.train_set.possible_viewpoints.cuda()
        out_list = []
        vert_list = []
        for images in images_list:
            ind = torch.randperm(v_new.shape[0])[:max(1, v.shape[0]//2)]

            if scale > 0:
                v_new[ind] += scale*torch.normal(0, torch.as_tensor([.001, 0.,.001]).repeat(ind.shape[0],1).to(v_new.device))
            
            out, vert, _ = self.get_out(images, v_new)
            out_list += [{'image':out, 'vertices':vert, 'view':v_new}]

        return out_list

    

    def predict_on_batch(self, batch):
        # soft render images
        
        if 1:
            images_a = batch['images_a'].cuda()
            images_b = batch['images_b'].cuda()
            viewpoints_a = batch['viewpoints_a'].cuda()
            viewpoints_b = batch['viewpoints_b'].cuda()
            
            v1, v2 = viewpoints_a[[0]], viewpoints_b[[0]]
            gt1, gt2 = images_a[[0]], images_b[[0]]
            # predict multi view
            vertices, faces = self.reconstruct(gt1)
            # [Raa, Rba, Rab, Rbb], cross render multiview images
            self.renderer.transform.set_eyes(v1)
            out1 = self.renderer(vertices, faces)

            # [Raa, Rba, Rab, Rbb], cross render multiview images
            self.renderer.transform.set_eyes(v2)
            out2 = self.renderer(vertices, faces)

            vertices, faces = self.reconstruct(out2)
            self.renderer.transform.set_eyes(v1)
            out_hat = self.renderer(vertices, faces)

            print(losses.iou_loss(out1, out_hat))

        vertices, faces = self.reconstruct(silhouettes[0])
        viewpoints = torch.cat((viewpoints_a,), dim=0)
        self.renderer.transform.set_eyes(viewpoints)
        silhouettes2 = self.renderer(vertices, faces)
 

        return silhouettes
    
    @torch.no_grad()
    def vis_on_batch(self, batch, savedir=None):
        self.eval() 
       
        images_a = batch['images_a'].cuda()
        images_b = batch['images_b'].cuda()
        viewpoints_a = batch['viewpoints_a'].cuda()
        viewpoints_b = batch['viewpoints_b'].cuda()

        render_images, laplacian_loss, flatten_loss = self.predict_multiview(images_a, images_b, 
                                                            viewpoints_a, viewpoints_b) 

        demo_image = images_a[0:1].cuda()
        demo_path = os.path.join(savedir, 'images', f'demo.obj')
        demo_v, demo_f = self.reconstruct(demo_image)
        srf.save_obj(demo_path, demo_v[0], demo_f[0])
        i1 = img_cvt(render_images[0][0])
        i2 = img_cvt(images_a[0])
        im = np.hstack([i2, i1])
        if savedir is None:
            return im
        imageio.imsave(os.path.join(savedir, 'images', f'gt_fake.png'), im)

    def train_on_dataset(self, dataset, num_workers=0):
        self.train()
        bs = self.exp_dict['batch_size']
        n_val = self.exp_dict.get('n_val')
        if n_val is None:
            n_val = 50
        rand_sampler = torch.utils.data.RandomSampler(dataset,  num_samples=self.n_iters_per_epoch*bs, replacement=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=bs, sampler=rand_sampler, num_workers=num_workers)

        train_list = []
        pbar = tqdm.tqdm(loader, desc='Training')
        for b in pbar:
            train_dict = self.train_on_batch(b)
            train_list += [train_dict]
            pbar.set_description(f'Training - Loss: {train_dict["train_loss"]:.3f}')
        train_dict = pd.DataFrame(train_list).mean().to_dict()

        return train_dict

    @torch.no_grad()
    def val_on_dataset(self, dataset, savedir_images):
        self.eval()
        device = 'cuda'
        iou_all = []
        s_time = time.time()
        classes_visited = set()
        for class_id, class_name in tqdm.tqdm(dataset.class_ids_pair, desc='Validating'):
            iou = 0
            n_count = 0
            
            class_items = dataset.get_all_batches_for_evaluation(self.exp_dict['batch_size'], class_id)
            n_val = self.exp_dict.get('n_val')
            for i, (im, vx) in enumerate(class_items):
                if n_val is not None and i > n_val:
                    break
                images = torch.autograd.Variable(im).cuda()
                voxels = vx.numpy()
      
                batch_iou, vertices, faces = self.evaluate_iou(images, voxels=voxels)
                iou += batch_iou.sum()
                n_count += batch_iou.shape[0]
                
                if class_id not in classes_visited:
                    k = 0
                    # Create a Meshes object
                    mesh = Meshes(
                        verts=[vertices[k].to(device)],   
                        faces=[faces[k].to(device)]
                    )
                    # save_obj('tmp.obj', )
                    # save_obj('tmp.obj', vertices[k], faces[k])
                    i1 = torch.as_tensor(get_pointcloud(mesh, title='finally').copy()).permute(2,0,1)[None]
                    i2 = F.interpolate(torch.as_tensor(img_cvt(images[k])).permute(2,0,1)[None, :3], i1.shape[-2:])
                    hu.save_image(f'{savedir_images}/{class_id}.png', torch.cat([i2,i1])/255.)
                    classes_visited.add(class_id)

            iou_cls = 100 * (iou / n_count)
            iou_all.append(iou_cls)
            
        iou_avg = sum(iou_all) / len(iou_all)
        return {'iou': iou_avg, 'time':time.time() - s_time}

    def get_state_dict(self):
        return {'model': self.state_dict(),
                'optimizer': self.opt.state_dict(),
                }

    def set_state_dict(self, state_dict):
        self.load_state_dict(state_dict['model'], strict=False)
        try:
            self.opt.load_state_dict(state_dict['optimizer'])
        except:
            pass
       



    def set_sigma(self, sigma):
        self.renderer.set_sigma(sigma)

    def reconstruct(self, images):
        vertices, faces = self.decoder(self.encoder(images))
        return vertices, faces


    def evaluate_iou(self, images, voxels):
        vertices, faces = self.reconstruct(images)

        faces_ = srf.face_vertices(vertices, faces).data
        faces_norm = faces_ * 1. * (32. - 1) / 32. + 0.5
        voxels_predict = voxelization(faces_norm, 32, False).cpu().numpy()
        voxels_predict = voxels_predict.transpose(0, 2, 1, 3)[:, :, :, ::-1]
        iou = (voxels * voxels_predict).sum((1, 2, 3)) / (0 < (voxels + voxels_predict)).sum((1, 2, 3))
        return iou, vertices, faces


class Siamese(nn.Module):

    def __init__(self, exp_dict):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    # 128@42*42
            nn.MaxPool2d(2),   # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(), # 128@18*18
            # nn.MaxPool2d(2), # 128@9*9
            # nn.Conv2d(128, 256, 4),
            # nn.ReLU(),   # 256@6*6
        )
        self.liner = nn.Sequential(nn.Linear(6272, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

        self.opt = torch.optim.Adam(self.parameters(), 1e-4)

    def forward_one(self, x):
        x = self.conv(x[:,:3])
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out

class Siamese2(nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
        self.vp_encoder = Encoder(im_size=exp_dict['image_size'], 
                                      use_dropout=exp_dict.get('use_dropout', False))
        self.out = nn.Linear(512, 1)
        self.liner = nn.Sequential(nn.Linear(512, 512), nn.Sigmoid())
        self.opt = torch.optim.Adam(self.parameters(), 1e-4)
    def forward_one(self, x):
        x = self.vp_encoder(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out

class Encoder(nn.Module):
    def __init__(self, dim_in=4, dim_out=512, dim1=64, dim2=1024, im_size=64, use_dropout=False):
        super(Encoder, self).__init__()
        dim_hidden = [dim1, dim1*2, dim1*4, dim2, dim2]

        self.conv1 = nn.Conv2d(dim_in, dim_hidden[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(dim_hidden[0], dim_hidden[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(dim_hidden[1], dim_hidden[2], kernel_size=5, stride=2, padding=2)

        # self.bn1 = nn.BatchNorm2d(dim_hidden[0])
        # self.bn2 = nn.BatchNorm2d(dim_hidden[1])
        # self.bn3 = nn.BatchNorm2d(dim_hidden[2])

        self.fc1 = nn.Linear(dim_hidden[2]*math.ceil(im_size/8)**2, dim_hidden[3])
        self.fc2 = nn.Linear(dim_hidden[3], dim_hidden[4])
        self.fc3 = nn.Linear(dim_hidden[4], dim_out)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        # x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc3(x), inplace=True)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class Decoder(nn.Module):
    def __init__(self, filename_obj, dim_in=512, centroid_scale=0.1, bias_scale=1.0, centroid_lr=0.1, bias_lr=1.0):
        super(Decoder, self).__init__()
        # load .obj
        self.template_mesh = sr.Mesh.from_obj(filename_obj)
        # vertices_base, faces = srf.load_obj(filename_obj)
        self.register_buffer('vertices_base', self.template_mesh.vertices.cpu()[0])#vertices_base)
        self.register_buffer('faces', self.template_mesh.faces.cpu()[0])#faces)

        self.nv = self.vertices_base.size(0)
        self.nf = self.faces.size(0)
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 0.5

        dim = 1024
        dim_hidden = [dim, dim*2]
        self.fc1 = nn.Linear(dim_in, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc_centroid = nn.Linear(dim_hidden[1], 3)
        self.fc_bias = nn.Linear(dim_hidden[1], self.nv*3)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        # decoder follows NMR
        centroid = self.fc_centroid(x) * self.centroid_scale

        bias = self.fc_bias(x) * self.bias_scale
        bias = bias.view(-1, self.nv, 3)

        base = self.vertices_base * self.obj_scale

        sign = torch.sign(base)
        base = torch.abs(base)
        base = torch.log(base / (1 - base))

        centroid = torch.tanh(centroid[:, None, :])
        scale_pos = 1 - centroid
        scale_neg = centroid + 1

        vertices = torch.sigmoid(base + bias) * sign
        vertices = F.relu(vertices) * scale_pos - F.relu(-vertices) * scale_neg
        vertices = vertices + centroid
        vertices = vertices * 0.5
        faces = self.faces[None, :, :].repeat(batch_size, 1, 1)

        return vertices, faces


def encode_labels(vals):
    v_list = []
    for v in vals:
        v_list += [hu.hash_dict({'id': v})]

    return v_list


def get_pointcloud(mesh, title):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close('all')
    return image_from_plot

def voxelization(faces, size, normalize=False):
    faces = faces.clone()
    if normalize:
        pass
    else:
        faces *= size

    voxels0 = voxelize_sub1(faces, size, 0)
    voxels1 = voxelize_sub1(faces, size, 1)
    voxels2 = voxelize_sub1(faces, size, 2)
    voxels3 = voxelize_sub2(faces, size)

    voxels = voxels0 + voxels1 + voxels2 + voxels3
    voxels = (voxels > 0).int()
    voxels = voxelize_sub3(faces, voxels.contiguous())

    return voxels

def adjust_learning_rate(optimizers, learning_rate, i, method):
    if method == 'step':
        lr, decay = learning_rate, 0.3
        if i >= 150000:
            lr *= decay
    elif method == 'constant':
        lr = learning_rate
    else:
        print("no such learing rate type")

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def adjust_sigma(sigma, i):
    decay = 0.3
    if i >= 150000:
        sigma *= decay
    return sigma


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims=512, hidden_dims=512, output_dims=1):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, output_dims))

        

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


def update_encoder(model, opt, disc, opt_disc, images_lbl, images_ubl):
    criterion = nn.CrossEntropyLoss()

    # zero gradients for opt
    opt_disc.zero_grad()

    # extract and concat features
    feat_src = model.encoder(images_lbl)
    feat_tgt = model.encoder(images_ubl)
    feat_concat = torch.cat((feat_src, feat_tgt), 0)

    # predict on discriminator
    pred_concat = disc(feat_concat.detach())

    # prepare real and fake label
    label_src = torch.ones(feat_src.size(0)).long()
    label_tgt = torch.zeros(feat_tgt.size(0)).long()
    label_concat = torch.cat((label_src, label_tgt), 0).cuda()

    # compute loss for disc
    loss_disc = criterion(pred_concat, label_concat)
    loss_disc.backward()

    # optimize disc
    opt_disc.step()

    pred_cls = torch.squeeze(pred_concat.max(1)[1])
    acc = (pred_cls == label_concat).float().mean()

    ############################
    # 2.2 train target encoder #
    ############################

    # zero gradients for opt
    opt_disc.zero_grad()
    opt.zero_grad()

    # extract and target features
    feat_tgt = model.encoder(images_ubl)

    # predict on discriminator
    pred_tgt = disc(feat_tgt)

    # prepare fake labels
    label_tgt = torch.ones(feat_tgt.size(0)).long().cuda()

    # compute loss for target encoder
    loss_tgt = criterion(pred_tgt, label_tgt)
    loss_tgt.backward()

    # optimize target encoder
    opt.step()