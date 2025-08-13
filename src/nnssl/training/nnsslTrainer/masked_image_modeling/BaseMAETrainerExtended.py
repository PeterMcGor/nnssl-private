import os
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
from deprecated import deprecated
from typing_extensions import override
from dataclasses import asdict


import torch
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.get_network_from_plan import get_network_from_plans
from nnssl.ssl_data.configure_basic_dummyDA import configure_rotation_dummyDA_mirroring_and_inital_patch_size

from nnssl.training.loss.mse_loss import MAEMSELoss, LossMaskMSELoss
from nnssl.training.loss.latent_loss import BottleNeckContrastiveLoss
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer
from nnssl.training.lr_scheduler.polylr import PolyLRScheduler
from torch import nn
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from torch import autocast
from nnssl.utilities.helpers import dummy_context
from torch.nn.parallel import DistributedDataParallel as DDP
from batchgenerators.utilities.file_and_folder_operations import join
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import save_json

from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
import numpy as np


class BaseMAETrainerExtended(BaseMAETrainer):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (160, 160, 160)
        self.mask_percentage: float = 0.75

        self.im_output_folder = os.path.join(self.output_folder, "img_log")
        os.makedirs(self.im_output_folder, exist_ok=True)
        self.save_imgs_every_n_epochs = 200
        self._feat_handle = None
        self._bottleneck_features = []
        
    def _get_net(self):
        return self.network.module if isinstance(self.network, DDP) else self.network

    def _resolve_last_encoder_block(self):
        net = self._get_net()
        # Prefer an explicit path if your arch has it:
        if hasattr(net, "encoder") and hasattr(net.encoder, "stages"):
            return net.encoder.stages[-1]
        if hasattr(net, "encoder"):
            # Fallback: last child of decoder
            children = list(net.encoder.children())
            if len(children) > 0:
                return children[-1]
        print(net.encoder)
        raise AttributeError("Could not resolve last decoder block for hook registration.")

    def _register_bottleneck_hook(self):
        if self._feat_handle is not None:
            self._feat_handle.remove()
            self._feat_handle = None
        
        def hook(module, input, output):
            self._bottleneck_features.append(output)

        block = self._resolve_last_encoder_block()            
        self._feat_handle = block.register_forward_hook(hook)        

    def get_bottleneck_features(self):
        return self.bottleneck_features
    
    def initialize(self):
        super(BaseMAETrainerExtended, self).initialize()
        
        print("Registering hook for bottleneck features...")
        print(self.network)
        self._register_bottleneck_hook()
        self.update_optimizer_params()  # To use the loss

    def update_optimizer_params(self):
        # Collect trainable loss params not already in optimizer
        self.loss.to(self.device)
        loss_params = [p for p in self.loss.parameters() if p.requires_grad]
        if not loss_params:
            return

        existing = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
        new_params = [p for p in loss_params if id(p) not in existing]
        if new_params:
            self.optimizer.add_param_group({"params": new_params})
            self.print_to_log_file(f"Added {len(new_params)} loss parameters to optimizer.")

        # Recreate scheduler so it knows about all param groups
        self.lr_scheduler = PolyLRScheduler(self.optimizer, self.initial_lr, self.num_epochs)     
    
    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return BottleNeckContrastiveLoss()
    
    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        data = data.to(self.device, non_blocking=True)

        # We use the self.batch_size as it is not identical with the plan batch_size in ddp cases.
        mask = self.mask_creation(self.batch_size, self.config_plan.patch_size, self.mask_percentage).to(
            self.device, non_blocking=True
        )
        # Make the mask the same size as the data
        rep_D, rep_H, rep_W = (
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
            data.shape[4] // mask.shape[4],
        )
        mask = mask.repeat_interleave(rep_D, dim=2).repeat_interleave(rep_H, dim=3).repeat_interleave(rep_W, dim=4)

        masked_data = data * mask

        self.optimizer.zero_grad(set_to_none=True)
        print(self.optimizer)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(masked_data)
            # del data
            latent = self._bottleneck_features
            l = self.loss(batch, output, mask, latent)
            # l = self.loss(output, data, mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        # Important: drop reference so graph can be freed
        self._bottleneck_features = []

        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        data = data.to(self.device, non_blocking=True)

        mask = self.mask_creation(self.batch_size, self.config_plan.patch_size, self.mask_percentage).to(
            self.device, non_blocking=True
        )
        # Make the mask the same size as the data
        rep_D, rep_H, rep_W = (
            data.shape[2] // mask.shape[2],
            data.shape[3] // mask.shape[3],
            data.shape[4] // mask.shape[4],
        )
        mask = mask.repeat_interleave(rep_D, dim=2).repeat_interleave(rep_H, dim=3).repeat_interleave(rep_W, dim=4)

        masked_data = data * mask

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(masked_data)

            latent = self._bottleneck_features
            l = self.loss(batch, output, mask, latent)
            # l = self.loss(output, data, mask)

        # Important: drop reference so graph can be freed
        self._bottleneck_features = []

        return {"loss": l.detach().cpu().numpy()}