import os
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
from deprecated import deprecated
from typing_extensions import override
from dataclasses import asdict

import numpy as np
from loguru import logger
import torch
from torch import distributed as dist
from nnssl.utilities.collate_outputs import collate_outputs
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.architectures.get_network_from_plan import get_network_from_plans
from nnssl.ssl_data.configure_basic_dummyDA import configure_rotation_dummyDA_mirroring_and_inital_patch_size

from nnssl.training.loss.mse_loss import MAEMSELoss, LossMaskMSELoss
from nnssl.training.loss.latent_loss import BottleNeckContrastiveLoss, ReconstructionAndSimilarityLoss, SubjectImageSimilarityLoss
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer
from nnssl.training.lr_scheduler.polylr import PolyLRScheduler

from torch import autocast
from nnssl.utilities.helpers import dummy_context
from torch.nn.parallel import DistributedDataParallel as DDP
from batchgenerators.utilities.file_and_folder_operations import join
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import save_json

from nnssl.experiment_planning.experiment_planners.plan import Plan

from nnssl.data.dataloading.dataset import nnSSLDatasetBlosc2
from nnssl.data.raw_dataset import Collection, IndependentImage
from nnssl.ssl_data.dataloading.data_loader_3d import nnsslDataLoader3D


class nnSSLDatasetBlosc2ExtendInfo(nnSSLDatasetBlosc2):
    """
    This dataset is used to load data that has been saved with the nnSSLDataLoaderBase.
    It extends the nnSSLDatasetBlosc2 with additional information that is needed for the dataloader.
    """
    
    @staticmethod
    def find_preferred_modality(img, preference_modality_list=None):
        """
        Find the first available modality from the preference list (case-insensitive).
        If none found, return the first available modality.
        
        Args:
            img: IndependentImage object with subject_info containing volumes
            preference_modality_list: List of preferred modalities (e.g., ['T1w', 'T2w', 'MPRAGE'])
        
        Returns:
            tuple: (session_key, modality_key, volumes_dict) or (None, None, None) if no volumes found
        """
        if preference_modality_list is None:
            preference_modality_list = ['T1w', 'T2w', 'MPRAGE', 'MP2RAGE', 'FLAIR', 'inplaneT2']
        
        volumes = img.subject_info.get('volumes', {})
        if not volumes:
            return None, None, None
        
        # Convert preference list to lowercase for case-insensitive comparison
        preference_modality_lower = [mod.lower() for mod in preference_modality_list]
        
        # First pass: look for preferred modalities (case-insensitive)
        for session_key, session_data in volumes.items():
            # Create a mapping of lowercase modality names to actual keys
            modality_mapping = {mod_key.lower(): mod_key for mod_key in session_data.keys()}
            
            for preferred_modality_lower in preference_modality_lower:
                if preferred_modality_lower in modality_mapping:
                    actual_modality_key = modality_mapping[preferred_modality_lower]
                    return session_key, actual_modality_key, session_data[actual_modality_key]
        
        # Second pass: if no preferred modality found, take the first available
        for session_key, session_data in volumes.items():
            if session_data:  # Check if session has any modalities
                first_modality = next(iter(session_data.keys()))
                return session_key, first_modality, session_data[first_modality]
        
        return None, None, None  # No volumes found

    @staticmethod
    def load_case(dataset_dir: str, image_dataset: dict[str, IndependentImage], image_identifier: str):
        img = image_dataset[image_identifier]
        if not 'volumes' in img.image_info.keys():
            raise RuntimeError(f"Skipping case {image_identifier} - Without  volumes in image_info, cannot load data.")
        session_key, modality_key, volumes_dict = nnSSLDatasetBlosc2ExtendInfo.find_preferred_modality(img)
        if volumes_dict is None:
            raise RuntimeError(f"Skipping case {image_identifier} - No preferred modality found in volumes.")
        if float(volumes_dict['total intracranial']) < 1000:
            raise RuntimeError(f"Skipping case {image_identifier} - Total intracranial volume is too small: {volumes_dict['total intracranial']}.")
        data, anon, anat, properties = nnSSLDatasetBlosc2.load_case(dataset_dir, image_dataset, image_identifier)
        return data, anon, anat, {**properties, **{"extra_info": img.to_dict(), 'subject_features':volumes_dict, 'subject_ids':img.image_path}}


class nnsslDataLoader3DSameSubject(nnsslDataLoader3D):
    def __init__(self, *args, **kwargs):
        """
        Initialize the dataloader and build subject-to-indices mapping for efficiency.
        
        WARNING: This dataloader ensures all samples in each batch come from the same subject.
        This constrains training in the following ways:
        - Reduces within-batch diversity (all samples from same subject)
        - May bias toward subjects with more data samples
        - Subjects with fewer samples than batch_size will be underrepresented
        - May require adjustment of learning rate or other hyperparameters
        """
        super().__init__(*args, **kwargs)
        self._subject_to_indices = None
        self._subjects_with_enough_samples = None
        
        # Build subject mapping once during initialization
        self._build_subject_mapping()
    
    def _build_subject_mapping(self):
        """
        Build mapping from subject prefixes to their indices for efficient lookup.
        Called once during initialization.
        """
        self._subject_to_indices = {}
        
        for idx in self.indices:
            subject_prefix = self._extract_subject_prefix(idx)
            if subject_prefix not in self._subject_to_indices:
                self._subject_to_indices[subject_prefix] = []
            self._subject_to_indices[subject_prefix].append(idx)
        
        # Pre-compute subjects that have enough samples for a full batch
        self._subjects_with_enough_samples = [
            subject for subject, indices in self._subject_to_indices.items() 
            if len(indices) >= self.batch_size
        ]
        
        if not self._subjects_with_enough_samples:
            print(f"Warning: No subjects have {self.batch_size} or more samples. "
                  f"All batches will use sampling with replacement. "
                  f"Consider reducing batch_size for better subject diversity.")

    def get_indices(self):
        """
        This method is overridden to ensure that all indices in a batch come from the same subject.
        This is important for training and validation to ensure subject-level consistency.
        
        WARNING: This sampling strategy may create training bias - see class docstring.
        """
        
        # if self.infinite, this is easy
        if self.infinite:
            return self._get_subject_constrained_batch()

        if self.last_reached:
            self.reset()
            raise StopIteration

        if not self.was_initialized:
            self.reset()

        indices = []

        for b in range(self.batch_size):
            if self.current_position < len(self.indices):
                indices.append(self.indices[self.current_position])
                self.current_position += 1
            else:
                self.last_reached = True
                break

        if len(indices) > 0 and ((not self.last_reached) or self.return_incomplete):
            self.current_position += (self.number_of_threads_in_multithreaded - 1) * self.batch_size
            return indices
        else:
            self.reset()
            raise StopIteration

    def _get_subject_constrained_batch(self):
        """
        Sample a batch where all samples come from the same subject.
        
        WARNING: This method may create training bias. See class docstring.
        """
        
        max_attempts = 10  # Prevent infinite loops
        attempts = 0
        
        while attempts < max_attempts:
            # First try subjects that have enough samples
            if self._subjects_with_enough_samples:
                # Sample a subject with enough data (uniform probability across subjects)
                selected_subject = np.random.choice(self._subjects_with_enough_samples)
                subject_indices = self._subject_to_indices[selected_subject]
            else:
                # Fallback: select any subject randomly
                if self.sampling_probabilities is not None:
                    random_sample = np.random.choice(self.indices, 1, replace=True, p=self.sampling_probabilities)[0]
                else:
                    random_sample = np.random.choice(self.indices, 1, replace=True)[0]
                
                subject_prefix = self._extract_subject_prefix(random_sample)
                subject_indices = self._subject_to_indices[subject_prefix]
            
            if len(subject_indices) >= self.batch_size:
                # We have enough samples from this subject
                return self._sample_from_subject_indices(subject_indices)
            else:
                # Not enough samples from this subject, try another
                attempts += 1
                if attempts == max_attempts:
                    print(f"Warning: Could not find subject with {self.batch_size} samples after "
                          f"{max_attempts} attempts. Using subject with {len(subject_indices)} samples "
                          f"with replacement.")
                    return self._sample_from_subject_indices(subject_indices)
        
        # This should never be reached, but just in case
        if self.sampling_probabilities is not None:
            return np.random.choice(self.indices, self.batch_size, replace=True, p=self.sampling_probabilities)
        else:
            return np.random.choice(self.indices, self.batch_size, replace=True)

    def _sample_from_subject_indices(self, subject_indices):
        """
        Sample batch_size indices from the given subject indices, respecting original probabilities.
        
        Args:
            subject_indices (list): List of indices belonging to a specific subject
            
        Returns:
            np.ndarray: Sampled indices
        """
        if self.sampling_probabilities is not None:
            # Create probabilities for this subject's indices only
            subject_positions = [self.indices.index(idx) for idx in subject_indices]
            subject_probs = self.sampling_probabilities[subject_positions]
            subject_probs = subject_probs / subject_probs.sum()  # Normalize
            
            return np.random.choice(subject_indices, self.batch_size, 
                                  replace=False, p=subject_probs)
        else:
            # Uniform sampling when no probabilities are provided
            return np.random.choice(subject_indices, self.batch_size, replace=False)

    def _extract_subject_prefix(self, filename):
        """
        Extract subject identifier from filename.
        
        Args:
            filename (str): Full filename like 'Dataset001_OpenMind__ds004146__sub-0157__ses-02__...'
            
        Returns:
            str: Subject prefix like 'Dataset001_OpenMind__ds004146__sub-0157'
        """
        # Split by '__' and take up to subject identifier
        parts = filename.split('__')
        if len(parts) >= 3:
            # Typically: Dataset001_OpenMind__ds004146__sub-0157__...
            return '__'.join(parts[:3])
        else:
            # Fallback: return the whole filename if pattern doesn't match
            return filename


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
        self.prev_loss = None
        self.prev_val_loss = None
        
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

    def get_tr_and_val_datasets(self):
        # create dataset split (We only have 'all' as splits anyway!)
        tr_subjects, val_subjects = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        collection = Collection.from_dict(self.pretrain_json)
        dataset_tr = nnSSLDatasetBlosc2ExtendInfo(self.preprocessed_dataset_folder, collection, tr_subjects, self.iimg_filters)
        dataset_val = nnSSLDatasetBlosc2ExtendInfo(
            self.preprocessed_dataset_folder, collection, val_subjects, self.iimg_filters
        )

        logger.info(f"Train dataset contains {len(dataset_tr.image_dataset)} images.")
        logger.info(f"Validation dataset contains {len(dataset_val.image_dataset)} images.")


        return dataset_tr, dataset_val
           
    
    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return ReconstructionAndSimilarityLoss(bottleneck_dim=(320,5,5,5), subject_dim=101)#BottleNeckContrastiveLoss()
    
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
        #print(self.optimizer)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(masked_data)
            # del data
            latent = self._bottleneck_features
            l = self.loss(batch, output, data, mask, latent)
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
            #l = self.loss(batch, output, mask, latent)
            l = self.loss(batch, output, data, mask, latent)
            # l = self.loss(output, data, mask)

        # Important: drop reference so graph can be freed
        self._bottleneck_features = []

        return {"loss": l.detach().cpu().numpy()}
    
    def on_train_epoch_end(self, train_outputs: List[dict]):
        self.interrupt_at_nans(train_outputs)
        outputs = collate_outputs(train_outputs)
        
        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs["loss"])
            loss_here = np.nanmean(np.vstack(losses_tr))  # Changed to nanmean
        else:
            loss_here = np.nanmean(outputs["loss"])  # Changed to nanmean
        
        self.logger.log("train_losses", loss_here, self.current_epoch)

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        
        if self.is_ddp:
            world_size = dist.get_world_size()
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated["loss"])
            loss_here = np.nanmean(np.vstack(losses_val))  # Changed to nanmean
        else:
            loss_here = np.nanmean(outputs_collated["loss"])  # Changed to nanmean
        
        self.logger.log("val_losses", loss_here, self.current_epoch)


class BaseMAETrainerExtendedSingleSubject(BaseMAETrainerExtended):

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...]):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnsslDataLoader3DSameSubject(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
        )
        dl_val = nnsslDataLoader3DSameSubject(
            dataset_val,
            self.batch_size,
            self.config_plan.patch_size,
            self.config_plan.patch_size,
            sampling_probabilities=None,
            pad_sides=None,
        )
        return dl_tr, dl_val




class BaseMAETrainerExtendedTest(BaseMAETrainerExtended):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 1
        self.num_iterations_per_epoch = 10
        self.num_val_iterations_per_epoch = 5
        self.num_epochs = 2
