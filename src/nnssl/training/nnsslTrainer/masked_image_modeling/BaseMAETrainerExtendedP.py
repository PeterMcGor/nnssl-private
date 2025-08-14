import numpy as np
from typing import Union, List, Tuple
from loguru import logger

from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer
from nnssl.data.dataloading.dataset import nnSSLDatasetBlosc2
from nnssl.data.raw_dataset import Collection, IndependentImage
from nnssl.ssl_data.dataloading.data_loader_3d import nnsslDataLoader3D

class nnSSLDatasetBlosc2ExtendInfo(nnSSLDatasetBlosc2):
    """
    This dataset is used to load data that has been saved with the nnSSLDataLoaderBase.
    It extends the nnSSLDatasetBlosc2 with additional information that is needed for the dataloader.
    """
    @staticmethod
    def load_case(dataset_dir: str, image_dataset: dict[str, IndependentImage], image_identifier: str):
        img = image_dataset[image_identifier]
        data, anon, anat, properties = nnSSLDatasetBlosc2.load_case(dataset_dir, image_dataset, image_identifier)
        return data, anon, anat, {**properties, **{"extra_info": img.to_dict()}}


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

