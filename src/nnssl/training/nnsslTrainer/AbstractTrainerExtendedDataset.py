from loguru import logger

from nnssl.data.dataloading.dataset import nnSSLDatasetBlosc2
from nnssl.data.raw_dataset import Collection, IndependentImage
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer


class nnSSLDatasetBlosc2ExtendInfo(nnSSLDatasetBlosc2):
    """
    This dataset is used to load data that has been saved with the nnSSLDataLoaderBase.
    It extends the nnSSLDatasetBlosc2 with additional information that is needed for the dataloader.
    """
    @staticmethod
    def load_case(dataset_dir: str, image_dataset: dict[str, IndependentImage], image_identifier: str):
        img = image_dataset[image_identifier]
        data, anon, anat, properties = super.load_case(dataset_dir, image_dataset, image_identifier)
        subject_info = {'subject_info': img.subject_info},
        return data, anon, anat, {**properties, **subject_info}


class AbstractBaseTrainerExtendedDataset(AbstractBaseTrainer):

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