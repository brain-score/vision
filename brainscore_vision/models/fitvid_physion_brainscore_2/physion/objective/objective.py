import os
import logging
import re
import numpy as np
import torch
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from torch.utils.data import DataLoader
from physopt.objective import ReadoutObjectiveBase, PhysOptModel

class PytorchModel(PhysOptModel):
    def __init__(self, *args, **kwargs): # TODO: better to not use init for this?
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # must set device first since used in get_model, called in super
        super().__init__(*args, **kwargs)

    def load_model(self, model_file):
        assert os.path.isfile(model_file), f'Cannot find model file: {model_file}'
        self.model.load_state_dict(torch.load(model_file))
        logging.info(f'Loaded existing ckpt from {model_file}')
        return self.model

    def save_model(self, model_file):
        logging.info(f'Saved model checkpoint to: {model_file}')
        torch.save(self.model.state_dict(), model_file)

    def init_seed(self):
        super().init_seed()
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

    def get_dataloader(self, TDWDataset, datapaths, random_seq, shuffle, num_workers=2):
        cfg = self.pretraining_cfg
        dataset = TDWDataset(
            data_root=datapaths,
            imsize=cfg.DATA.IMSIZE,
            seq_len=cfg.DATA.SEQ_LEN,
            state_len=cfg.DATA.STATE_LEN,
            random_seq=random_seq,
            debug=self.cfg.DEBUG,
            subsample_factor=cfg.DATA.SUBSAMPLE_FACTOR,
            seed=self.seed,
            )
        dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=num_workers)
        return dataloader

class PhysionReadoutObjective(ReadoutObjectiveBase):
    def get_readout_model(self):
        steps = [('clf', LogisticRegression(max_iter=self.readout_cfg.MODEL.MAX_ITER))]
        if self.readout_cfg.MODEL.NORM_INPUT:
            steps.insert(0, ('scale', StandardScaler()))
        logging.info(f'Readout model steps: {steps}')
        pipe = Pipeline(steps)

        assert len(self.readout_cfg.MODEL.LOGSPACE) == 3, 'logspace must contain start, stop, and num'
        grid_search_params = {
            'clf__C': np.logspace(*self.readout_cfg.MODEL.LOGSPACE),
            }
        skf = StratifiedKFold(n_splits=self.readout_cfg.MODEL.CV, shuffle=True, random_state=self.seed)
        logging.info(f'CV folds: {skf}')
        readout_model = GridSearchCV(pipe, param_grid=grid_search_params, cv=skf, verbose=3)
        return readout_model

    @staticmethod
    def get_readout_model_info(readout_model):
        results = {}
        results['best_params'] = readout_model.best_params_
        results['cv_results'] = readout_model.cv_results_
        return results
