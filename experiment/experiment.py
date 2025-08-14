import logging
from time import time

import numpy as np
import pandas as pd
from copy import deepcopy 

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from omegaconf import DictConfig, OmegaConf
import torch
import os
import torch.nn as nn

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import dataset.customdataset as customdataset
from dataset import CustomDataset
from hydra.utils import to_absolute_path
from .utils import set_seed, cal_kappa_score
from .optimizers import get_optimizer_grouped_parameters
from sklearn.metrics import cohen_kappa_score
from torchinfo import summary
import tqdm
from .optuna import OptimParam , xgboost_config,lightgbm_config,catboost_config

from model import get_classifier, get_tree_classifier

from sklearn.model_selection import StratifiedKFold, train_test_split

from torch.cuda.amp import autocast,GradScaler

scaler = GradScaler()

logger = logging.getLogger(__name__)

class ExpBase:
    def __init__(self, config):
        set_seed(config.seed)
        self.seed = config.seed
        self.scaler = GradScaler()
        self.model_name = config.model.name
        self.model_config = config.model.params
        self.exp_config = config.exp
        self.data_config = config.data

        self.epochs = config.exp.epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.best_kappa = 0

        self.epoch_count = 1

        df: CustomDataset = getattr(customdataset, self.data_config.name)(seed=self.seed, **self.data_config)
        self.train, self.test = df.train, df.test
        self.feature_columns = df.feature_columns
        self.target_column = df.target_column
        self.train_dataset, self.val_dataset, self.train_loader, self.val_loader, self.test_loader = df.prepare_loaders()
        self.id = df.id
        self.num_labels = df.num_labels

        # Loss function
        # self.loss_fn = CrossEntropyLoss().to(self.device)
        
        # class_counts = self.train[self.target_column].value_counts()
        # class_weights = 1. / class_counts
        # class_weights = class_weights / class_weights.sum()

        class_weights = [0.090092, 0.023882, 0.017961, 0.028730, 0.116284, 0.723050]

        # クラス重みを設定した損失関数
        weights = torch.tensor(class_weights, dtype=torch.float)


        # Loss function
        self.loss_fn = CrossEntropyLoss(weight=weights).to(self.device)


    def choice_layer(self):

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # for name, param in self.model.model.deberta.encoder.layer.named_parameters():
        for name, param in self.model.model.bert.encoder.layer[10:].named_parameters():  
            param.requires_grad = True
        for name, param in self.model.named_parameters():
            if param.requires_grad : 
                print(name)
        
    def print_model_parameters(self, indent=0):
            if hasattr(self.model, 'named_parameters'):
                for name, param in self.model.named_parameters():
                    print(f"{name}: {param.size()}")
            else:
                    print("The model does not have named_parameters method")

    def train_epoch(self,n_examples):
        self.model.train()
        losses = []
        self.llm_pred_data = self.train[['essay_id', 'score']]

        for d in tqdm.tqdm(self.train_loader):
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            labels = d["label"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            for idx, essay_id in enumerate(d["idx"]):
                score = int(outputs.logits[idx].argmax().item() + 1)
                self.llm_pred_data.loc[self.llm_pred_data['essay_id'] == essay_id, 'prediction_score'] = score 

            loss = self.loss_fn(outputs.logits, labels)
            losses.append(loss.item())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

            
        return np.mean(losses)
    
    def eval_model(self,n_examples):
        self.model.eval()
        losses = []
        true_labels = []
        pred_labels = []
        
        with torch.no_grad():
            for d in self.val_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                labels = d["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                for idx, essay_id in enumerate(d["idx"]):
                    score = int (outputs.logits[idx].argmax().item() + 1)
                    self.llm_pred_data.loc[self.llm_pred_data['essay_id'] == essay_id, 'prediction_score'] = score 

                _, preds = torch.max(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, labels)

                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())
                losses.append(loss.item())
            self.llm_pred_data.to_csv(f'prediction_score_{self.epoch_count}.csv', index=False)
            self.epoch_count+=1

        kappa = cohen_kappa_score(true_labels,pred_labels,weights='quadratic')
        return kappa, np.mean(losses)
    
    def get_predictions(self):
        self.model.eval()
        predictions = []
    
        with torch.no_grad():
            for d in self.test_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                 
                _, preds = torch.max(outputs.logits, dim=1)

                preds += 1

                predictions.extend(preds)


        predictions = [pred.item() for pred in predictions]
        return predictions

    def run(self):
        model_config = self.get_model_config()
        self.model = get_classifier(
            self.model_name,
            device = self.device, 
            model_config = model_config, 
            num_labels=self.num_labels,
            seed=self.seed
        )
        self.model.add_layer(additional_layers=1)

        if(self.model_name == "deberta"):
            self.model.model.resize_token_embeddings(len(self.train_dataset.tokenizer))

        # Optimizer and scheduler
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.model,self.model_name, lr=2e-5,  weight_decay=0.01, lr_decay=0.95)
        self.optimizer = AdamW(optimizer_grouped_parameters)
        # self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        self.total_steps = len(self.train_loader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=self.total_steps)

        logger.info(f"model name: {self.model_name} device: {self.device}")
        # self.choice_layer()

        best_epoch = 0

        for epoch in range(self.epochs):
            logger.info(f'Epoch {epoch + 1}/{self.epochs}')

            start_time = time()
            train_loss = self.train_epoch(len(self.train_dataset))
            logger.info(f'Train loss: {train_loss} train time: {time() - start_time}')

            start_time = time()
            val_kappa, val_loss = self.eval_model(len(self.val_dataset))
            logger.info(f'Val loss {val_loss} kappa {val_kappa} eval time: {time() - start_time}')
            
            torch.save(self.model.state_dict(), f'best_model_state{epoch+1}.bin')
            
            if val_kappa >= self.best_kappa:
                self.best_kappa = val_kappa
                best_epoch = epoch  + 1
                # torch.save(self.model.state_dict(), f'best_model_state{epoch+1}.bin')
        print(f'Best model is {best_epoch} epoch.')
      
              
        if os.path.exists(f'best_model_state{best_epoch}.bin'):
            self.model.load_state_dict(torch.load(f'best_model_state{best_epoch}.bin'))
        else:
            logger.error("No model file found. Ensure training completes successfully.")


        preds = self.get_predictions()

        submission_df = pd.DataFrame({
            'essay_id': self.id,
            'score': preds
        })
        print(submission_df)

        submission_df.to_csv('submission.csv', index=False)
    
    def get_model_config(self, *args, **kwargs):
        raise NotImplementedError()
   
class ExpSimple(ExpBase):
    def __init__(self, config):
        super().__init__(config)
    
    def get_model_config(self, *args, **kwargs):
        return self.model_config

class ExpStacking(ExpBase):
    def __init__(self, config):
        super().__init__(config)

        self.input_dim = 6
        self.output_dim = 6
        self.n_splits = 7

    def each_fold(self, i_fold, train_data, val_data):
        x, y = self.get_x_y(train_data)

        model_config = self.get_model_config(i_fold=i_fold, x=x, y=y, val_data=val_data)
        model = get_tree_classifier(
            self.model_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_config=model_config,
        )
        start = time()
        model.fit(
            x,
            y,
            eval_set=(val_data[self.feature_columns], val_data[self.target_column].values.squeeze()),
        )
        end = time() - start
        logger.info(f"[Fit {self.model_name}] Time: {end}")
        return model, end
    
    def run(self):
        self.train['score'] = self.train['score']-1
        

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        y_test_pred_all = []
        score_all = 0
        for i_fold, (train_idx, val_idx) in enumerate(skf.split(self.train, self.train[self.target_column])):

            train_data, val_data = self.train.iloc[train_idx], self.train.iloc[val_idx]
            model, time = self.each_fold(i_fold, train_data, val_data)

            kappa = cal_kappa_score(model, val_data, self.feature_columns, self.target_column)

            logger.info(
                f"[{self.model_name} results ({i_fold+1} / {self.n_splits})] kappa score: {kappa}"
            )

            score_all += kappa

            y_test_pred_all.append(
                model.predict_proba(self.test[self.feature_columns]).reshape(-1, 1, 6)
            )
        
        # y_test_pred_all = np.argmax(np.concatenate(y_test_pred_all, axis=1).mean(axis=1), axis=1)
        # submit_df = pd.DataFrame(self.id)
        # print(y_test_pred_all)
        # print(submit_df)
        # exit()
        # submit_df['score'] = y_test_pred_all+1

        # print(submit_df)
        # submit_df.to_csv("submit.csv", index=False)

        logger.info(f" {self.model_name} score average: {score_all/self.n_splits} ")

    def get_model_config(self, *args, **kwargs):
            return self.model_config

    def get_x_y(self, train_data):
        x, y = train_data[self.feature_columns], train_data[self.target_column].values.squeeze()
        return x, y

class ExpOptuna(ExpBase):
    def __init__(self, config):
        super().__init__(config)
        self.n_trials = config.exp.n_trials
        self.n_startup_trials = config.exp.n_startup_trials
        self.n_splits = config.n_splits

        self.storage = to_absolute_path(config.exp.storage)
        self.study_name = config.exp.study_name
        self.cv = config.exp.cv
        self.n_jobs = config.exp.n_jobs
        self.input_dim = 7
        self.output_dim = 6
        self.verbose = 1.1
        self.task = config.exp.task

        # ディレクトリが存在しない場合は作成
        os.makedirs(self.storage, exist_ok=True)

    def each_fold(self, i_fold, train_data, val_data):
        x, y = self.get_x_y(train_data)

        model_config = self.get_model_config(i_fold=i_fold, x=x, y=y, val_data=val_data)
        model = get_tree_classifier(
            self.model_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_config=model_config,
        )
        start = time()
        model.fit(
            x,
            y,
            eval_set=(val_data[self.feature_columns], val_data[self.target_column].values.squeeze()),
        )
        end = time() - start
        logger.info(f"[Fit {self.model_name}] Time: {end}")
        return model, end

    def run(self):
        self.train['score'] = self.train['score'] - 1
        

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        y_test_pred_all = []
        score_all = 0
        for i_fold, (train_idx, val_idx) in enumerate(skf.split(self.train, self.train[self.target_column])):
            # if len(self.writer["fold"]) != 0 and self.writer["fold"][-1] >= i_fold:
            #     logger.info(f"Skip {i_fold + 1} fold. Already finished.")
            #     continue

            train_data, val_data = self.train.iloc[train_idx], self.train.iloc[val_idx]
            model, time = self.each_fold(i_fold, train_data, val_data)

            kappa = cal_kappa_score(model, val_data, self.feature_columns, self.target_column)

            logger.info(
                f"[{self.model_name} results ({i_fold+1} / {self.n_splits})] kappa score: {kappa}"
            )

            score_all += kappa


            y_test_pred_all.append(
                model.predict_proba(self.test[self.feature_columns]).reshape(-1, 1, 6)
            )
        final_score = score_all / self.n_splits
        # logger.info(f"Average Kappa Score: {final_score}")
        logger.info(f" {self.model_name} kappa score average: {final_score} ")
        # return final_score, y_test_pred_all
    

        
        # if self.exp_config.delete_study:
        #     for i in range(self.n_splits):
        #         optuna.delete_study(
        #             study_name=f"{self.exp_config.study_name}_{i}",
        #             storage=f"sqlite:///{to_absolute_path(self.exp_config.storage)}/optuna.db",
        #         )
        #         print(f"delete successful in {i}")
        #     return
        # self.optimize_hyperparameters()
        # super().run()

    # def optimize_hyperparameters(self):
    #     def objective(trial):
    #         # チューニングするハイパーパラメータを定義
    #         model_config = self.get_model_config(trial, deepcopy(self.model_config))

    #         self.model = get_tree_classifier(
    #             self.model_name,
    #             input_dim=self.input_dim,
    #             output_dim=self.output_dim,
    #             model_config=model_config,
    #             verbose=self.verbose,
    #             seed=self.seed
    #         )

    #         # self.model.add_layer(additional_layers=1)


    #         optimizer_grouped_parameters = get_optimizer_grouped_parameters(
    #             self.model, self.model_name, lr=model_config["learning_rate"], weight_decay=0.01, lr_decay=0.95
    #         )
    #         self.optimizer = AdamW(optimizer_grouped_parameters)
    #         self.total_steps = len(self.train_loader) * self.epochs
    #         self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=self.total_steps)

    #         best_kappa = 0

    #         for epoch in range(self.epochs):
    #             train_loss = self.train_epoch(len(self.train_dataset))
    #             val_kappa, val_loss = self.eval_model(len(self.val_dataset))

    #             if val_kappa > best_kappa:
    #                 best_kappa = val_kappa

    #             trial.report(val_loss, epoch)
    #             if trial.should_prune():
    #                 raise optuna.exceptions.TrialPruned()

    #         return best_kappa

    #     study = optuna.create_study(
    #         direction="maximize",
    #         study_name=self.study_name,
    #         storage=f"sqlite:///{to_absolute_path(self.storage)}/optuna.db",
    #         load_if_exists=True
    #     )
    #     study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

    #     logger.info(f"best trial: {study.best_trial.value}")
    #     logger.info(f"best parameter: {study.best_trial.params}")

    #     # ベストなハイパーパラメータを使用して最終モデルをトレーニング
    #     self.model_config.update(study.best_trial.params)
    #     super().run()

    # def get_model_config(self, trial, default_config):
    #     if self.model_name == "xgboost":
    #         return xgboost_config(trial, default_config)
    #     elif self.model_name == "lightgbm":
    #         return lightgbm_config(trial, default_config)
    #     elif self.model_name == "catboost":
    #         return catboost_config(trial, default_config)
    #     else:
    #         raise ValueError("Invalid model name")
    

    def get_model_config(self, i_fold, x, y, val_data):
        op = OptimParam(
            self.model_name,
            default_config=self.model_config,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            X=x,
            y=y,
            val_data=val_data,
            columns = self.feature_columns,
            target_column=self.target_column,
            n_trials=self.n_trials,
            n_startup_trials=self.n_startup_trials,
            storage=self.storage,
            study_name=f"{self.study_name}_{i_fold}",
            cv=self.cv,
            n_jobs=self.n_jobs,
            seed=self.seed,
            task = 'classifier'
        )
        return op.get_best_config()

    def get_x_y(self, train_data):
        x, y = train_data[self.feature_columns], train_data[self.target_column].values.squeeze()
        return x, y



