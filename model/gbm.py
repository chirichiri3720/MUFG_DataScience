import xgboost as xgb
import lightgbm as lgb
import catboost as cbt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .base_model import BaseClassifier

# from .utils import cohen_kappa_eval

class XGBoostClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=self.output_dim,
            # eval_metric=f1_micro,
            early_stopping_rounds=50,
            **self.model_config
        )

    def fit(self, X, y, eval_set):
        self.model.fit(X, y, eval_set=[eval_set], verbose=self.verbose > 0)

class LightGBMClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=42) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            verbose=self.verbose,
            random_state=seed,
            num_class=output_dim,
            **self.model_config,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns

        self.model.fit(
            X,
            y,
            eval_set=[eval_set],
            # eval_metric=f1_micro_lgb,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=self.verbose > 0)],
        )

class CBTClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=42) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = cbt.CatBoostClassifier(
            loss_function="MultiClass",
            verbose=self.verbose,
            random_seed=seed,
            eval_metric='WKappa',
            **self.model_config,
            early_stopping_rounds=50, #optunaはこっち
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns

        self.model.fit(
            X,
            y,
            eval_set=[eval_set],
        )