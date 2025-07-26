from pathlib import Path

import catboost as cat
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.metrics_calculator import MetricsCalculator
from src.ranker import Ranker
from src.schema.config import Config


class CatRanker(Ranker):
    def __init__(
        self,
        cfg: Config,
        metrics_calculator: MetricsCalculator,
    ) -> None:
        super().__init__(cfg, metrics_calculator)

    def train(self, result_dir: Path) -> "CatRanker":
        train_pool = cat.Pool(
            self.train_trans_df[
                self.cfg.model.features.cat + self.cfg.model.features.num
            ],
            self.train_trans_df["purchased"],
            group_id=self.train_trans_df["customer_id"].values,
            cat_features=range(len(self.cfg.model.features.cat)),
        )
        val_pool = cat.Pool(
            self.val_trans_df[
                self.cfg.model.features.cat + self.cfg.model.features.num
            ],
            self.val_trans_df["purchased"],
            group_id=self.val_trans_df["customer_id"].values,
            cat_features=range(len(self.cfg.model.features.cat)),
        )
        self.ranker = cat.CatBoostRanker(
            iterations=self.cfg.model.params.cat.iterations,
            learning_rate=self.cfg.model.params.cat.learning_rate,
            depth=self.cfg.model.params.cat.depth,
            loss_function=self.cfg.model.params.cat.loss_function,
            eval_metric=self.cfg.model.params.cat.eval_metric,
            random_seed=self.cfg.seed,
            allow_writing_files=False,
        )
        self.ranker.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=self.cfg.model.params.cat.early_stopping_round,
        )

        metric_values = self.ranker.get_evals_result()["validation"][
            f"{self.cfg.model.params.cat.eval_metric};type=Base"
        ]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        ax.plot(metric_values)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Score")
        plt.tight_layout()
        plt.savefig(
            result_dir.joinpath(f"{self.__class__.__name__}_learning_curve.png")
        )
        plt.close(fig)

        feat_imp_df = self.ranker.get_feature_importance(
            train_pool, type="PredictionValuesChange", prettified=True
        )
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        sns.barplot(feat_imp_df, x="Importances", y="Feature Id")
        plt.tight_layout()
        plt.savefig(
            result_dir.joinpath(f"{self.__class__.__name__}_feature_importance.png")
        )
        plt.close(fig)

        explainer = shap.TreeExplainer(self.ranker)
        shap_samples = self.val_trans_df.sample(
            self.cfg.exp.summary_plot_num_sample, random_state=self.cfg.seed
        )[self.cfg.model.features.cat + self.cfg.model.features.num]
        shap_values = explainer.shap_values(shap_samples)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        shap.summary_plot(shap_values, shap_samples, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(
            result_dir.joinpath(f"{self.__class__.__name__}_shap_summary_plot_bar.png")
        )
        plt.close(fig)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        shap.summary_plot(shap_values, shap_samples, plot_type="dot", show=False)
        plt.tight_layout()
        plt.savefig(
            result_dir.joinpath(f"{self.__class__.__name__}_shap_summary_plot_dot.png")
        )
        plt.close(fig)

        return self
