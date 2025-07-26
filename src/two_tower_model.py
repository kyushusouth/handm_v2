from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.dataset import Dataset as MyDataset
from src.logger import get_logger
from src.schema.config import Config

logger = get_logger(__file__)


def define_cat_dim(dim: int, max_emb_size: int) -> int:
    """
    Rule of thumb to pick embedding size corresponding to dim.
    ref: https://github.com/fastai/fastai/blob/master/fastai/tabular/model.py#L12-L16
    """
    cat_dim = (dim + 1, min(max_emb_size, round(1.6 * dim**0.56)))  # unknown: 0
    return cat_dim


class TTMDataset(Dataset):
    def __init__(
        self,
        cfg: Config,
        trans_df: pd.DataFrame,
        customer_df: pd.DataFrame,
        article_df: pd.DataFrame,
    ) -> None:
        self.cfg = cfg
        trans_df = self.create_ttm_features(trans_df, customer_df, article_df)
        trans_df = self.numerical_preprocess(
            trans_df, self.cfg.model.params.ttm.user_num_cols
        )

        self.trans_df = trans_df
        self.user_num_feature_df = (
            trans_df[self.cfg.model.params.ttm.user_num_cols]
            .fillna(0)
            .reset_index(drop=True)
        )
        self.user_cat_feature_df = trans_df[
            self.cfg.model.params.ttm.user_cat_cols
        ].reset_index(drop=True)
        self.item_cat_feature_df = trans_df[
            self.cfg.model.params.ttm.item_cat_cols
        ].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.user_num_feature_df)

    def __getitem__(self, index: int) -> tuple[str, str, dict[str, torch.Tensor]]:
        customer_id = self.trans_df.iloc[index]["customer_id"]
        article_id = self.trans_df.iloc[index]["article_id"]
        user_num_feature = torch.tensor(self.user_num_feature_df.iloc[index].values).to(
            torch.float32
        )
        user_cat_feature = torch.tensor(self.user_cat_feature_df.iloc[index].values).to(
            torch.long
        )
        item_cat_feature = torch.tensor(self.item_cat_feature_df.iloc[index].values).to(
            torch.long
        )
        inputs = {
            "user_num_feature": user_num_feature,
            "user_cat_feature": user_cat_feature,
            "item_cat_feature": item_cat_feature,
        }
        return customer_id, article_id, inputs

    def create_ttm_features(
        self,
        df: pd.DataFrame,
        customer_df: pd.DataFrame,
        article_df: pd.DataFrame,
    ) -> pd.DataFrame:
        user_feature_cols = [
            "FN",
            "Active",
            "club_member_status",
            "fashion_news_frequency",
            "age",
        ]
        user_features = customer_df[["customer_id"] + user_feature_cols].copy()
        for col in user_feature_cols:
            if col == "age":
                continue
            cats = customer_df.loc[customer_df[col].notnull()][col].unique().tolist()
            encoding_cat_feature_dict = dict([(c, i + 1) for i, c in enumerate(cats)])
            user_features[col] = (
                user_features[col].map(encoding_cat_feature_dict).fillna(0).astype(int)
            )

        item_feature_cols = [
            "colour_group_name",
            "department_name",
            "department_no",
            "garment_group_name",
            "graphical_appearance_name",
            "index_group_name",
            "index_name",
            "perceived_colour_master_name",
            "perceived_colour_value_name",
            "product_group_name",
            "product_type_name",
            "section_name",
        ]
        item_features = article_df[["article_id"] + item_feature_cols].copy()
        for col in item_feature_cols:
            cats = article_df.loc[article_df[col].notnull()][col].unique().tolist()
            encoding_cat_feature_dict = dict([(c, i + 1) for i, c in enumerate(cats)])
            item_features[col] = (
                item_features[col].map(encoding_cat_feature_dict).fillna(0).astype(int)
            )

        df = df.merge(user_features, on="customer_id", how="left").merge(
            item_features, on="article_id", how="left"
        )
        return df

    def numerical_preprocess(self, df: pd.DataFrame, num_cols: int) -> pd.DataFrame:
        """対数変換しておく"""
        for c in num_cols:
            df[c] = np.log1p(df[c])
        return df


class TTMItemDataset(Dataset):
    def __init__(self, cfg: Config, article_df: pd.DataFrame):
        self.item_feature_df = self.create_item_features(article_df)
        self.item_cat_feature_df = self.item_feature_df[
            cfg.model.params.ttm.item_cat_cols
        ]

    def __len__(self):
        return len(self.item_cat_feature_df)

    def __getitem__(self, index: int) -> tuple[str, dict[str, torch.Tensor]]:
        article_id = self.item_feature_df.iloc[index]["article_id"]
        item_cat_feature = torch.tensor(self.item_cat_feature_df.iloc[index].values).to(
            torch.long
        )
        inputs = {"item_cat_feature": item_cat_feature}
        return article_id, inputs

    def create_item_features(self, article_df: pd.DataFrame) -> pd.DataFrame:
        item_feature_cols = [
            "colour_group_name",
            "department_name",
            "department_no",
            "garment_group_name",
            "graphical_appearance_name",
            "index_group_name",
            "index_name",
            "perceived_colour_master_name",
            "perceived_colour_value_name",
            "product_group_name",
            "product_type_name",
            "section_name",
        ]
        item_features = article_df[["article_id"] + item_feature_cols].copy()
        for col in item_feature_cols:
            cats = article_df.loc[article_df[col].notnull()][col].unique().tolist()
            encoding_cat_feature_dict = dict([(c, i + 1) for i, c in enumerate(cats)])
            item_features[col] = (
                item_features[col].map(encoding_cat_feature_dict).fillna(0).astype(int)
            )
        return item_features


class TTMUserDataset(Dataset):
    def __init__(self, cfg: Config, customer_df: pd.DataFrame):
        self.user_features_df = self.create_user_features(customer_df)
        self.user_num_feature_df = self.user_features_df[
            cfg.model.params.ttm.user_num_cols
        ]
        self.user_cat_feature_df = self.user_features_df[
            cfg.model.params.ttm.user_cat_cols
        ]

    def __len__(self):
        return len(self.user_features_df)

    def __getitem__(self, index: int) -> tuple[str, dict[str, torch.Tensor]]:
        customer_id = self.user_features_df.iloc[index]["customer_id"]
        user_num_feature = torch.tensor(self.user_num_feature_df.iloc[index].values).to(
            torch.float32
        )
        user_cat_feature = torch.tensor(self.user_cat_feature_df.iloc[index].values).to(
            torch.long
        )
        inputs = {
            "user_num_feature": user_num_feature,
            "user_cat_feature": user_cat_feature,
        }
        return customer_id, inputs

    def create_user_features(self, customer_df: pd.DataFrame) -> pd.DataFrame:
        user_feature_cols = [
            "FN",
            "Active",
            "club_member_status",
            "fashion_news_frequency",
            "age",
        ]
        user_features = customer_df[["customer_id"] + user_feature_cols].copy()
        for col in user_feature_cols:
            if col == "age":
                continue
            cats = customer_df.loc[customer_df[col].notnull()][col].unique().tolist()
            encoding_cat_feature_dict = dict([(c, i + 1) for i, c in enumerate(cats)])
            user_features[col] = (
                user_features[col].map(encoding_cat_feature_dict).fillna(0).astype(int)
            )
        return user_features


class ItemEmbeddingModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.cat_emb = nn.ModuleList(
            [
                nn.Embedding(x, y, padding_idx=0)
                for x, y in cfg.model.params.ttm.item_cat_dims
            ]
        )
        n_cat_emb_out = sum([y for x, y in cfg.model.params.ttm.item_cat_dims])
        self.cat_proj = nn.Sequential(
            nn.Linear(n_cat_emb_out, cfg.model.params.ttm.item_cat_hidden_size),
            nn.LayerNorm(cfg.model.params.ttm.item_cat_hidden_size),
        )
        self._init_weight(self.cat_proj)

        head_hidden_size = cfg.model.params.ttm.item_cat_hidden_size
        self.head = nn.Sequential(
            nn.Linear(head_hidden_size, head_hidden_size),
            nn.BatchNorm1d(head_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(head_hidden_size, cfg.model.params.ttm.emb_size),
        )
        self._init_weight(self.head)

    def _init_weight(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, cat_features: torch.Tensor) -> torch.Tensor:
        cat_embs = [
            emb_layer(cat_features[:, j]) for j, emb_layer in enumerate(self.cat_emb)
        ]
        cat_embs = torch.cat(cat_embs, 1)
        cat_embs = self.cat_proj(cat_embs)

        concat_embs = cat_embs
        embs = self.head(concat_embs)

        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs


class UserEmbeddingModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.cat_emb = nn.ModuleList(
            [
                nn.Embedding(x, y, padding_idx=0)
                for x, y in cfg.model.params.ttm.user_cat_dims
            ]
        )
        n_cat_emb_out = sum([y for x, y in cfg.model.params.ttm.user_cat_dims])
        self.cat_proj = nn.Sequential(
            nn.Linear(n_cat_emb_out, cfg.model.params.ttm.user_cat_hidden_size),
            nn.LayerNorm(cfg.model.params.ttm.user_cat_hidden_size),
        )
        self._init_weight(self.cat_proj)

        self.num_emb = nn.Sequential(
            nn.BatchNorm1d(len(cfg.model.params.ttm.user_num_cols)),
            nn.Linear(
                len(cfg.model.params.ttm.user_num_cols),
                cfg.model.params.ttm.user_num_hidden_size,
            ),
            nn.BatchNorm1d(cfg.model.params.ttm.user_num_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(
                cfg.model.params.ttm.user_num_hidden_size,
                cfg.model.params.ttm.user_num_hidden_size,
            ),
            nn.BatchNorm1d(cfg.model.params.ttm.user_num_hidden_size),
            nn.LeakyReLU(),
        )
        self._init_weight(self.num_emb)

        head_hidden_size = (
            cfg.model.params.ttm.user_cat_hidden_size
            + cfg.model.params.ttm.user_num_hidden_size
        )
        self.head = nn.Sequential(
            nn.Linear(head_hidden_size, head_hidden_size),
            nn.BatchNorm1d(head_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(head_hidden_size, cfg.model.params.ttm.emb_size),
        )
        self._init_weight(self.head)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, num_features: torch.Tensor, cat_features: torch.Tensor
    ) -> torch.Tensor:
        num_embs = self.num_emb(num_features)

        cat_embs = [
            emb_layer(cat_features[:, j]) for j, emb_layer in enumerate(self.cat_emb)
        ]
        cat_embs = torch.cat(cat_embs, 1)
        cat_embs = self.cat_proj(cat_embs)

        concat_embs = torch.cat((num_embs, cat_embs), 1)
        embs = self.head(concat_embs)

        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs


class TwoTowerModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.item_emb_model = ItemEmbeddingModel(cfg)
        self.user_emb_model = UserEmbeddingModel(cfg)
        self.logit_scale = nn.Parameter(torch.ones(1) * 2.5)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        item_embs = self.item_emb_model(inputs["item_cat_feature"])
        user_embs = self.user_emb_model(
            inputs["user_num_feature"], inputs["user_cat_feature"]
        )
        logit_scale = self.logit_scale.exp()
        logits = torch.matmul(user_embs, item_embs.T) * logit_scale.unsqueeze(0)
        return logits


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


def plot_learning_curve(epoch_loss_history: list[float], result_dir: Path):
    plt.figure(figsize=(12, 8))
    plt.plot([i + 1 for i in range(len(epoch_loss_history))], epoch_loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(result_dir.joinpath("ttm_learning_curve.png"))
    plt.close()


def train(cfg: Config, model: TwoTowerModel, dataset: MyDataset, result_dir: Path):
    logger.info("two tower model training")
    ttm_dataset = TTMDataset(
        cfg,
        dataset.past_trans_df.drop(columns=["t_dat", "price", "sales_channel_id"])
        .drop_duplicates(["customer_id", "article_id"])
        .copy(),
        dataset.customer_df,
        dataset.article_df,
    )
    ttm_dataloader = DataLoader(
        ttm_dataset,
        batch_size=cfg.model.params.ttm.train_batch_size,
        shuffle=True,
        num_workers=cfg.model.params.ttm.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    model.train()
    epoch_loss_history = []
    for epoch in range(cfg.model.params.ttm.max_epoch):
        print(f"epoch: {epoch + 1}")
        iter_loss_history = []
        for batch in ttm_dataloader:
            _, _, inputs = batch
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            y_preds = model(inputs)
            loss = contrastive_loss(y_preds)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iter_loss_history.append(loss.item())
        epoch_loss_history.append(sum(iter_loss_history) / len(iter_loss_history))

    plot_learning_curve(epoch_loss_history, result_dir)
    return model
