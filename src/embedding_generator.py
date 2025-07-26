import gensim
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import Dataset
from src.logger import get_logger
from src.schema.config import Config
from src.two_tower_model import TTMItemDataset, TTMUserDataset, TwoTowerModel

logger = get_logger(__file__)


class EmbeddingGenerator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def create_item2vec_embeddings(
        self, trans_df: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        logger.info("Creating item2vec embeddings...")

        purchase_histories = (
            trans_df.sort_values("t_dat", ascending=True)
            .groupby("customer_id")["article_id"]
            .apply(list)
        )
        model = gensim.models.Word2Vec(
            sentences=purchase_histories,
            vector_size=self.cfg.model.params.item2vec.vector_size,
            window=self.cfg.model.params.item2vec.window,
            min_count=self.cfg.model.params.item2vec.min_count,
            workers=self.cfg.model.params.item2vec.workers,
            sg=self.cfg.model.params.item2vec.sg,
            seed=self.cfg.seed,
        )
        item_embeddings = {
            article_id: model.wv[article_id] for article_id in model.wv.index_to_key
        }
        return item_embeddings

    def create_ttm_embeddings(
        self, dataset: Dataset, ttm: TwoTowerModel
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        logger.info("Creating TwoTowerModel embeddings...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        user_dataset = TTMUserDataset(self.cfg, dataset.customer_df)
        user_dataloader = DataLoader(
            user_dataset,
            batch_size=self.cfg.model.params.ttm.train_batch_size,
            shuffle=False,
            num_workers=self.cfg.model.params.ttm.num_workers,
        )
        customer_embs = {}
        for batch in user_dataloader:
            customer_ids, inputs = batch
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            with torch.no_grad():
                user_embs = ttm.user_emb_model(
                    inputs["user_num_feature"], inputs["user_cat_feature"]
                ).numpy()
            for c_i in customer_ids:
                customer_embs[c_i] = user_embs[c_i]

        item_dataset = TTMItemDataset(self.cfg, dataset.customer_df)
        item_dataloader = DataLoader(
            item_dataset,
            batch_size=self.cfg.model.params.ttm.train_batch_size,
            shuffle=False,
            num_workers=self.cfg.model.params.ttm.num_workers,
        )
        article_embs = {}
        for batch in item_dataloader:
            article_ids, inputs = batch
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            with torch.no_grad():
                item_embs = ttm.item_emb_model(inputs["item_cat_feature"]).numpy()
            for a_i in article_ids:
                customer_embs[a_i] = item_embs[a_i]

        return article_embs, customer_embs
