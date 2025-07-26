from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.candidates_generator import CandidatesGenerator
from src.cat_ranker import CatRanker
from src.dataset import Dataset
from src.embedding_generator import EmbeddingGenerator
from src.metrics_calculator import MetricsCalculator
from src.schema.config import Config
from src.two_tower_model import (
    TTMDataset,
    TwoTowerModel,
    contrastive_loss,
    define_cat_dim,
)
from src.utils import set_seed

plt.rcParams["font.size"] = 18
sns.set_style("whitegrid")


def main():
    cfg = Config.load(Path(__file__).parent.parent.joinpath("conf", "config.yaml"))
    current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = Path(__file__).parent.parent.joinpath("result", current_datetime)
    result_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    dataset = Dataset(cfg)

    user_cat_dims = []
    for col in cfg.model.params.ttm.user_cat_cols:
        user_cat_dims.append(
            define_cat_dim(
                dataset.customer_df.loc[
                    dataset.customer_df[col].notnull(), col
                ].nunique(),
                cfg.model.params.ttm.cat_max_dims,
            )
        )
    cfg.model.params.ttm.user_cat_dims = user_cat_dims

    item_cat_dims = []
    for col in cfg.model.params.ttm.item_cat_cols:
        item_cat_dims.append(
            define_cat_dim(
                dataset.article_df.loc[
                    dataset.article_df[col].notnull(), col
                ].nunique(),
                cfg.model.params.ttm.cat_max_dims,
            )
        )
    cfg.model.params.ttm.item_cat_dims = item_cat_dims

    emb_generator = EmbeddingGenerator(cfg)
    all_embeddings = {}
    all_embeddings["item2vec"] = emb_generator.create_item2vec_embeddings(
        dataset.past_trans_df
    )

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
    ttm = TwoTowerModel(cfg)
    ttm.to(device)

    optimizer = torch.optim.Adam(ttm.parameters(), lr=0.001, betas=(0.9, 0.999))

    ttm.train()
    epoch_loss_history = []
    for epoch in range(1):
        print(f"epoch: {epoch + 1}")
        iter_loss_history = []
        for batch in ttm_dataloader:
            customer_id, article_id, inputs = batch
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            y_preds = ttm(inputs)
            loss = contrastive_loss(y_preds)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iter_loss_history.append(loss.item())
            print(f"loss: {loss}")
            break
        epoch_loss_history.append(sum(iter_loss_history) / len(iter_loss_history))

    plt.figure(figsize=(12, 8))
    plt.plot([i + 1 for i in range(1)], epoch_loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(result_dir.joinpath("ttm_learning_curve.png"))
    plt.close()

    article_ttm_embs, customer_ttm_embs = emb_generator.create_ttm_embeddings(
        dataset, ttm
    )

    cand_generator = CandidatesGenerator(cfg)
    cand_generator.generate_candidates(
        dataset, all_embeddings, article_ttm_embs, customer_ttm_embs
    )
    cand_generator.evaluate_candidates(dataset, result_dir)

    metrics_calculator = MetricsCalculator(cfg)
    cat_ranker = CatRanker(cfg, metrics_calculator)

    cat_ranker.preprocess(dataset, cand_generator.candidates_df, all_embeddings)
    cat_ranker.train(result_dir).evaluate(result_dir, dataset)


if __name__ == "__main__":
    main()
