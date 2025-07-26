from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.candidates_generator import CandidatesGenerator
from src.cat_ranker import CatRanker
from src.dataset import Dataset
from src.embedding_generator import EmbeddingGenerator
from src.metrics_calculator import MetricsCalculator
from src.schema.config import Config
from src.two_tower_model import (
    TwoTowerModel,
    define_cat_dim,
    train,
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
    article_item2vec_embs = emb_generator.create_item2vec_embeddings(
        dataset.past_trans_df
    )

    ttm = TwoTowerModel(cfg)
    ttm = train(cfg, ttm, dataset, result_dir)
    article_ttm_embs, customer_ttm_embs = emb_generator.create_ttm_embeddings(
        dataset, ttm
    )

    cand_generator = CandidatesGenerator(cfg)
    cand_generator.generate_candidates(
        dataset, article_item2vec_embs, article_ttm_embs, customer_ttm_embs
    )
    cand_generator.evaluate_candidates(dataset, result_dir)

    metrics_calculator = MetricsCalculator(cfg)
    cat_ranker = CatRanker(cfg, metrics_calculator)

    cat_ranker.preprocess(
        dataset,
        cand_generator.candidates_df,
        article_item2vec_embs,
        article_ttm_embs,
        customer_ttm_embs,
    )
    cat_ranker.train(result_dir).evaluate(result_dir, dataset)


if __name__ == "__main__":
    main()
