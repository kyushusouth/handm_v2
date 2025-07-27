from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.candidates_generator import CandidatesGenerator
from src.cat_ranker import CatRanker
from src.dataset import Dataset
from src.embedding_generator import EmbeddingGenerator
from src.metrics_calculator import MetricsCalculator
from src.mmr_reranker import MMRReranker
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

    dataset.article_df[["article_id"]].merge(
        dataset.past_trans_df.groupby("article_id")
        .size()
        .to_frame(name="purchase_cnt")
        .reset_index(),
        on="article_id",
        how="left",
    )

    # user_cat_dims = []
    # for col in cfg.model.params.ttm.user_cat_cols:
    #     user_cat_dims.append(
    #         define_cat_dim(
    #             dataset.customer_df.loc[
    #                 dataset.customer_df[col].notnull(), col
    #             ].nunique(),
    #             cfg.model.params.ttm.cat_max_dims,
    #         )
    #     )
    # cfg.model.params.ttm.user_cat_dims = user_cat_dims

    # item_cat_dims = []
    # for col in cfg.model.params.ttm.item_cat_cols:
    #     item_cat_dims.append(
    #         define_cat_dim(
    #             dataset.article_df.loc[
    #                 dataset.article_df[col].notnull(), col
    #             ].nunique(),
    #             cfg.model.params.ttm.cat_max_dims,
    #         )
    #     )
    # cfg.model.params.ttm.item_cat_dims = item_cat_dims

    emb_generator = EmbeddingGenerator(cfg)
    article_item2vec_embs = emb_generator.create_item2vec_embeddings(
        dataset.past_trans_df
    )

    # ttm = TwoTowerModel(cfg)
    # ttm = train(cfg, ttm, dataset, result_dir)
    # article_ttm_embs, customer_ttm_embs = emb_generator.create_ttm_embeddings(
    #     dataset, ttm
    # )
    article_ttm_embs = {}
    customer_ttm_embs = {}

    metrics_calculator = MetricsCalculator(cfg)

    cand_generator = CandidatesGenerator(cfg)
    cand_generator.generate_candidates(
        dataset, article_item2vec_embs, article_ttm_embs, customer_ttm_embs
    )
    cand_generator.evaluate_candidates(dataset, result_dir, metrics_calculator)

    cat_ranker = CatRanker(cfg, metrics_calculator)
    cat_ranker.preprocess(
        dataset,
        cand_generator.candidates_df,
        article_item2vec_embs,
        article_ttm_embs,
        customer_ttm_embs,
    )
    cat_ranker.train(result_dir)
    rec_results = cat_ranker.evaluate(result_dir, dataset)

    reranker = MMRReranker(cfg)

    purchase_cnt_df = dataset.past_trans_df.groupby("article_id").size()
    purchase_cnt_df = purchase_cnt_df / purchase_cnt_df.max()
    item_purchase_cnt_map = purchase_cnt_df.to_dict()
    ws = [0.1, 0.5, 0.9]
    metrics = []
    for w in ws:
        for rec_result in rec_results:
            pred_items_rerank = reranker.rerank(
                item_purchase_cnt_map,
                rec_result["pred_items"],
                rec_result["pred_scores"],
                w,
            )
            for k in cfg.exp.ks:
                metrics.append(
                    {
                        "w": w,
                        "k": k,
                        "precision": metrics_calculator.precision_at_k(
                            rec_result["true_items"], pred_items_rerank[:k]
                        ),
                        "recall": metrics_calculator.recall_at_k(
                            rec_result["true_items"], pred_items_rerank[:k]
                        ),
                        "map": metrics_calculator.ap_at_k(
                            rec_result["true_items"], pred_items_rerank[:k]
                        ),
                        "ndcg": metrics_calculator.ndcg_at_k(
                            rec_result["true_items"], pred_items_rerank[:k]
                        ),
                    }
                )
    metrics_df = pd.DataFrame(metrics)
    metrics_df = (
        metrics_df.groupby(["w", "k"])
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            map=("map", "mean"),
            ndcg=("ndcg", "mean"),
        )
        .reset_index()
    )
    for metric_name in ["precision", "recall", "map", "ndcg"]:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        sns.lineplot(metrics_df, x="k", y=metric_name, hue="w", ax=ax)
        fig.tight_layout()
        fig.savefig(result_dir.joinpath(f"rerank_{metric_name}.png"))


if __name__ == "__main__":
    main()
