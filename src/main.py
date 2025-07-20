from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.candidates_generator import CandidatesGenerator
from src.cat_ranker import CatRanker
from src.dataset import Dataset
from src.embedding_generator import EmbeddingGenerator
from src.lgb_ranker import LGBRanker
from src.metrics_calculator import MetricsCalculator
from src.schema.config import Config

plt.rcParams["font.size"] = 18
sns.set_style("whitegrid")


def main():
    cfg = Config.load(Path(__file__).parent.parent.joinpath("conf", "config.yaml"))

    current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = Path(__file__).parent.parent.joinpath("result", current_datetime)
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset = Dataset(cfg)
    metrics_calculator = MetricsCalculator(cfg)
    cand_generator = CandidatesGenerator(cfg)
    emb_generator = EmbeddingGenerator(cfg)
    lgb_ranker = LGBRanker(cfg, metrics_calculator)
    cat_ranker = CatRanker(cfg, metrics_calculator)

    all_embeddings = {}
    all_embeddings["item2vec"] = emb_generator.create_item2vec_embeddings(
        dataset.past_trans_df
    )

    cand_generator.generate_candidates(dataset, all_embeddings)
    cand_generator.evaluate_candidates(dataset, result_dir)

    lgb_ranker.preprocess(dataset, cand_generator.candidates_df, all_embeddings)
    lgb_ranker.train(result_dir)
    lgb_ranker.evaluate(result_dir)

    cat_ranker.preprocess(dataset, cand_generator.candidates_df, all_embeddings)
    cat_ranker.train(result_dir)
    cat_ranker.evaluate(result_dir)


if __name__ == "__main__":
    main()
