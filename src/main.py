from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.candidates_generator import CandidatesGenerator
from src.dataset import Dataset
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
    ranker = LGBRanker(cfg, metrics_calculator)

    cand_generator.generate_candidates(dataset)
    cand_generator.evaluate_candidates(dataset, result_dir)

    ranker.preprocess(dataset, cand_generator.candidates_df)
    ranker.train(result_dir)
    ranker.evaluate(result_dir)


if __name__ == "__main__":
    main()
