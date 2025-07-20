from datetime import date

import yaml
from pydantic import BaseModel


class Data(BaseModel):
    articles_path: str
    customers_path: str
    transactions_path: str
    train_start_date: date
    past_start_date: date
    past_end_date: date
    train_end_date: date
    val_start_date: date
    val_end_date: date
    test_start_date: date
    test_end_date: date
    chunksize: int
    num_customer: int


class Popular(BaseModel):
    topk: int


class AgePopular(BaseModel):
    topk: int


class MostFreqCatPopular(BaseModel):
    topk: int


class Similar(BaseModel):
    min_df: int
    topk: int


class Item2Vec(BaseModel):
    topk: int


class COOC(BaseModel):
    topk: int


class Node2Vec(BaseModel):
    topk: int


class Transition(BaseModel):
    topk: int


class Candidates(BaseModel):
    popular: Popular
    age_popular: AgePopular
    most_freq_cat_popular: MostFreqCatPopular
    similar: Similar
    item2vec: Item2Vec
    cooc: COOC
    node2vec: Node2Vec
    transition: Transition
    use: list[str]


class Features(BaseModel):
    num: list[str]
    cat: list[str]


class ParamsLGBM(BaseModel):
    objective: str
    eval_metric: list[str]
    eval_at: list[int]
    max_depth: int
    learning_rate: float
    n_estimators: int
    importance_type: str
    early_stopping_round: int


class ParamsCatBoost(BaseModel):
    loss_function: str
    eval_metric: str
    iterations: int
    learning_rate: float
    depth: int
    early_stopping_round: int


class ParamsItem2Vec(BaseModel):
    vector_size: int
    window: int
    min_count: int
    workers: int
    sg: int


class ParamsCooccurence(BaseModel):
    n_components: int
    use_ppmi: bool


class ParamsNode2Vec(BaseModel):
    dimensions: int
    p: float
    q: float
    walk_length: int
    num_walks: int
    workers: int
    window: int
    min_count: int


class Params(BaseModel):
    lgb: ParamsLGBM
    cat: ParamsCatBoost
    item2vec: ParamsItem2Vec
    cooc: ParamsCooccurence
    node2vec: ParamsNode2Vec


class Model(BaseModel):
    features: Features
    params: Params


class MMRRerankerItem2Vec(BaseModel):
    w: float


class Reranker(BaseModel):
    mmr_item2vec: MMRRerankerItem2Vec


class Exp(BaseModel):
    num_rec: int
    ks: list[int]
    summary_plot_num_sample: int


class Config(BaseModel):
    seed: int
    data: Data
    candidates: Candidates
    model: Model
    reranker: Reranker
    exp: Exp

    @classmethod
    def load(cls, config_path: str):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg)
