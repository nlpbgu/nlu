
# Define global parameters
params = {
    "aggregator": ["mean", "median"],
    "EXTRACTOR_TYPE": ["decoupled"],
    "EMBEDDING_TYPE": ["uncontextualized", "elmo", "bert"],
    "ENCODER_TYPE": ["bilstmmax", "bertconcat"],
    "LOSS_TYPE": ["pointwise-l2", "pointwise-log", "pairwise-hinge"],
    "ENCODER_DIM": [1024],
    "MARGIN": [0.3, 0.35, 0.4, 0.6, 0.7, 0.8],
    "CRR_WEIGHT": [0.0837, 0.0419, 0.00837, 0.00419],
    "CROSS_SAMPLE": [1, 2]
}
