SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize",
    },
    "parameters": {
        "learning_rate": {
            "values": [0.001, 0.0001, 0.00001],
        },
        "batch_size": {
            "values": [16, 32],
        },
        "dropout": {
            "values": [0.2, 0.3, 0.5],
        },
        "dense_units": {
            "values": [128, 256],
        },
        "epochs": {
            "value": 10,
        },
    },
}
