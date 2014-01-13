from linkpred.external import dispatch

__all__ = [
    "prediction_finished",
    "evaluation_finished",
    "dataset_finished",
    "run_finished",
]

prediction_finished = dispatch.Signal(providing_args=["scoresheet",
                                                      "dataset",
                                                      "predictor"])
evaluation_finished = dispatch.Signal(providing_args=["evaluation",
                                                      "dataset",
                                                      "predictor"])
dataset_finished = dispatch.Signal(providing_args=["dataset"])
run_finished = dispatch.Signal()
