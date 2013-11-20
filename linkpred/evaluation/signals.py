from linkpred.external import dispatch

__all__ = ["new_prediction", "new_evaluation", "datagroup_finished",
           "dataset_finished", "run_finished"]

new_prediction     = dispatch.Signal(providing_args=["prediction", "dataset",
                                                     "predictor"])
new_evaluation     = dispatch.Signal(providing_args=["evaluation", "dataset",
                                                     "predictor"])
datagroup_finished = dispatch.Signal(providing_args=["dataset", "predictor"])
dataset_finished   = dispatch.Signal(providing_args=["dataset"])
run_finished       = dispatch.Signal()
