from .static import StaticEvaluation


def ranked_evaluation(retrieved, relevant, n=None, **kwargs):
    """Generator for ranked evaluation of IR

    Arguments
    ---------
    retrieved : a Scoresheet
        score sheet of ranked retrieved results

    relevant : a set
        set of relevant results

    n : an integer
        At each step, the next n items on the retrieved score sheet are
        added to the set of retrieved items that are compared to the relevant
        ones.

    """
    evaluation = StaticEvaluation(relevant=relevant, **kwargs)
    for ret in retrieved.successive_sets(n=n):
        evaluation.update_retrieved(ret)
        yield evaluation
