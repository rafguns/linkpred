* Work with other Python versions than 2.7
* Implement Dice predictor
* Add waf script, makefile or similar for common operations:

        pyinstaller linkpred.spec
        git archive --format=tar.gz --prefix=linkpred/ HEAD > linkpred.tar.gz
        ...

* Simplify evaluation, to work more like `sklearn.metrics` (plain functions)
* Figure out if we need the `new_prediction` and `new_evaluation` signals. Can't we just pass around the `Scoresheet` and a list of `(tp, fp, fn, tn)` tuples (or a `numpy.matrix`)?
