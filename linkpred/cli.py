import argparse
import json

from .predictors import all_predictors
from .util import log

__all__ = ["load_profile", "get_profile", "handle_arguments"]


class Profile(object):
    pass


def load_profile(fname):
    profile = {}
    try:
        with open(fname) as f:
            if fname.endswith(".yaml"):
                import yaml
                profile = yaml.safe_load(f)
            else:
                profile = json.load(f)
    except (AttributeError, TypeError) as e:
        log.logger.error("Encountered error '%s'" % e)
    finally:
        return profile


def get_profile():
    args = handle_arguments()
    try:
        profile = load_profile(args.profile)
    except AttributeError:
        profile = {}

    for k, v in args.iteritems():
        # Handle predictors separately
        if k == "predictors":
            profile[k] = []
            for predictor in v:
                profile[k].append({"name": predictor})
        else:
            profile[k] = v

    return profile


def handle_arguments():
    """Get nice CLI interface and return arguments."""

    parser = argparse.ArgumentParser(
        version="0.1", description="Easy link prediction tool",
        usage="%(prog)s [options] training [test]")

    parser.add_argument("--debug", action="store_true", dest="debug",
                        default=False, help="Show debug messages")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Don't show info messages")

    # TODO allow case-insensitive match
    output_types = ["recall-precision", "F-score", "ROC"]
    output_help = "Type of output(s) to produce (default: %(default)s)"
    parser.add_argument("-o", "--output", help=output_help, action="append",
                        choices=output_types, default=output_types[0:1])

    # TODO allow case-insensitive match
    parser.add_argument("-f", "--chart-filetype", default="pdf",
                        help="File type for charts (default: %(default)s)")

    parser.add_argument("-i", "--interpolation",
                        help="Interpolate recall-precision charts",
                        action="store_true", default=False)

    # TODO allow case-insensitive match
    predictors = [p.__name__ for p in all_predictors()]
    parser.add_argument("-p", "--predictor", action="append",
                        dest="predictors", choices=predictors, default=[],
                        help="Predictor(s) to use for link prediction")

    all_help = "Predict all links, "\
               "including ones present in the training network"
    parser.add_argument("-a", "--all", action="store_false", dest="only_new",
                        default=True, help=all_help)

    parser.add_argument("-P", "--profile", help="JSON/YAML profile file")

    parser.add_argument("training")
    parser.add_argument("test", nargs="?")

    results = parser.parse_args()

    if results.debug:
        log.logger.setLevel(log.logging.DEBUG)
    elif results.quiet:
        log.logger.setLevel(log.logging.WARNING)
    else:
        log.logger.setLevel(log.logging.INFO)

    return vars(results)
