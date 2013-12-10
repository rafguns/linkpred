"""CLI handling"""
import argparse
import json

from .exceptions import LinkPredError
from .predictors import all_predictors
from .util import log

__all__ = ["load_profile", "get_profile", "handle_arguments"]


def load_profile(fname):
    """Load the JSON or YAML profile with the given filename"""
    try:
        with open(fname) as f:
            if fname.endswith(".yaml"):
                import yaml
                return yaml.safe_load(f)
            else:
                return json.load(f)
    except Exception as e:
        raise LinkPredError("Encountered error while loading profile '%s'. "
                            "Error message: '%s'" % (fname, e))


def get_profile(args=None):
    """Load profile based on command-line arguments

    If a YAML-or JSON-based profile file is supplied, any settinsg therein take
    prioirity over command-line arguments.

    """
    args = handle_arguments(args)

    profilename = args.pop('profile')

    profile = {}
    predictorlist = [{'name': predictor} for predictor
                     in args.pop('predictors')]
    profile['predictors'] = predictorlist
    profile.update(args)

    if profilename:
        profile.update(load_profile(profilename))

    return profile


def handle_arguments(args=None):
    """Get nice CLI interface and return arguments."""

    parser = argparse.ArgumentParser(
        version="0.1", description="Easy link prediction tool",
        usage="%(prog)s training-file [test-file] [options]")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--debug", action="store_true", dest="debug",
                       default=False, help="Show debug messages")
    group.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                       default=False, help="Don't show info messages")

    # TODO allow case-insensitive match
    output_types = ["recall-precision", "f-score", "roc", "cache-predictions",
                    "cache-evaluations", "fmax"]
    output_help = "Type of output(s) to produce (default: recall-precision). "\
                  "Allowed values are: " + ", ".join(output_types)
    parser.add_argument("-o", "--output", help=output_help, nargs="*",
                        choices=output_types, default=output_types[0:1],
                        metavar="OUTPUT")

    # TODO allow case-insensitive match
    parser.add_argument("-f", "--chart-filetype", default="pdf",
                        help="File type for charts (default: %(default)s)")

    parser.add_argument("-i", "--no-interpolation", dest="interpolation",
                        help="Do not interpolate recall-precision charts",
                        action="store_false", default=True)

    # TODO allow case-insensitive match
    predictors = [p.__name__ for p in all_predictors()]
    predictor_help = "Predictor(s) to use for link prediction. "\
                     "Allowed values are: " + ", ".join(predictors)
    parser.add_argument("-p", "--predictors", nargs="*", choices=predictors,
                        default=[], help=predictor_help, metavar="PREDICTOR")

    all_help = "Predict all links, "\
               "including ones present in the training network"
    parser.add_argument("-a", "--all", action="store_false", dest="only_new",
                        default=True, help=all_help)

    parser.add_argument("-P", "--profile", help="JSON/YAML profile file")

    parser.add_argument("training-file", help="File with the training network",
                        type=argparse.FileType())
    parser.add_argument("test-file", nargs="?", type=argparse.FileType(),
                        help="File with the test network")

    results = parser.parse_args(args)

    if results.debug:
        log.logger.setLevel(log.logging.DEBUG)
    elif results.quiet:
        log.logger.setLevel(log.logging.WARNING)
    else:
        log.logger.setLevel(log.logging.INFO)

    # Return as plain dictionary
    return vars(results)
