"""CLI handling"""
import argparse
import json
import logging
import sys

from .exceptions import LinkPredError
from .predictors import all_predictors

log = logging.getLogger('linkpred')

__all__ = ["load_profile", "get_config", "handle_arguments"]


def setup_logger():
    streamhandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                  "%H:%M:%S")
    streamhandler.setFormatter(formatter)
    log.addHandler(streamhandler)


def load_profile(fname):
    """Load the JSON or YAML profile with the given filename"""
    try:
        with open(fname) as f:
            if fname.endswith(".yaml") or fname.endswith(".yml"):
                import yaml
                return yaml.safe_load(f)
            else:
                return json.load(f)
    except Exception as e:
        raise LinkPredError("Encountered error while loading profile '%s'. "
                            "Error message: '%s'" % (fname, e))


def get_config(args=None):
    """Get configuration as supplied by the user

    If a YAML-or JSON-based profile is supplied, any settinsg therein take
    priority over command-line arguments.

    """
    args = handle_arguments(args)

    profile = args.pop('profile')

    config = {}
    predictorlist = [{'name': predictor} for predictor
                     in args.pop('predictors')]
    config['predictors'] = predictorlist
    config.update(args)

    if profile:
        config.update(load_profile(profile))

    return config


def handle_arguments(args=None):
    """Get nice CLI interface and return arguments."""

    parser = argparse.ArgumentParser(
        description="Easy link prediction tool",
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
    parser.add_argument("-a", "--all", action="store_const", dest="exclude",
                        const="", default="old", help=all_help)

    parser.add_argument("-P", "--profile", help="JSON/YAML profile file")

    parser.add_argument("training-file", help="File with the training network")
    parser.add_argument("test-file", nargs="?",
                        help="File with the test network")

    results = parser.parse_args(args)

    if results.debug:
        log.setLevel(logging.DEBUG)
    elif results.quiet:
        log.setLevel(logging.WARNING)
    else:
        log.setLevel(logging.INFO)

    # Return as plain dictionary
    return vars(results)
