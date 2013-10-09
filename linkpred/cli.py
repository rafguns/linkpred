import json
from optparse import OptionParser

from .predictors import all_predictors
from .util import log

__all__ = ["load_profile", "get_profile", "get_profile_by_options",
           "options_n_args"]


def data_from_profile(fname):
    data = {}
    try:
        with open(fname) as f:
            if fname.endswith(".yaml"):
                import yaml
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
    except (AttributeError, TypeError) as e:
        log.logger.warning("Encountered error '%s'" % e)
    finally:
        return data


def fancy_update(base, new):
    updated = dict(base.iteritems())
    for k, v in new.iteritems():
        if k not in base:
            updated[k] = v
        elif type(base[k]) == type(v) == dict:
            updated[k] = fancy_update(base[k], v)
        elif type(base[k]) == type(v) == list:
            updated[k].extend(v)
        else:
            updated[k] = v
    return updated


def load_profile(*fnames):
    """
    Load profile from one or more files

    Arguments
    ---------
    fnames : one or more strings (file names)

    """
    profile = {}
    for fname in fnames:
        data = data_from_profile(fname)
        profile = fancy_update(profile, data)
    return profile


def get_profile(**kwargs):
    options, args = options_n_args(**kwargs)
    if args:
        log.logger.warning("Ignoring arguments: %s" % str(args))
    return get_profile_by_options(options)


def get_profile_by_options(options):
    """Determine a profile based on available options

    If multiple profiles are passed through the CLI interface,
    they are merged into one. In case of conflicts, the last profile
    supersedes the previous ones.
    Other CLI options supersede the profiles.

    Arguments
    ---------

    options : an optparse.Options object

    Returns
    -------

    profile : a dict

    """
    profile = load_profile(*options.profile)

    option_names = ["charts", "filetype", "interpolation", "steps", "only_new"]
    for option_name in option_names:
        try:
            option = getattr(options, option_name)
        except AttributeError:
            continue
        profile[option_name] = option

    if hasattr(options, 'predictors') and options.predictors:
        profile['predictors'] = []
        for p in options.predictors:
            profile['predictors'].append({'name': p})

    return profile


def options_n_args(choose_chart=True, choose_profile=True,
                   choose_predictor=True, choose_filetype=False,
                   choose_weight=False, choose_interpolation=False):
    """Get nice CLI interface and return options 'n arguments."""

    parser = OptionParser()
    parser.add_option("--debug", action="store_true", dest="debug",
                      default=False, help="Log debug messages")
    if choose_chart:
        chart_help = "Type of chart(s) to produce (default: all available)."
        chart_types = ["recall-precision", "F-score", "ROC"]
        parser.add_option("-c", "--chart", help=chart_help, action="append",
                          choices=chart_types, dest="charts", default=chart_types)
    if choose_filetype:
        parser.add_option("-f", "--filetype",
                          help="Output file type (default: %default)", default="pdf")
    if choose_interpolation:
        parser.add_option("-i", "--no-interpolation",
                          help="Do not interpolate precision", action="store_false",
                          dest="interpolation", default=True)
    if choose_predictor:
        predictors = [p.__name__ for p in all_predictors()]
        parser.add_option(
            "-p", "--predictors", action="append", dest="predictors",
            help="Predicting methods to use (default: all available)",
            choices=predictors, default=[])
        parser.add_option(
            "-n", "--only-new", action="store_true", dest="only_new",
            default=False,
            help="Only consider new (unattested) predictions")
    if choose_profile:
        parser.add_option("-P", "--profile", action="append",
                          help="JSON profile file", default=[])

    options, args = parser.parse_args()
    if options.debug:
        log.logger.setLevel(log.logging.DEBUG)
    else:
        log.logger.setLevel(log.logging.INFO)

    return options, args
