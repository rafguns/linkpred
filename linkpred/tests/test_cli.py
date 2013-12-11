from nose.tools import assert_equal, assert_dict_equal, raises, assert_raises
import nose

import os
import tempfile
from linkpred.cli import load_profile, get_config, handle_arguments
from contextlib import contextmanager


@contextmanager
def temp_empty_file():
    fh = tempfile.NamedTemporaryFile("r", delete=False)
    yield fh.name
    assert_equal(fh.read(), "")
    fh.close()


class TestProfileFile:
    def setup(self):
        self.yaml_fd, self.yaml_fname = tempfile.mkstemp(suffix=".yaml")
        self.json_fd, self.json_fname = tempfile.mkstemp(suffix=".json")
        self.expected = {
            "predictors": [
                {"name": "CommonNeighbours",
                 "displayname": "Common neighbours"},
                {"name": "Cosine"},
            ],
            "steps": 5,
            "interpolation": True
        }

    def teardown(self):
        for fd, fname in ((self.yaml_fd, self.yaml_fname),
                          (self.json_fd, self.json_fname)):
            os.close(fd)
            os.unlink(fname)

    def test_load_profile_yaml(self):
        with open(self.yaml_fname, "w") as fh:
            fh.write("""predictors:
- name: CommonNeighbours
  displayname: Common neighbours
- name: Cosine
steps: 5
interpolation: true""")
        profile = load_profile(self.yaml_fname)
        assert_dict_equal(profile, self.expected)

    def test_load_profile_json(self):
        with open(self.json_fname, "w") as fh:
            fh.write("""{"predictors":
    [{"name": "CommonNeighbours",
      "displayname": "Common neighbours"},
     {"name": "Cosine"}],
    "steps": 5,
    "interpolation": true}""")
        profile = load_profile(self.json_fname)
        assert_dict_equal(profile, self.expected)

    def test_load_profile_error(self):
        with open(self.json_fname, "w") as fh:
            fh.write("foobar")
        with assert_raises(Exception):
            load_profile(self.json_fname)

        with open(self.yaml_fname, "w") as fh:
            fh.write("{foobar")
        with assert_raises(Exception):
            print load_profile(self.yaml_fname)

    def test_get_config(self):
        with open(self.yaml_fname, "w") as fh:
            fh.write("""predictors:
- name: CommonNeighbours
  displayname: Common neighbours
- name: Cosine
steps: 5
interpolation: true""")

        fh = tempfile.NamedTemporaryFile("r", delete=False)
        with temp_empty_file() as training:
            config = get_config([training, "-P", self.yaml_fname])
            for k, v in self.expected.iteritems():
                assert_equal(config[k], v)

        with temp_empty_file() as training:
            config = get_config([fh.name, "-P", self.yaml_fname, "-p",
                                 "Katz", "-i"])
            # Profile gets priority
            for k, v in self.expected.iteritems():
                assert_equal(config[k], v)
        fh.close()


@raises(SystemExit)
def test_no_training_file():
    # Skip because nose always displays stderr output :-/
    raise nose.SkipTest()
    handle_arguments([])


@raises(SystemExit)
def test_nonexisting_predictor():
    # Skip because nose always displays stderr output :-/
    raise nose.SkipTest()
    fh = tempfile.NamedTemporaryFile("r", delete=False)
    fh.close()
    handle_arguments([fh.name, "-p", "Aargh"])


def test_handle_arguments():
    expected = {
        "debug": False,
        "quiet": False,
        "output": ["recall-precision"],
        "chart_filetype": "pdf",
        "interpolation": True,
        "predictors": [],
        "only_new": True,
        "profile": None,
    }

    with temp_empty_file() as training:
        args = handle_arguments([training])
        for k, v in expected.iteritems():
            assert_equal(args[k], v)
        assert_equal(args["test-file"], None)

    with temp_empty_file() as training, temp_empty_file() as test:
        args = handle_arguments([training, test])
        for k, v in expected.iteritems():
            assert_equal(args[k], v)

    argstr = "-p CommonNeighbours Cosine -o fmax " \
        "recall-precision -- %s" % training
    args = handle_arguments(argstr.split())
    expected_special = {"predictors": ["CommonNeighbours", "Cosine"],
                        "output": ["fmax", "recall-precision"]}
    for k, v in expected_special.iteritems():
        assert_equal(args[k], v)

    args = handle_arguments([training, "-i"])
    assert_equal(args["interpolation"], False)

    args = handle_arguments([training, "-a"])
    assert_equal(args["only_new"], False)

    args = handle_arguments([training, "-P", "foo.json"])
    assert_equal(args["profile"], "foo.json")

    args = handle_arguments([training, "-f", "eps"])
    assert_equal(args["chart_filetype"], "eps")
