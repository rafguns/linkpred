import os
import sys
import tempfile
from contextlib import contextmanager

import pytest
from linkpred.cli import get_config, handle_arguments, load_profile


@contextmanager
def temp_empty_file():
    fh = tempfile.NamedTemporaryFile("r", delete=False)
    yield fh.name
    assert fh.read() == ""
    fh.close()


class TestProfileFile:
    def setup(self):
        self.yaml_fd, self.yaml_fname = tempfile.mkstemp(suffix=".yaml")
        self.json_fd, self.json_fname = tempfile.mkstemp(suffix=".json")
        self.expected = {
            "predictors": [
                {"name": "CommonNeighbours", "displayname": "Common neighbours"},
                {"name": "Cosine"},
            ],
            "interpolation": True,
        }

    def teardown(self):
        for fd, fname in (
            (self.yaml_fd, self.yaml_fname),
            (self.json_fd, self.json_fname),
        ):
            os.close(fd)
            os.unlink(fname)

    def test_load_profile_yaml(self):
        with open(self.yaml_fname, "w") as fh:
            fh.write(
                """predictors:
- name: CommonNeighbours
  displayname: Common neighbours
- name: Cosine
interpolation: true"""
            )
        profile = load_profile(self.yaml_fname)
        assert profile == self.expected

    def test_load_profile_json(self):
        with open(self.json_fname, "w") as fh:
            fh.write(
                """{"predictors":
                [{"name": "CommonNeighbours",
                "displayname": "Common neighbours"},
                {"name": "Cosine"}],
                "interpolation": true}"""
            )
        profile = load_profile(self.json_fname)
        assert profile == self.expected

    def test_load_profile_error(self):
        with open(self.json_fname, "w") as fh:
            fh.write("foobar")
        with pytest.raises(Exception):
            load_profile(self.json_fname)

        with open(self.yaml_fname, "w") as fh:
            fh.write("{foobar")
        with pytest.raises(Exception):
            load_profile(self.yaml_fname)

    def test_get_config(self):
        with open(self.yaml_fname, "w") as fh:
            fh.write(
                """predictors:
- name: CommonNeighbours
  displayname: Common neighbours
- name: Cosine
interpolation: true"""
            )

        fh = tempfile.NamedTemporaryFile("r", delete=False)
        with temp_empty_file() as training:
            config = get_config([training, "-P", self.yaml_fname])
            for k, v in self.expected.items():
                assert config[k] == v

        with temp_empty_file() as training:
            config = get_config([fh.name, "-P", self.yaml_fname, "-p", "Katz", "-i"])
            # Profile gets priority
            for k, v in self.expected.items():
                assert config[k] == v
        fh.close()


def test_no_training_file():
    with pytest.raises(SystemExit):
        handle_arguments([])


def test_nonexisting_predictor():
    with pytest.raises(SystemExit):
        handle_arguments(["some-network", "-p", "Aargh"])


def test_handle_arguments():
    expected = {
        "debug": False,
        "quiet": False,
        "output": ["recall-precision"],
        "chart_filetype": "pdf",
        "interpolation": True,
        "predictors": [],
        "exclude": "old",
        "profile": None,
        "training-file": "training",
    }

    args = handle_arguments(["training"])
    for k, v in expected.items():
        assert args[k] == v

    args = handle_arguments(["training", "test"])
    for k, v in expected.items():
        assert args[k] == v
    assert args["test-file"] == "test"

    argstr = "-p CommonNeighbours Cosine -o fmax " "recall-precision -- training"
    args = handle_arguments(argstr.split())
    expected_special = {
        "predictors": ["CommonNeighbours", "Cosine"],
        "output": ["fmax", "recall-precision"],
    }
    for k, v in expected_special.items():
        assert args[k] == v

    args = handle_arguments(["training", "-i"])
    assert args["interpolation"] is False

    args = handle_arguments(["training", "-a"])
    assert args["exclude"] == ""

    args = handle_arguments(["training", "-P", "foo.json"])
    assert args["profile"] == "foo.json"

    args = handle_arguments(["training", "-f", "eps"])
    assert args["chart_filetype"] == "eps"
