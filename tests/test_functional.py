# Should be at start of file
import matplotlib

matplotlib.use("Agg")

import os
from linkpred.cli import main


def test_simple_run():
    # TODO do this test in a temp directory and clean up afterwards
    num_files = len(os.listdir("examples"))
    main(
        "examples/inf1990-2004.net examples/inf2005-2009.net"
        " -p CommonNeighbours --quiet".split()
    )
    assert len(os.listdir("examples")) == num_files + 1
