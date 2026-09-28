"""Run the pipeline up to one module and show that module's debug overlay.

    python tools/run_module.py --until identity --source data/clips/crossing.mp4

Same flags as ipcvgame.app.main; --until is required here.
"""

import sys

from ipcvgame.app.main import main

if __name__ == "__main__":
    if "--until" not in sys.argv:
        sys.exit("usage: run_module.py --until {pose,identity,motion,face,gesture,game} [...]")
    main(sys.argv[1:])
