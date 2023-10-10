def _add_autotsad_to_path():
    import os
    import sys

    # allow importing autotsad code
    wdir = os.getcwd()
    if wdir not in sys.path:
        sys.path.append(wdir)


_add_autotsad_to_path()
