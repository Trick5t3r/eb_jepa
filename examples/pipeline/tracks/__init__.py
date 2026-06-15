"""Importing this package registers every concrete track.

Add one line per new track you create (after copying template_track.py):
    from examples.pipeline.tracks import my_track   # noqa: F401
"""
from examples.pipeline.tracks import synthetic_track  # noqa: F401
# from examples.pipeline.tracks import my_track       # TODO: register your track here
