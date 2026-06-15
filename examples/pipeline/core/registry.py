"""Track registry. A track registers itself with @register_track("name"); the CLI
resolves --track <name> through here. Importing examples.pipeline.tracks triggers
registration of every track module."""
_TRACKS = {}


def register_track(name):
    def deco(cls):
        _TRACKS[name] = cls
        cls.name = name
        return cls
    return deco


def get_track(name):
    import examples.pipeline.tracks  # noqa: F401  (populates the registry)
    if name not in _TRACKS:
        raise KeyError(f"unknown track {name!r}; available: {sorted(_TRACKS)}")
    return _TRACKS[name]()


def list_tracks():
    import examples.pipeline.tracks  # noqa: F401
    return sorted(_TRACKS)
