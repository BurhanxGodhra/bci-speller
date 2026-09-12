import pylsl


class MarkerOutlet:
    def __init__(self, name="CalibrationMarkers"):
        info = pylsl.StreamInfo(name=name, type="Markers", channel_count=1,
                                 nominal_srate=0, channel_format="string", source_id=name)
        self.outlet = pylsl.StreamOutlet(info)

    def push(self, label: str) -> float:
        t = pylsl.local_clock()
        self.outlet.push_sample([label], t)
        return t
