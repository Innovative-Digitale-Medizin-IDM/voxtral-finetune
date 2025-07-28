from audiomentations import TimeStretch
import numpy as np

class LengthAwareTimeStretch(TimeStretch):
    """
    A TimeStretch that never stretches a clip past `max_duration_sec`.
    If the randomly chosen rate would make the clip longer than the limit
    we either (a) skip the transform altogether, or (b) pick a legal rate.
    """
    supports_multichannel = True

    def __init__(
        self,
        max_duration_sec: float = 30.0,
        min_rate: float = 0.5,
        max_rate: float = 2.0,
        skip_if_too_long: bool = True,
        p: float = 1.0,
    ):
        super().__init__(min_rate=min_rate, max_rate=max_rate, p=p)
        self.max_duration_sec = max_duration_sec
        self.skip_if_too_long = skip_if_too_long

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)

        if not self.parameters["should_apply"]:
            return

        duration = len(samples) / sample_rate
        rate     = self.parameters["rate"]      
        new_len  = duration / rate         

        if new_len <= self.max_duration_sec:
            return

        if self.skip_if_too_long:
            self.parameters["should_apply"] = False
        else:
            min_allowed_rate = max(duration / self.max_duration_sec, 1.0)
            self.parameters["rate"] = np.random.uniform(
                min_allowed_rate, self.max_rate
            )
            
