def get_sampling_scheduler(schedule: str = None,
                           step_size: int = 100,
                           decay: float = 1,
                           initial_sampling_fraction: float = 1):
    if schedule == 'step':
        return StepSamplingSchedule(step_size=step_size,
                                    decay=decay,
                                    sampling_fraction=initial_sampling_fraction)
    else:
        return None


class SamplingScheduler:
    def __init__(self, sampling_fraction: float = 1):
        self.sampling_fraction = sampling_fraction
        self._step_count = 0

    def step(self) -> float:
        raise NotImplementedError("This method needs to be implemented for each Scheduling Algorithm")


class StepSamplingSchedule(SamplingScheduler):
    def __init__(self,
                 step_size: int,
                 decay: float = 0.1,
                 sampling_fraction: float = 1):
        SamplingScheduler.__init__(self, sampling_fraction=sampling_fraction)
        self.step_size = step_size
        self.decay = decay

    def step(self) -> float:
        self._step_count += 1
        if self._step_count % self.step_size == 0:
            self.sampling_fraction *= self.decay
            print('updating sample fraction at step {} to {}'.format(self._step_count, self.sampling_fraction))
        return self.sampling_fraction


class MultiStepSamplingSchedule(SamplingScheduler):
    def __init__(self, milestones, total_steps: int, beta: float = 0.1):
        SamplingScheduler.__init__(self)
        self.milestones = list(sorted(map(lambda x: int(total_steps * x), milestones)))

    def step(self):
        pass


# Test Script
if __name__ == '__main__':
    sampling_scheduler = StepSamplingSchedule(step_size=25,
                                              beta=0.1)
    for j in range(100):
        sampling_scheduler.step()
