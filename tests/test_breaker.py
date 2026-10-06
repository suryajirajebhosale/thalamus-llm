from thalamus import BreakerState, CircuitBreaker


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make(clock):
    return CircuitBreaker("t", failure_threshold=0.5, window_size=4, min_calls=4, open_timeout=10, clock=clock)


def test_opens_once_failure_rate_crosses_threshold():
    breaker = make(FakeClock())
    breaker.record_success()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED  # below min_calls
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert breaker.allow() is False


def test_half_open_allows_exactly_one_probe_then_closes_on_success():
    clock = FakeClock()
    breaker = make(clock)
    for _ in range(4):
        breaker.record_failure()
    clock.now = 10
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.allow() is True
    assert breaker.allow() is False  # probe already in flight
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED
    assert breaker.status()["calls_in_window"] == 0


def test_failed_probe_reopens():
    clock = FakeClock()
    breaker = make(clock)
    for _ in range(4):
        breaker.record_failure()
    clock.now = 10
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    clock.now = 15
    assert breaker.allow() is False
