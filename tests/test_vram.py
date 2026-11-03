from amdlora.monitor.vram import GpuTimer, peak_vram_gib, vram_summary


def test_timer_measures_elapsed_time():
    with GpuTimer() as timer:
        sum(range(200_000))
    assert timer.seconds > 0


def test_vram_helpers_return_sane_values_with_or_without_gpu():
    assert peak_vram_gib() >= 0
    assert isinstance(vram_summary(), str)