"""ECG Heart Rate & HRV Analyzer.

Pipeline: load/generate ECG -> bandpass filter -> R-peak detection
-> heart rate -> HRV metrics (time domain) -> plots.
"""
import argparse

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def generate_synthetic_ecg(duration=30, fs=250, heart_rate=72, noise=0.05, seed=42):
    """Create a simple synthetic ECG (sum of Gaussians per beat) with noise and baseline wander."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / fs)
    ecg = np.zeros_like(t)

    beat_times, current = [], 0.5
    while current < duration:
        beat_times.append(current)
        # Natural beat-to-beat variability (~3%)
        current += (60 / heart_rate) * (1 + rng.normal(0, 0.03))

    # (offset in s, amplitude, width in s) for the P, Q, R, S, T waves
    waves = [(-0.20, 0.15, 0.025), (-0.04, -0.12, 0.010),
             (0.00, 1.00, 0.012), (0.04, -0.20, 0.010), (0.25, 0.30, 0.045)]
    for bt in beat_times:
        for off, amp, width in waves:
            ecg += amp * np.exp(-((t - bt - off) ** 2) / (2 * width ** 2))

    ecg += 0.1 * np.sin(2 * np.pi * 0.25 * t)       # baseline wander (breathing)
    ecg += rng.normal(0, noise, size=t.shape)        # sensor noise
    return t, ecg, np.array(beat_times)


def bandpass_filter(signal, fs, low=0.5, high=40.0, order=4):
    """Zero-phase Butterworth bandpass filter to remove baseline wander and high-freq noise."""
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def detect_r_peaks(ecg, fs):
    """Detect R-peaks using squared signal + adaptive height + refractory period."""
    squared = ecg ** 2
    threshold = 0.3 * np.max(squared)
    peaks, _ = find_peaks(squared, height=threshold, distance=int(0.25 * fs))
    return peaks


def compute_hrv(r_peaks, fs):
    """Return heart rate and time-domain HRV metrics from R-peak indices."""
    rr = np.diff(r_peaks) / fs * 1000  # RR intervals in ms
    if len(rr) < 2:
        raise ValueError("Not enough beats detected to compute HRV.")
    diff_rr = np.diff(rr)
    return {
        "mean_hr_bpm": 60000 / np.mean(rr),
        "mean_rr_ms": np.mean(rr),
        "sdnn_ms": np.std(rr, ddof=1),
        "rmssd_ms": np.sqrt(np.mean(diff_rr ** 2)),
        "pnn50_percent": 100 * np.mean(np.abs(diff_rr) > 50),
    }


def plot_results(t, raw, filtered, r_peaks, output="ecg_result.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, raw, lw=0.8)
    axes[0].set_title("Raw ECG (noise + baseline wander)")
    axes[1].plot(t, filtered, lw=0.8)
    axes[1].plot(t[r_peaks], filtered[r_peaks], "rv", label="R-peaks")
    axes[1].set_title("Filtered ECG with detected R-peaks")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    for ax in axes:
        ax.set_ylabel("Amplitude (a.u.)")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    return output


def main():
    p = argparse.ArgumentParser(description="ECG heart rate & HRV analyzer")
    p.add_argument("--csv", help="CSV file with one ECG column (optional; synthetic if omitted)")
    p.add_argument("--fs", type=int, default=250, help="Sampling rate in Hz")
    p.add_argument("--duration", type=int, default=30, help="Synthetic ECG duration (s)")
    p.add_argument("--hr", type=int, default=72, help="Synthetic heart rate (bpm)")
    p.add_argument("--output", default="ecg_result.png", help="Output plot filename")
    args = p.parse_args()

    if args.csv:
        raw = np.loadtxt(args.csv, delimiter=",", skiprows=1, usecols=0)
        t = np.arange(len(raw)) / args.fs
    else:
        t, raw, _ = generate_synthetic_ecg(args.duration, args.fs, args.hr)

    filtered = bandpass_filter(raw, args.fs)
    r_peaks = detect_r_peaks(filtered, args.fs)
    metrics = compute_hrv(r_peaks, args.fs)

    print(f"Beats detected : {len(r_peaks)}")
    print(f"Mean HR        : {metrics['mean_hr_bpm']:.1f} bpm")
    print(f"Mean RR        : {metrics['mean_rr_ms']:.1f} ms")
    print(f"SDNN           : {metrics['sdnn_ms']:.1f} ms")
    print(f"RMSSD          : {metrics['rmssd_ms']:.1f} ms")
    print(f"pNN50          : {metrics['pnn50_percent']:.1f} %")
    print(f"Plot saved to  : {plot_results(t, raw, filtered, r_peaks, args.output)}")


if __name__ == "__main__":
    main()
