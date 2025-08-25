from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import time
import math
import random
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1 - DOMAIN MODELS (TRACK, CAR)
# -----------------------------
 ##지금 tracksegment 얘가 말썽임.
class TrackSegment:
    """A track segment with type, length (m), and a target cruising speed (km/h)."""
    def __init__(self, name: str, kind: str, length_m: float, target_speed_kph: float = None):
        self.name = name
        self.kind = kind
        self.length_m = length_m
        self.target_speed_kph = target_speed_kph

class OvalTrack:
    """
    Simple oval: Straight A -> Corner A -> Straight B -> Corner B.
    Total length = 3500 m.
    Straights target 200 km/h, corners target 90 km/h (tweakable).
    """
    def __init__(self):
        self.segments = [
            TrackSegment("Straight A", "straight", 1000.0, 200.0),
            TrackSegment("Corner A",   "corner",     750.0,  90.0),
            TrackSegment("Straight B", "straight",  1000.0, 200.0),
            TrackSegment("Corner B",   "corner",     750.0,  90.0),
        ]
        self.total_length_m = sum(s.length_m for s in self.segments)

    def segment_at(self, s_m: float) -> Tuple[TrackSegment, float]:
        """
        Given distance into the lap (meters), return (segment, distance_into_segment).
        """
        d = s_m % self.total_length_m
        cum = 0.0
        for seg in self.segments:
            if d < cum + seg.length_m:
                return seg, d - cum
            cum += seg.length_m
        return self.segments[-1], self.segments[-1].length_m  # fallback
    

class CarParams:
    """Car constants (approximate, not meant to be exact physics)."""
    mass_kg: float = 300.0
    max_rpm: float = 12000.0
    idle_rpm: float = 1500.0
    max_accel_mps2: float = 10.0  # 작은 기어에서 풀 쓰로틀 씨의 peak accel

    # Simple drag: F_drag = 0.5 * rho * CdA * v^2
    rho_air: float = 1.225
    CdA: float = 0.7
    c_rr: float = 0.015  # rolling resistance coeff

    # Speed bands per gear (km/h)
    gear_bands = {
        1: (0, 40),
        2: (30, 70),
        3: (60, 110),
        4: (100, 150),
        5: (140, 190),
        6: (180, 220), # not really sure about the ratios though.
    }

class CarState:
    """Mutable state of the car while simulating."""
    def __init__(self):
        # Time & kinematics
        self.t = 0.0                 # time since start [s]
        self.v_mps = 0.0             # speed [m/s]
        self.dist_total_m = 0.0      # total distance [m]
        self.dist_lap_m = 0.0        # distance within current lap [m]
        self.lap = 1                 # current lap (1에서 시작)
        self.lap_start_t = 0.0       # time when current lap started [s]

        # Powertrain
        self.throttle = 0.0          # 0~1 사이의 값
        self.gear = 1                # 1~6 (1이 가장 낮은 기어)
        self.rpm = CarParams.idle_rpm

        # Metrics
        self.max_rpm_session = 0.0
        self.avg_speed_session_kph = 0.0
        self._speed_sum = 0.0
        self._n_samples = 0



# ----------------------------
# 2- UTILITY HELPERS
# ----------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def kph_to_mps(v_kph: float) -> float:
    return v_kph / 3.6

def mps_to_kph(v_mps: float) -> float:
    return v_mps * 3.6

def drag_force(v_mps: float) -> float:
    return 0.5 * CarParams.rho_air * CarParams.CdA *v_mps * v_mps

def rolling_resistance() -> float:
    return CarParams.c_rr * CarParams.mass_kg * 9.81



# ----------------------------
# 3 - CONTROLLER + PHYSICS
# ----------------------------

class ThrottleController:
    """
    가상의 드라이버를 넣어놓은 느낌. 구간에 따라서 악셀을 밟고 말고를 정함.
    choose throttle to chase the segment's target speed.
    noise 추가해서 조금 더 자연스럽게 보이도록 함.
    """
    def __init__(self, kp: float = 0.03):
        self.kp = kp

    def decide(self, target_kph: float, current_kph: float, is_corner: bool) -> float:
        error = target_kph - current_kph
        u = self.kp * error
        if is_corner:
            u *= 0.6  # 코너에서 좀 가속을 줄이도록.
        u += random.uniform(-0.02, 0.02)  # small jitter to make it realistic
        return clamp(u, 0.0, 1.0)
    
class Gearbox:
    """
    Shift logic using speed bands with hysteresis to avoid hunting:
    - speed > (upper - margin) : upshift
    - speed < (lower + margin): downshift
    """
    def __init__(self, margin_kph: float = 5.0):
        self.margin = margin_kph

    def update_gear(self, speed_kph: float, current_gear: int) -> int:
        low, high = CarParams.gear_bands[current_gear]
        if speed_kph > high - self.margin and current_gear < max(CarParams.gear_bands):
            return current_gear + 1
        if speed_kph < low + self.margin and current_gear > 1:
            return current_gear - 1
        return current_gear

    def estimate_rpm(self, speed_kph: float, gear: int) -> float:
        low, high = CarParams.gear_bands[gear]
        if high <= low:
            return CarParams.idle_rpm
        # Map speed within [low, high] → RPM within [idle, max]
        alpha = clamp((speed_kph - low) / (high - low), 0.0, 1.0)
        rpm = CarParams.idle_rpm + alpha * (CarParams.max_rpm - CarParams.idle_rpm)
        rpm += np.random.normal(0, 80)  # touch of noise
        return clamp(rpm, CarParams.idle_rpm, CarParams.max_rpm)
    
class LongitudinalModel:
    """
    Longitudinal dynamics 좀 고려하는 게 거의 없긴 함ㅋㅋ: a = (F_engine - F_drag - F_roll) / m
    Engine force는 그냥 이렇게 간소화한 형태로: F_engine ≈ throttle * max_accel * m / gear_factor
    Higher gears --> less effective acceleration
    """
    def __init__(self):
        # Higher gear -> higher factor -> lower effective acceleration
        self.gear_factor = {1:1.0, 2:1.2, 3:1.4, 4:1.7, 5:2.0, 6:2.5}

    def step(self, v_mps: float, throttle: float, gear: int, dt: float) -> float:
        m = CarParams.mass_kg
        F_eng = throttle * CarParams.max_accel_mps2 * m / self.gear_factor[gear]
        a = (F_eng - drag_force(v_mps) - rolling_resistance()) / m
        a = clamp(a, -8.0, 8.0) # 이걸 해야 이상한 spike 안 생기고 안정적이래.
        return max(0.0, v_mps + a*dt)
    

# ----------------------------
# 4 - TELEMETRY GENERATION STEP
# ----------------------------

class TelemetrySimulator:
    """
    step()으로 시뮬레이션을 진행하면서 차량의 상태를 업데이트하고, 현재 구간에 대한 정보를 수집함.
    - 차량의 현재 속도, RPM, 기어, 스로틀, 주행
    - 현재 구간의 이름과 종류 (직선, 코너 등)
    - 현재 랩의 거리, 총 거리, 현재 랩의 시작 시간
    - 현재 랩의 시간, 세션 동안의 평균 속도, 최대 RPM 등
    """
    def __init__(self, track: OvalTrack, hz: int):
        self.track = track
        self.dt = 1.0 / hz
        self.ctrl = ThrottleController(kp=0.03)
        self.gb = Gearbox(margin_kph=5.0)
        self.dyn = LongitudinalModel()
        self.state = CarState()

        # Rolling window - 랩 타임 예상용(seconds)
        self.window_sec = 10.0
        self.time_hist: List[float] = []
        self.speed_hist_kph: List[float] = []

    def _update_metrics(self):
        s = self.state
        s._speed_sum += mps_to_kph(s.v_mps)
        s._n_samples += 1
        s.avg_speed_session_kph = s._speed_sum / max(1, s._n_samples)
        s.max_rpm_session = max(s.max_rpm_session, s.rpm)

    def _estimate_lap_time(self) -> float:
        """Elapsed + remaining / average speed over the last window_sec."""
        s = self.state
        if not self.time_hist:
            return float("nan")
        t_now = self.time_hist[-1]
        # Keep only last window second of the data
        while self.time_hist and t_now - self.time_hist[0] > self.window_sec:
            self.time_hist.pop(0)
            self.speed_hist_kph.pop(0)
        if not self.speed_hist_kph:
            return float("nan")
        avg_mps = max(0.1, float(np.mean(self.speed_hist_kph)) / 3.6)
        elapsed = s.t - s.lap_start_t
        remaining_m = self.track.total_length_m - s.dist_lap_m
        return elapsed + remaining_m / avg_mps

    def step(self):
        """Advance simulation by dt and return one telemetry sample (dict)."""
        s = self.state
        dt = self.dt #dt라는 작은 시간 단위만큼 step을 밟아가면서 시뮬레이션을 진행함.

        # 1 - which segment the car is in?? --> affects target speed
        seg, _ = self.track.segment_at(s.dist_lap_m)
        target_kph = seg.target_speed_kph

        # 2 - Imaginary driver (controller) picks throttle to chase target speed
        current_kph = mps_to_kph(s.v_mps)
        throttle = self.ctrl.decide(target_kph, current_kph, seg.kind == "corner")

        # 3 - physics: update speed
        s.v_mps = self.dyn.step(s.v_mps, throttle, s.gear, dt)

        # 4 - Time and distance updates
        s.t += dt
        ds = s.v_mps * dt
        s.dist_total_m += ds
        s.dist_lap_m += ds

        # 5 - Lap count, lap start time, and distance into lap
        if s.dist_lap_m >= self.track.total_length_m:
            s.lap += 1
            s.dist_lap_m -= self.track.total_length_m
            s.lap_start_t = s.t

        # 6 - Gear logic + RPM
        speed_kph = mps_to_kph(s.v_mps)
        s.gear = self.gb.update_gear(speed_kph, s.gear)
        s.rpm = self.gb.estimate_rpm(speed_kph, s.gear)

        # 7 - metrics update and estimation
        self.time_hist.append(s.t)
        self.speed_hist_kph.append(speed_kph)
        self._update_metrics()
        est_lap = self._estimate_lap_time()

        # 8 - Return a telemetry record
        return {
            "time_s": s.t,
            "lap": s.lap,
            "lap_time_s": s.t - s.lap_start_t,
            "distance_m": s.dist_total_m,
            "distance_into_lap_m": s.dist_lap_m,
            "segment": seg.name,
            "segment_type": seg.kind,
            "speed_kph": speed_kph,
            "rpm": s.rpm,
            "throttle": throttle,
            "gear": s.gear,
            "avg_speed_session_kph": s.avg_speed_session_kph,
            "max_rpm_session": s.max_rpm_session,
            "est_lap_time_s": est_lap,
        }



# ----------------------------
# 5 - CSV STREAMING
# ----------------------------

class CSVStreamer:
    """Writes each sample line-by-line so you never lose a session."""
    def __init__(self, path: Optional[Path]):
        self.file = None
        self.writer = None
        if path:
            self.file = open(path, "w", newline="", buffering=1)
            self.writer = csv.writer(self.file)
            self.writer.writerow([
                "time_s","lap","lap_time_s","distance_m","distance_into_lap_m","segment","segment_type",
                "speed_kph","rpm","throttle","gear","avg_speed_session_kph","max_rpm_session","est_lap_time_s"
            ])

    def write(self, sample: dict):
        if self.writer:
            self.writer.writerow([
                f"{sample['time_s']:.3f}", sample["lap"], f"{sample['lap_time_s']:.3f}",
                f"{sample['distance_m']:.2f}", f"{sample['distance_into_lap_m']:.2f}",
                sample["segment"], sample["segment_type"],
                f"{sample['speed_kph']:.2f}", int(sample["rpm"]), f"{sample['throttle']:.3f}", sample["gear"],
                f"{sample['avg_speed_session_kph']:.2f}", int(sample['max_rpm_session']),
                f"{sample['est_lap_time_s']:.2f}" if not math.isnan(sample['est_lap_time_s']) else ""
            ])

    def close(self):
        try:
            if self.file:
                self.file.flush()
                self.file.close()
        except Exception:
            pass



# ----------------------------
# 6 - VISUALIZATION (MATPLOTLIB)
# ----------------------------

class LivePlots:
    """
      - Speed vs Time
      - RPM vs Time
      - Dashboard (text: lap time, est lap, avg speed, max rpm, etc.)
    """
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax_speed = self.fig.add_subplot(3, 1, 1)
        self.ax_rpm = self.fig.add_subplot(3, 1, 2)
        self.ax_dash = self.fig.add_subplot(3, 1, 3)
        self.ax_dash.axis("off")
        self.fig.suptitle("Oval Telemetry Simulator — real-time", fontsize=12)

        (self.speed_line,) = self.ax_speed.plot([], [], label="Speed (km/h)")
        (self.rpm_line,) = self.ax_rpm.plot([], [], label="RPM")

        for ax in [self.ax_speed, self.ax_rpm]:
            ax.grid(True, linestyle="--", linewidth=0.5)
            ax.legend(loc="upper left")

        self.t_hist: List[float] = []
        self.speed_hist: List[float] = []
        self.rpm_hist: List[float] = []

    def update(self, sample: dict):
        # Append histories (show last ~60s)
        t = sample["time_s"]
        self.t_hist.append(t)
        self.speed_hist.append(sample["speed_kph"])
        self.rpm_hist.append(sample["rpm"])

        while self.t_hist and self.t_hist[-1] - self.t_hist[0] > 60.0:
            self.t_hist.pop(0); self.speed_hist.pop(0); self.rpm_hist.pop(0)

        # Update lines + limits
        self.speed_line.set_data(self.t_hist, self.speed_hist)
        self.rpm_line.set_data(self.t_hist, self.rpm_hist)

        if self.t_hist:
            tmin, tmax = self.t_hist[0], self.t_hist[-1]
            for ax in [self.ax_speed, self.ax_rpm]:
                ax.set_xlim(max(0, tmax - 60), tmax + 1)
        self.ax_speed.set_ylim(0, max(120, (max(self.speed_hist) if self.speed_hist else 0) * 1.2))
        self.ax_rpm.set_ylim(0, CarParams.max_rpm * 1.05)

        # Dashboard text
        dash = (
            f"Lap: {sample['lap']}   |   Segment: {sample['segment']} ({sample['segment_type']})\n"
            f"Speed: {sample['speed_kph']:.1f} km/h    RPM: {int(sample['rpm']):5d}    "
            f"Throttle: {sample['throttle']*100:5.1f}%    Gear: {sample['gear']}\n"
            f"Distance (session): {sample['distance_m']/1000:.2f} km    "
            f"Distance into lap: {sample['distance_into_lap_m']:.1f} m\n"
            f"Lap time (elapsed): {sample['lap_time_s']:.2f} s    "
            f"Estimated lap time: {sample['est_lap_time_s']:.2f} s\n"
            f"Average speed (session): {sample['avg_speed_session_kph']:.1f} km/h    "
            f"Max RPM (session): {int(sample['max_rpm_session']):5d}"
        )
        self.ax_dash.clear()
        self.ax_dash.axis("off")
        self.ax_dash.text(0.01, 0.95, dash, va="top", ha="left", fontsize=11, family="monospace")

        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.pause(0.001)



# ----------------------------
# 7 - ORCHESTRATION
# ----------------------------

def run(hz: int, csv_path: Optional[Path]):
    track = OvalTrack()
    sim = TelemetrySimulator(track, hz=hz)
    csv_stream = CSVStreamer(csv_path)
    plots = LivePlots()

    # physics rate랑 연동되는 시계를 옆에다가 그냥 붙임..
    last_tick = time.perf_counter()
    try:
        while True:
            sample = sim.step()
            plots.update(sample)
            csv_stream.write(sample)

            # 터미널에 출력할거
            print(
                f"\rt={sample['time_s']:6.2f}s | lap={sample['lap']:2d} | seg={sample['segment']:<10} | "
                f"v={sample['speed_kph']:6.1f} km/h | rpm={int(sample['rpm']):5d} | thr={sample['throttle']*100:5.1f}% | "
                f"gear={sample['gear']} | avg={sample['avg_speed_session_kph']:6.1f} km/h | est_lap={sample['est_lap_time_s']:6.2f}s ",
                end="", flush=True
            )

            # Sleep to maintain real-time cadence
            dt = 1.0 / hz
            now = time.perf_counter()
            time.sleep(max(0.0, dt - (now - last_tick)))
            last_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        csv_stream.close()
        plt.ioff()
        try:
            plt.show(block=False)
        except Exception:
            pass


# ----------------------------
# 8 - CLI
# ----------------------------

def main():
    p = argparse.ArgumentParser(description="Oval Telemetry Simulator")
    p.add_argument("--hz", type=int, default=20, help="Sampling rate in Hz(per second, default 20)")
    p.add_argument("--csv", type=str, default="telemetry.csv", help="CSV output path (default telemetry.csv)")
    args = p.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    run(hz=args.hz, csv_path=csv_path)

if __name__ == "__main__":
    main()