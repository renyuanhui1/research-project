"""Measure the existing RGB-D subscription and recorder on the WSL gateway."""
import argparse
from bisect import bisect_left
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

from projectairsim import Drone, ProjectAirSimClient, World

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from common.runtime import configure_client_log
from control import run_approach as controller

parser = argparse.ArgumentParser()
parser.add_argument('--address', required=True)
parser.add_argument('--duration', type=float, default=35)
parser.add_argument('--out-dir', type=Path, required=True, help='New directory for this measurement')
parser.add_argument('--scene', type=Path, default=Path(__file__).resolve().parent / 'transport/runtime-scene.json')
args = parser.parse_args()
out = args.out_dir.resolve()
out.mkdir()
configure_client_log(out)
controller.OUT = out
source = args.scene.resolve()
scene = json.loads(source.read_text())
(out / 'runtime-scene.json').write_bytes(source.read_bytes())
recorders = {tag: controller.Recorder(tag) for tag in ('front', 'chase')}
rows = defaultdict(list)
clock = []
report = dict(address=args.address, status='running', source=str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              duration_requested_s=args.duration, motion_commands=0)
client = ProjectAirSimClient(address=args.address)
world = World.__new__(World)
world.client = client
world.sim_config_path = str(source.parent)
world.sim_instance_idx = -1


def callback(tag):
    def receive(_, message):
        rows[tag].append((time.monotonic(), int(message['time_stamp'])))
        if tag in recorders:
            recorders[tag].push(message)
    return receive


def summarize(samples):
    arrivals = [arrival for arrival, _ in samples]
    return dict(received=len(samples), source_fps=len({ts for _, ts in samples}) / (end - start),
                max_arrival_gap_s=max(b - a for a, b in zip([start] + arrivals, arrivals + [end])))


def rgbd_summary():
    depths = sorted({ts for _, ts in rows['depth']})
    skews = []
    for _, timestamp in rows['front']:
        index = bisect_left(depths, timestamp)
        near = depths[max(0, index - 1):index + 1]
        if near:
            skews.append(min(abs(timestamp - value) for value in near) / 1e6)
    paired_ages = []
    missed = 0
    used = -1
    for observed, sim_time in clock:
        available_depths = [ts for arrival, ts in rows['depth'] if arrival <= observed][-20:]
        available_front = [ts for arrival, ts in rows['front'] if arrival <= observed][-20:]
        candidate = next((ts for ts in reversed(available_front) if ts > used and
                          any(abs(ts - depth) <= 250_000_000 for depth in available_depths)), None)
        if candidate is None:
            missed += 1
        else:
            used = candidate
            paired_ages.append((sim_time - candidate) / 1e9)
    return dict(front_frames=len(rows['front']), time_support_ms=[50, 250],
                nearest_depth_matches=[sum(skew <= bound for skew in skews) for bound in (50, 250)],
                check_ticks=len(clock), new_pairs=len(paired_ages), ticks_without_new_pair=missed,
                age_median_s=statistics.median(paired_ages) if paired_ages else None,
                age_max_s=max(paired_ages, default=None))


with tempfile.TemporaryDirectory() as temporary:
    stop = Path(temporary) / 'stop'
    monitor_script = ROOT / 'debug/camera/20261111-window-state/window_monitor.ps1'

    def quoted_windows(path):
        return subprocess.check_output(['wslpath', '-w', str(path)], text=True).strip().replace("'", "''")

    ps = (f"$p=Get-Process -Name UnrealEditor | Where-Object MainWindowHandle -ne 0 | Select-Object -First 1; "
          f"& ([ScriptBlock]::Create([IO.File]::ReadAllText('{quoted_windows(monitor_script)}'))) "
          f"-OutputPath '{quoted_windows(out / 'window-state.jsonl')}' "
          f"-StopPath '{quoted_windows(stop)}' -TargetProcessId $p.Id")
    monitor = subprocess.Popen(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', ps],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    start = end = None
    try:
        client.connect()
        client.socket_services.recv_timeout = 12000
        client.socket_topics.recv_buffer_size = 32
        world.load_scene(scene, delay_after_load_sec=3)
        drone = Drone(client, world, 'Drone1')
        for tag, camera, topic in (('front', 'FrontCamera', 'scene_camera'),
                                  ('depth', 'FrontCamera', 'depth_camera'),
                                  ('chase', 'Chase', 'scene_camera')):
            client.subscribe(drone.sensors[camera][topic], callback(tag))
        deadline = time.monotonic() + 15
        while not all(rows[tag] for tag in ('front', 'depth', 'chase')):
            if time.monotonic() >= deadline:
                raise RuntimeError('Missing one or more image streams')
            time.sleep(.05)
        start = time.monotonic() + .2
        for recorder in recorders.values():
            recorder.launch(start)
        end = start + args.duration
        client.socket_services.recv_timeout = 2000
        tick = start
        while tick < end:
            time.sleep(max(0, tick - time.monotonic()))
            clock.append((time.monotonic(), world.get_sim_time()))
            tick += .2
        for recorder in recorders.values():
            recorder.end = end
        report['recordings'] = {tag: recorder.finish(end) for tag, recorder in recorders.items()}
        client.unsubscribe_all()
        for tag in rows:
            rows[tag] = [row for row in rows[tag] if start <= row[0] <= end]
        report['streams'] = {tag: summarize(samples) for tag, samples in rows.items()}
        report['rgbd_after_recording'] = rgbd_summary()
        if any(item['error'] for item in report['recordings'].values()):
            raise RuntimeError('Recording failed')
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        for recorder in recorders.values():
            if recorder.start is not None and recorder.thread.is_alive():
                recorder.end = time.monotonic()
                recorder.thread.join()
        if client.state:
            client.disconnect()
        stop.touch()
        try:
            _, error = monitor.communicate(timeout=8)
            if error:
                report['window_monitor_stderr'] = error
        except subprocess.TimeoutExpired:
            monitor.terminate()
            monitor.wait(timeout=5)
        with (out / 'sensor-arrivals.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('sensor', 'arrival_monotonic_s', 'source_timestamp_ns'))
            writer.writerows((tag, arrival, ts) for tag, values in rows.items() for arrival, ts in values)
        with (out / 'clock.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('observed_monotonic_s', 'sim_timestamp_ns'))
            writer.writerows(clock)
        (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
