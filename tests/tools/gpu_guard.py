"""Serialize this user's validation jobs and refuse an occupied physical GPU."""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import tempfile
import time


def require_idle_gpu(gpu):
    if gpu < 0:
        raise ValueError('GPU index must be nonnegative')
    result = subprocess.run(
        ['nvidia-smi', f'--id={gpu}',
         '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=15)
    fields = result.stdout.strip().split(',')
    if len(fields) != 3:
        raise RuntimeError(f'Cannot identify GPU {gpu}: {result.stdout.strip()}')
    uuid, memory, utilization = (field.strip() for field in fields)
    processes = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=15)
    active = [line for line in processes.stdout.splitlines() if line.split(',')[0].strip() == uuid]
    if active or int(memory) > 64 or int(utilization) != 0:
        raise RuntimeError(f'GPU {gpu} is occupied: memory={memory} MiB, '
                           f'utilization={utilization}%, compute processes={active}')
    return uuid


@contextmanager
def idle_gpu(gpu):
    """Hold an advisory per-user lock; also check for unrelated GPU processes."""
    # nvidia-smi utilization averages can lag a completed child by a second.
    # Wait briefly for the device to become idle; never bypass an occupied check.
    deadline = time.monotonic() + 5
    while True:
        try:
            uuid = require_idle_gpu(gpu)
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)
    lock = Path(tempfile.gettempdir()) / f'go-fann-gpu-{os.getuid()}-{uuid}.lock'
    with lock.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another GO-FANN validation owns GPU {gpu}') from error
        require_idle_gpu(gpu)
        # CUDA ordinals can differ from nvidia-smi indices; bind the device
        # whose UUID we actually checked and locked.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_DYNAMIC='FALSE')
        prefix = env.get('CONDA_PREFIX')
        if prefix:
            libraries = [str(Path(prefix) / 'lib'), str(Path(prefix) / 'targets/x86_64-linux/lib')]
            libraries.extend(filter(None, [env.get('LD_LIBRARY_PATH', '')]))
            env['LD_LIBRARY_PATH'] = ':'.join(libraries)
        yield env


def run_on_idle_gpu(command, gpu, **kwargs):
    with idle_gpu(gpu) as env:
        return subprocess.run(command, env=env, **kwargs)
