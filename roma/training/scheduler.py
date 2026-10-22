"""Bounded subprocess scheduler for independent seed/model/dataset groups."""
import json
import os
from pathlib import Path
import subprocess
import time
import signal
from collections import deque


def is_memory_error(message):
    message = message.lower()
    return any(token in message for token in ('out of memory', 'outofmemoryerror',
        'cublas_status_alloc_failed', 'cudnn_status_alloc_failed', 'cannot allocate memory', 'unable to allocate'))


def run_jobs(jobs, output, concurrency=1, threads=2, cwd=None, max_retries=3, retry_delay=30, worker_timeout=7200):
    if concurrency < 1 or threads < 1 or max_retries < 0 or retry_delay < 0 or worker_timeout < 0:
        raise ValueError('Invalid concurrency, threads or retry settings')
    output = Path(output)
    logs = output/'workers'
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({k: str(threads) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    env.update(ROMA_RETRY_WORKER='1', ROMA_RESUME_COMPLETED='1')
    active, queue, limit = {}, deque(jobs), concurrency
    progress_path = output/'progress.json'
    previous_jobs = {j['id']: j for j in json.loads(progress_path.read_text()).get('jobs', [])} if progress_path.exists() else {}
    for job in jobs:
        previous = previous_jobs.get(job['id'], {})
        job.update(state='pending', attempts=previous.get('attempts', 0), run_attempts=0,
                   attempt_history=list(previous.get('attempt_history', [])), not_before=0)

    def stop(proc, sig):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    def save():
        done = sum(j['state'] in ('completed', 'failed') for j in jobs)
        state = {'total_groups': len(jobs), 'completed_groups': done,
                 'running_groups': len(active), 'concurrency': concurrency, 'current_concurrency': limit,
                 'queued_groups': len(queue), 'max_retries': max_retries,
                 'group_progress_percent': 100*done/len(jobs) if jobs else 100,
                 'note': 'Group completion is not equal to runtime progress; final result audit determines experiment success.',
                 'jobs': jobs}
        tmp = output/'progress.json.tmp'
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(output/'progress.json')

    def finish(job, code, message):
        nonlocal limit
        memory = code != 0 and (code in (86, -9, 137) or is_memory_error(message))
        kind = 'worker_timeout' if job.get('timed_out') else ('memory_pressure' if memory else ('worker_error' if code else None))
        elapsed = time.time()-job['started_at']
        job.update(returncode=code, elapsed_s=elapsed, failure_kind=kind)
        job['attempt_history'].append({'attempt': job['attempts'], 'returncode': code,
            'failure_kind': kind, 'log': job['log'], 'started_at': job['started_at'], 'elapsed_s': elapsed})
        if code == 0:
            job['state'] = 'completed'
            return
        job['last_error'] = message[-4000:]
        if memory:
            limit = max(1, limit//2)
            print(f'[Retry] memory pressure: limiting concurrency to {limit}', flush=True)
        if job['run_attempts'] <= max_retries:
            delay = min(300, retry_delay * 2**(job['run_attempts']-1))
            job.update(state='retry_wait', not_before=time.time()+delay)
            queue.append(job)
            print(f'[Retry] {job["id"]} returned to queue tail; retry after {delay:g}s '
                  f'({job["run_attempts"]}/{max_retries} retries scheduled this launch)', flush=True)
        else:
            job['state'] = 'failed'
            print(f'[Failed] {job["id"]} exhausted {job["attempts"]} attempts; see {job["log"]}', flush=True)

    save()
    try:
        while queue or active:
            # Inspect each queued job at most once per polling cycle.
            for _ in range(len(queue)):
                if len(active) >= limit:
                    break
                job = queue.popleft()
                if job['not_before'] > time.time():
                    queue.append(job)
                    continue
                job['attempts'] += 1
                job['run_attempts'] += 1
                job['timed_out'] = False
                log_path = logs/f'{job["id"]}.attempt{job["attempts"]:02d}.log'
                handle = log_path.open('w')
                worker_env = dict(env, ROMA_PARALLEL_JOBS=str(limit), ROMA_WORKER_ATTEMPT=str(job['attempts']))
                worker_env.update({key: str(value) for key, value in job.get('env', {}).items()})
                command = job['command']
                worker_env['ROMA_WORKER_SEED'] = command[command.index('--seed')+1] if '--seed' in command else '2021'
                try:
                    proc = subprocess.Popen(command, cwd=cwd, env=worker_env,
                                            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                            start_new_session=True)
                except OSError as error:
                    handle.write(f'Worker launch failed: {error}\n')
                    handle.close()
                    job.update(log=str(log_path), started_at=time.time())
                    finish(job, 127, str(error))
                    save()
                    continue
                except BaseException:
                    handle.close()
                    raise
                job.update(state='running', pid=proc.pid, log=str(log_path), started_at=time.time(), timed_out=False)
                active[proc.pid] = (proc, handle, job)
                print(f'[Parallel] started {job["id"]} attempt={job["attempts"]} pid={proc.pid}; log={log_path}', flush=True)
                save()
            for pid, (proc, handle, job) in list(active.items()):
                code = proc.poll()
                if code is None:
                    if worker_timeout and time.time()-job['started_at'] > worker_timeout:
                        if not job.get('timed_out'):
                            job.update(timed_out=True, timeout_at=time.time())
                            stop(proc, signal.SIGTERM)
                            print(f'[Timeout] stopping {job["id"]}', flush=True)
                        elif time.time()-job['timeout_at'] > 10:
                            stop(proc, signal.SIGKILL)
                    continue
                handle.close()
                del active[pid]
                with Path(job['log']).open('rb') as log:
                    log.seek(max(0, Path(job['log']).stat().st_size-65536))
                    tail = log.read().decode(errors='replace')
                finish(job, 124 if job.get('timed_out') else code,
                       tail + ('\nWorker runtime exceeded timeout' if job.get('timed_out') else ''))
                done = sum(j['state'] in ('completed', 'failed') for j in jobs)
                print(f'[Parallel] {done}/{len(jobs)} groups ({100*done/len(jobs):.1f}%) '
                      f'{job["id"]} exit={code}', flush=True)
                save()
            if active or queue:
                time.sleep(.2)
    finally:
        # Includes Ctrl-C, launcher errors and scheduler exceptions.
        for proc, _, _ in active.values():
            if proc.poll() is None:
                stop(proc, signal.SIGTERM)
        for proc, handle, job in active.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                stop(proc, signal.SIGKILL)
                proc.wait()
            handle.close()
            job.update(state='interrupted', returncode=proc.returncode)
        active.clear()
        save()
    failed = [j['id'] for j in jobs if j['state'] != 'completed']
    return failed
