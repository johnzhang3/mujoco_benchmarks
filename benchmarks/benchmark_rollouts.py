import concurrent.futures
import numpy as np
import threading
import mujoco
import timeit
from mujoco import rollout
from etils import epath

path = epath.Path(epath.resource_path('mujoco')) / (
        'mjx/benchmark/model/humanoid')
model = mujoco.MjModel.from_xml_path(
    (path / 'humanoid.xml').as_posix())
# model = mujoco.MjModel.from_xml_path("robots/a1/scene.xml")
# model = mujoco.MjModel.from_xml_path("robots/google_barkour_v0/scene.xml")

data = mujoco.MjData(model)

thread_local = threading.local()

def thread_initializer():
    thread_local.data = mujoco.MjData(model)

def call_rollout(initial_state, ctrl, state):
    rollout.rollout(model, thread_local.data, skip_checks=True,
                    nstate=initial_state.shape[0], nstep=nstep,
                    initial_state=initial_state, ctrl=ctrl, state=state)
    
def threaded_rollout(state, ctrl, initial_state, num_workers=32, nstep=5):

    n = initial_state.shape[0] // num_workers  # integer division
    chunks = []  # a list of tuples, one per worker
    for i in range(num_workers-1):
        chunks.append(
            (initial_state[i*n:(i+1)*n], ctrl[i*n:(i+1)*n], state[i*n:(i+1)*n]))

    # last chunk, absorbing the remainder:
    chunks.append(
        (initial_state[(num_workers-1)*n:], ctrl[(num_workers-1)*n:],
            state[(num_workers-1)*n:]))

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers, initializer=thread_initializer) as executor:
        futures = []
        for chunk in chunks:
            futures.append(executor.submit(call_rollout, *chunk))
        for future in concurrent.futures.as_completed(futures):
            future.result()

num_workers = 20
nstate = 10000
nstep = 5
state = np.zeros((nstate, nstep, model.nq+model.nv+model.na))
ctrl = np.random.randn(nstate, nstep, model.nu)
initial_state = np.random.randn(nstate, model.nq+model.nv+model.na)

t_rollout_thread = timeit.timeit(lambda: threaded_rollout(state, ctrl, initial_state, 
                                             num_workers=num_workers, nstep=nstep), number=10)/10

print(f"rollout {nstate} states with {num_workers} threads in parallel took {t_rollout_thread} seconds") # 0.09 seconds
print(f"steps per second: {nstate*nstep/t_rollout_thread}")

t_rollout_single = timeit.timeit(lambda: rollout.rollout(model, data, skip_checks=True,
                nstate=nstate, nstep=nstep, initial_state=initial_state, ctrl=ctrl, state=state), number=10)/10

print(f"rollout {nstate} states with 1 thread took {t_rollout_single} seconds") # 0.87 seconds
print(f"steps per second: {nstate*nstep/t_rollout_single}")

