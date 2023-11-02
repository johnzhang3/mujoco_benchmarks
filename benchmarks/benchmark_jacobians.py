import concurrent.futures
import numpy as np
import threading
import mujoco
import timeit
import torch 

model = mujoco.MjModel.from_xml_path("robots/a1/scene.xml")
data = mujoco.MjData(model)

thread_local = threading.local()

def thread_initializer():
    thread_local.data = mujoco.MjData(model)

def compute_jacobian(A, B, qpos, start, end):
    for i in range(start, end):
        thread_local.data.qpos[:] = qpos[i]
        mujoco.mjd_transitionFD(model, thread_local.data, 1e-6, True, A[i], B[i], None, None)

def compute_multi_jacobian(n=100, num_workers=10):
    A = np.zeros((n, 2*model.nv+model.na, 2*model.nv+model.na))
    B = np.zeros((n, 2*model.nv+model.na, model.nu))
    qpos = np.random.rand(n, model.nq) # replace later with pytorch input

    n_per_worker = n // num_workers
    chunks = [(i * n_per_worker, (i + 1) * n_per_worker) for i in range(num_workers)]
    # Adjust the last chunk to cover the remainder
    chunks[-1] = (chunks[-1][0], n)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers, initializer=thread_initializer) as executor:
        futures = [executor.submit(compute_jacobian, A, B, qpos, start, end) for start, end in chunks]
        concurrent.futures.wait(futures)

    return A, B

# Now A, B, and qpos are updated in place, and you can access them as needed
n = 4096
num_workers = 20

A, B = compute_multi_jacobian(n=n, num_workers=num_workers)
print(f"A shape: {A.shape}")

t_to_torch = timeit.timeit(lambda: torch.from_numpy(A).to("cuda"), number=10)/10
print(f"cpu to gpu time: {t_to_torch}") # 0.25 seconds

A_torch = torch.from_numpy(A).to("cuda")
t_to_cpu = timeit.timeit(lambda: A_torch.cpu().numpy(), number=10)/10 
print(f"gpu to cpu time: {t_to_cpu}") # 0.01 seconds

t_jacobian = timeit.timeit(lambda: compute_multi_jacobian(n=4000, num_workers=20), number=10)/10
print(f"evaluating {n} jacobians on {num_workers} threads took: {t_jacobian} seconds") #0.32 seconds
print(f"jacobian evals per second: {n/t_jacobian}")


