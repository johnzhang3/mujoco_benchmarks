import mujoco
import mujoco_viewer
import numpy as np
import copy as cp
from utils import infinite_horizon_lqr, quaternion_to_axis_angle

model = mujoco.MjModel.from_xml_path("robots/a1/scene.xml")
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)

# reset robot (keyframes are defined in the xml)
mujoco.mj_resetDataKeyframe(model, data, 0) # stand position
mujoco.mj_forward(model, data)
q_ref_mj = cp.deepcopy(data.qpos) # save reference pose
u_ref_mj = cp.deepcopy(data.ctrl) # save reference control

# linearize about the stand position
A = np.zeros((2*model.nv, 2*model.nv))
B = np.zeros((2*model.nv, model.nu))
mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)

# from sitting position doesn't work yet
# mujoco.mj_resetDataKeyframe(model, data, 1) # set the robot sitting down
# mujoco.mj_forward(model, data)

# compute LQR gains
Q = np.eye(2*model.nv) * np.concatenate((np.ones(6)*0.1, np.ones(12)*10, np.ones(model.nv)*0.1))
R = np.eye(model.nu)*1e-6

K, P = infinite_horizon_lqr(A, B, Q, R)

# import matplotlib.pyplot as plt
# plt.imshow(K)
# plt.colorbar()
# plt.show()

while True:
    # tracking q_ref, note that rotation in q_ref is in quaternion form but the derivatives are in the 3 parameter space
    x = np.concatenate((data.qpos[0:3], quaternion_to_axis_angle(data.qpos[3:7]), data.qpos[7:], data.qvel))
    data.ctrl[:] = u_ref_mj - K @ (x - np.concatenate((q_ref_mj[0:3], quaternion_to_axis_angle(q_ref_mj[3:7]), q_ref_mj[7:], np.zeros(model.nv))))
    mujoco.mj_step(model, data)

    if viewer.is_alive:
        viewer.render()
    else:
        break

    # print(np.linalg.norm(q_ref_mj[7:] - data.qpos[7:], ord=np.inf))
    print(data.qpos[7:10]) # print the joint angles for one leg

