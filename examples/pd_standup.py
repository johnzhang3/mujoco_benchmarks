import mujoco
import mujoco_viewer
import numpy as np
import copy as cp

model = mujoco.MjModel.from_xml_path("robots/a1/scene.xml")
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)

# reset robot (keyframes are defined in the xml)
mujoco.mj_resetDataKeyframe(model, data, 0) # stand position
mujoco.mj_forward(model, data)
q_ref_mj = cp.deepcopy(data.qpos) # save reference pose

mujoco.mj_resetDataKeyframe(model, data, 1) # set the robot sitting down
mujoco.mj_forward(model, data)

# pd gains
kp = 100
kd = 10

while True:
    # tracking joint angles
    tau = kp*(q_ref_mj[7:] - data.qpos[7:]) - kd*data.qvel[6:]
    data.ctrl[:] = tau
    mujoco.mj_step(model, data)
    if viewer.is_alive:
        viewer.render()
    else:
        break

    # print(np.linalg.norm(q_ref_mj[7:] - data.qpos[7:], ord=np.inf))
    print(data.qpos[7:10]) # print the joint angles for one leg