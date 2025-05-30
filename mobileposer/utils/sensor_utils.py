import numpy as np
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from collections import deque

from mobileposer.config import *
from mobileposer.constants import *


class SensorData:
    """Store sensor data from devices."""
    def __init__(self):
        self.raw_acc_buffer = {
            _id: deque(np.zeros((BUFFER_SIZE, 3)), maxlen=BUFFER_SIZE) 
            for _id in sensor.device_ids.values()
        }
        self.raw_ori_buffer = {
            _id: deque(np.array([[0, 0, 0, 1]] * BUFFER_SIZE), maxlen=BUFFER_SIZE) 
            for _id in sensor.device_ids.values()
        }
        self.calibration_quats = {
            _id: np.array([0, 0, 0, 1]) 
            for _id in sensor.device_ids.values()
        }
        self.virtual_acc = {
            _id: np.zeros((1, 3)) 
            for _id in sensor.device_ids.values()
        }
        self.virtual_ori = {
            _id: np.array([0, 0, 0, 1]) 
            for _id in sensor.device_ids.values()
        }
        self.reference_times = {
            _id: None 
            for _id in sensor.device_ids.values()
        }

    def update(self, device_id, curr_acc, curr_ori, timestamps):
        if self.reference_times[device_id] is None:
            self.reference_times[device_id] = [timestamps[0], timestamps[1]]

        curr_timestamp = (
            self.reference_times[device_id][0] + 
            timestamps[1] - self.reference_times[device_id][1]
        )

        self.raw_acc_buffer[device_id].append(curr_acc.flatten())
        self.raw_ori_buffer[device_id].append(curr_ori.flatten())

        #bug? does not update timestamp
        
        return curr_timestamp
    
    def calibrate(self):
        for _id, ori_buffer in self.raw_ori_buffer.items():
            if len(ori_buffer) < 30:
                print(f"Not enough data to calibrate for device {_id}.")
                continue
            # Convert deque to list and then to Rotation objects
            quaternions = np.array(ori_buffer)[-30:]
            rotations = R.from_quat(quaternions)
            # Compute the mean rotation
            mean_rotation = rotations.mean()
            self.calibration_quats[_id] = mean_rotation.as_quat()

    def get_timestamps(self, device_id):
        return self.reference_times[device_id][-1]

    def get_orientation(self, device_id):
        return self.raw_ori_buffer[device_id][-1]

    def get_acceleration(self, device_id):
        return self.raw_acc_buffer[device_id][-1]

    def update_virtual(self, device_id, glb_acc, glb_ori):
        self.virtual_acc[device_id] = glb_acc.reshape(1, 3)
        self.virtual_ori[device_id] = glb_ori


def process_data(message):
    """Process the data from the sensors (e.g., iPhone, Apple Watch, etc.)."""
    message = message.strip()
    if not message:
        return
    message = message.decode('utf-8')
    if message == STOP:
        return
    if SEP not in message:
        return 

    try:
        device_id, raw_data_str = message.split(";")
        device_type, data_str = raw_data_str.split(":")
    except Exception as e:
        print("(1) Exception encountered: ", e)
        return

    data = []
    for d in data_str.strip().split(" "):
        try: 
            data.append(float(d))
        except Exception as e:
            print("(2) Exception encountered: ", e)
            continue
    
    if (len(data) != len(KEYS)) and (len(data) != len(KEYS) - 3):
        print("Missing data!")
        return
    
    device_name = sensor.device_ids[f"{device_id.capitalize()}_{device_type}"]
    send_str = f"w{data[8]}wa{data[5]}ab{data[6]}bc{data[7]}c"

    # update the buffers
    curr_acc = np.array(data[2:5]).reshape(1, 3)
    curr_ori = np.array(data[5:9]).reshape(1, 4)
    timestamps = data[:2]

    if device_name == Devices.Right_Headphone:
        curr_euler = R.from_quat(curr_ori).as_euler("xyz").squeeze()
        fixed_euler = np.array([[curr_euler[0] * -1, curr_euler[2], curr_euler[1]]])
        curr_ori = R.from_euler("xyz", fixed_euler).as_quat().reshape(1, 4)
        curr_acc = np.array([[curr_acc[0, 0]*-1, curr_acc[0, 2], curr_acc[0, 1]]])

    return send_str, device_name, curr_acc, curr_ori, timestamps

def matrix_to_quaternion_single(matrix: torch.Tensor) -> torch.Tensor:
    """
    Args
    ----
    matrix : torch.Tensor, shape (3, 3)
             Rotation matrix  (rows are destination axes).

    Returns
    -------
    quat   : torch.Tensor, shape (4,)  --  (w, x, y, z) with ||quat|| == 1
    """

    # ------------------------------
    # 1.  Pull out the nine elements
    # ------------------------------
    m00, m01, m02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    m10, m11, m12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    m20, m21, m22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]

    one = matrix.new_tensor(1.0)

    # -------------------------------------------------------
    # 2.  Same four “absolute” terms as the original function
    # -------------------------------------------------------
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    one + m00 + m11 + m22,        # w‑candidate
                    one + m00 - m11 - m22,        # x‑candidate
                    one - m00 + m11 - m22,        # y‑candidate
                    one - m00 - m11 + m22,        # z‑candidate
                ]
            ),
            min=0.0,
        )
    )                                           # shape (4,)

    # ---------------------------------------------------
    # 3.  Build the four candidate quaternions (unnormal)
    # ---------------------------------------------------
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[0] ** 2,
                         m21 - m12,
                         m02 - m20,
                         m10 - m01]),
            torch.stack([m21 - m12,
                         q_abs[1] ** 2,
                         m10 + m01,
                         m02 + m20]),
            torch.stack([m02 - m20,
                         m10 + m01,
                         q_abs[2] ** 2,
                         m12 + m21]),
            torch.stack([m10 - m01,
                         m20 + m02,
                         m21 + m12,
                         q_abs[3] ** 2]),
        ],
        dim=0,                                   # shape (4, 4)
    )

    # ---------------------------------------------------
    # 4.  Divide each row by the same “safe” denominator
    # ---------------------------------------------------
    floor = matrix.new_tensor(0.1)
    denom = 2.0 * torch.max(q_abs, floor)        # shape (4,)
    quat_candidates = quat_by_rijk / denom.unsqueeze(-1)

    # ----------------------------------------------------------
    # 5.  Pick the best‑conditioned row (largest q_abs element)
    #     – use tensor‑indexing so the tracer records a gather
    # ----------------------------------------------------------
    best_idx = torch.argmax(q_abs)               # tensor scalar
    quat = quat_candidates[best_idx]             # shape (4,)

    # 6.  Normalize to exactly unit length and return (w, x, y, z)
    return quat / quat.norm(p=2)


def sensor2global(ori, acc, calibration_quats, device_id):
    """Convert the sensor data to the global inertial frame."""
    device_mean_quat = calibration_quats[device_id]
    device_mean_quat = torch.tensor(device_mean_quat, dtype=torch.float32)
    ori = torch.tensor(ori, dtype=torch.float32)
    acc = torch.tensor(acc, dtype=torch.float32)

    og_mat = quaternion_to_matrix(ori)
    global_inertial_frame = quaternion_to_matrix(device_mean_quat)
    global_mat = torch.matmul(global_inertial_frame.T, og_mat)         # R_g←s
    global_ori = matrix_to_quaternion_single(global_mat)          # (..., 4)
    acc_ref   = torch.matmul(og_mat, acc.unsqueeze(-1)).squeeze(-1)
    global_acc = torch.matmul(global_inertial_frame.T, acc_ref.unsqueeze(-1)).squeeze(-1)

    return np.array(global_ori), np.array(global_acc)
    
    og_mat = R.from_quat(ori).as_matrix() # rotation matrix from quaternion
    global_inertial_frame = R.from_quat(device_mean_quat).as_matrix()
    
    global_mat = (global_inertial_frame.T).dot(og_mat)
    global_ori = R.from_matrix(global_mat).as_quat()
    
    acc_ref_frame = og_mat.dot(acc) # align acc. to sensor frame of reference
    global_acc = (global_inertial_frame.T).dot(acc_ref_frame) # align acc. to world frame

    return  global_ori, global_acc

