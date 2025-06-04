import numpy as np
from scipy.spatial.transform import Rotation as R
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


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert a quaternion into a 3×3 rotation matrix.
    Expects q in the format [x, y, z, w] with shape (..., 4).
    Returns a matrix of shape (..., 3, 3).
    """
    # Ensure q is a float tensor and normalize
    q = q.to(dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True)

    x, y, z, w = q.unbind(-1)  # each has shape (...)

    # Compute products once
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    xw = x * w
    yz = y * z
    yw = y * w
    zw = z * w

    # Build the rotation matrix entries
    m00 = ww + xx - yy - zz
    m01 = 2 * (xy - zw)
    m02 = 2 * (xz + yw)

    m10 = 2 * (xy + zw)
    m11 = ww - xx + yy - zz
    m12 = 2 * (yz - xw)

    m20 = 2 * (xz - yw)
    m21 = 2 * (yz + xw)
    m22 = ww - xx - yy + zz

    # Stack into shape (..., 3, 3)
    row0 = torch.stack((m00, m01, m02), dim=-1)
    row1 = torch.stack((m10, m11, m12), dim=-1)
    row2 = torch.stack((m20, m21, m22), dim=-1)

    return torch.stack((row0, row1, row2), dim=-2)


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a rotation matrix (shape (..., 3, 3)) into a quaternion [x,y,z,w].
    Returns a tensor of shape (..., 4).
    Uses the “trace” method.
    """
    # Assume R is float32
    m00 = R[..., 0, 0]
    m11 = R[..., 1, 1]
    m22 = R[..., 2, 2]
    trace = m00 + m11 + m22  # (...)

    # Allocate q components
    x = torch.empty_like(trace)
    y = torch.empty_like(trace)
    z = torch.empty_like(trace)
    w = torch.empty_like(trace)

    # Case 1: trace > 0
    t0 = trace + 1.0
    mask0 = trace > 0
    if mask0.any():
        s0 = torch.sqrt(t0[mask0]) * 2.0  # s = 4*w
        w[mask0] = 0.25 * s0
        x[mask0] = (R[..., 2, 1][mask0] - R[..., 1, 2][mask0]) / s0
        y[mask0] = (R[..., 0, 2][mask0] - R[..., 2, 0][mask0]) / s0
        z[mask0] = (R[..., 1, 0][mask0] - R[..., 0, 1][mask0]) / s0

    # Case 2: R[0,0] is largest diagonal
    mask1 = (~mask0) & (m00 > m11) & (m00 > m22)
    if mask1.any():
        t1 = 1.0 + m00[mask1] - m11[mask1] - m22[mask1]
        s1 = torch.sqrt(t1) * 2.0  # s = 4*x
        w[mask1] = (R[..., 2, 1][mask1] - R[..., 1, 2][mask1]) / s1
        x[mask1] = 0.25 * s1
        y[mask1] = (R[..., 0, 1][mask1] + R[..., 1, 0][mask1]) / s1
        z[mask1] = (R[..., 0, 2][mask1] + R[..., 2, 0][mask1]) / s1

    # Case 3: R[1,1] is largest diagonal
    mask2 = (~mask0) & (~mask1) & (m11 > m22)
    if mask2.any():
        t2 = 1.0 - m00[mask2] + m11[mask2] - m22[mask2]
        s2 = torch.sqrt(t2) * 2.0  # s = 4*y
        w[mask2] = (R[..., 0, 2][mask2] - R[..., 2, 0][mask2]) / s2
        x[mask2] = (R[..., 0, 1][mask2] + R[..., 1, 0][mask2]) / s2
        y[mask2] = 0.25 * s2
        z[mask2] = (R[..., 1, 2][mask2] + R[..., 2, 1][mask2]) / s2

    # Case 4: R[2,2] is largest diagonal
    mask3 = (~mask0) & (~mask1) & (~mask2)
    if mask3.any():
        t3 = 1.0 - m00[mask3] - m11[mask3] + m22[mask3]
        s3 = torch.sqrt(t3) * 2.0  # s = 4*z
        w[mask3] = (R[..., 1, 0][mask3] - R[..., 0, 1][mask3]) / s3
        x[mask3] = (R[..., 0, 2][mask3] + R[..., 2, 0][mask3]) / s3
        y[mask3] = (R[..., 1, 2][mask3] + R[..., 2, 1][mask3]) / s3
        z[mask3] = 0.25 * s3

    return torch.stack((x, y, z, w), dim=-1)

def sensor2global(ori, acc, calibration_quats, device_id):
    """Convert the sensor data to the global inertial frame."""
    device_mean_quat = calibration_quats[device_id]
    device_mean_quat = torch.tensor(device_mean_quat, dtype=torch.float32)
    ori = torch.tensor(ori, dtype=torch.float32)
    acc = torch.tensor(acc, dtype=torch.float32)

    og_mat = quaternion_to_matrix(ori)
    global_inertial_frame = quaternion_to_matrix(device_mean_quat)
    global_mat = torch.matmul(global_inertial_frame.T, og_mat)         # R_g←s
    global_ori = matrix_to_quaternion(global_mat)          # (..., 4)
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

