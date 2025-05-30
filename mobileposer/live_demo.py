"""
Code adapted from: https://github.com/Xinyu-Yi/TransPose/blob/main/live_demo.py
"""

import os
import time
import socket
import threading
import torch
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from pygame.time import Clock
import pickle

from articulate.math import *
from mobileposer.models import *
from mobileposer.utils.model_utils import *
from mobileposer.config import *
import struct

# Configurations 
USE_PHONE_AS_WATCH = False


class IMUSet:
    """
    Sensor order: left forearm, right forearm, left lower leg, right lower leg, head, pelvis
    """
    def __init__(self, imu_host='127.0.0.1', imu_port=7777, buffer_len=26):
        """
        Init an IMUSet for Noitom Perception Legacy IMUs. Please follow the instructions below.

        Instructions:
        --------
        1. Start `Axis Legacy` (Noitom software).
        2. Click `File` -> `Settings` -> `Broadcasting`, check `TCP` and `Calculation`. Set `Port` to 7002.
        3. Click `File` -> `Settings` -> `Output Format`, change `Calculation Data` to
           `Block type = String, Quaternion = Global, Acceleration = Sensor local`
        4. Place 1 - 6 IMU on left lower arm, right lower arm, left lower leg, right lower leg, head, root.
        5. Connect 1 - 6 IMU to `Axis Legacy` and continue.

        :param imu_host: The host that `Axis Legacy` runs on.
        :param imu_port: The port that `Axis Legacy` runs on.
        :param buffer_len: Max number of frames in the readonly buffer.
        """
        self.imu_host = imu_host
        self.imu_port = imu_port
        self.clock = Clock()

        self._imu_socket = None
        self._buffer_len = buffer_len
        self._quat_buffer = []
        self._acc_buffer = []
        self._is_reading = False
        self._read_thread = None

    def _read(self):
        """
        The thread that reads imu measurements into the buffer. It is a producer for the buffer.
        """
        while self._is_reading:
            data, addr = self._imu_socket.recvfrom(1024)
 
            data_str = data.decode("utf-8")
            print(data_str)
            a = np.array(data_str.split("#")[0].split(",")).astype(np.float64)
            q = np.array(data_str.split("#")[1].strip("$").split(",")).astype(np.float64)

            acc = a.reshape((-1, 3))
            quat = q.reshape((-1, 4))

            tranc = int(len(self._quat_buffer) == self._buffer_len)
            self._quat_buffer = self._quat_buffer[tranc:] + [quat.astype(float)]
            self._acc_buffer = self._acc_buffer[tranc:] + [-9.8 * acc.astype(float)]
            self.clock.tick()


    def start_reading(self):
        """
        Start reading imu measurements into the buffer.
        """
        if self._read_thread is None:
            self._is_reading = True
            self._read_thread = threading.Thread(target=self._read)
            self._read_thread.setDaemon(True)
            self._quat_buffer = []
            self._acc_buffer = []
            self._imu_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._imu_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._imu_socket.bind((self.imu_host, self.imu_port))
            self._read_thread.start()
        else:
            print('Failed to start reading thread: reading is already start.')

    def stop_reading(self):
        """
        Stop reading imu measurements.
        """
        if self._read_thread is not None:
            self._is_reading = False
            self._read_thread.join()
            self._read_thread = None
            self._imu_socket.close()

    def get_current_buffer(self):
        """
        Get a view of current buffer.

        :return: Quaternion and acceleration torch.Tensor in shape [buffer_len, 6, 4] and [buffer_len, 6, 3].
        """
        q = torch.from_numpy(np.array(self._quat_buffer)).float()
        a = torch.from_numpy(np.array(self._acc_buffer)).float()
        return q, a

    def get_mean_measurement_of_n_second(self, num_seconds=3, buffer_len=120):
        """
        Start reading for `num_seconds` seconds and then close the connection. The average of the last
        `buffer_len` frames of the measured quaternions and accelerations are returned.
        Note that this function is blocking.

        :param num_seconds: How many seconds to read.
        :param buffer_len: Buffer length. Must be smaller than 60 * `num_seconds`.
        :return: The mean quaternion and acceleration torch.Tensor in shape [6, 4] and [6, 3] respectively.
        """
        save_buffer_len = self._buffer_len
        self._buffer_len = buffer_len
        self.start_reading()
        time.sleep(num_seconds)
        self.stop_reading()
        q, a = self.get_current_buffer()
        self._buffer_len = save_buffer_len
        return q.mean(dim=0), a.mean(dim=0)


def get_input():
    global running, start_recordingm
    while running:
        c = input()
        if c == 'q':
            running = False
        elif c == 'r':
            start_recording = True
        elif c == 's':
            start_recording = False

def pack_tensor(tensor) -> bytes:
    arr = np.array(tensor, dtype=np.float32)
    shape = arr.shape
    assert shape == (1, 45, 60) or shape == (1,), f"Shape mismatch: got {shape!r}"
    data = arr.tobytes()
    buf = struct.pack('<I', len(shape))
    for d in shape:
        buf += struct.pack('<I', d)
    buf += struct.pack('<I', len(data))
    buf += data
    return buf

def send_tensors(tensors, connection):
    body = struct.pack('<I', len(tensors))          
    for t in tensors:
        body += pack_tensor(t)

    # prefix with total length
    packet = struct.pack('<I', len(body)) + body
    connection.sendall(packet)


class PhoneUnpacker:
    def __init__(self):
        # holds any bytes we've received but not yet parsed
        self._buffer = b''

    def unpack_from_phone(self, new_data):
        """
        Feed in the next chunk of bytes from conn.recv(), and once
        we have one full packet, parse out the four tensors and
        return them as NumPy arrays.  Otherwise return None.
        """
        self._buffer += new_data

        # need at least 4 bytes for the total‐length header
        if len(self._buffer) < 4:
            return None

        # read the packet length (little‐endian UInt32)
        total_len = struct.unpack_from('<I', self._buffer, 0)[0]

        # wait until we have the full packet
        if len(self._buffer) < 4 + total_len:
            return None

        # extract the packet body and advance the buffer
        body = self._buffer[4:4 + total_len]
        self._buffer = self._buffer[4 + total_len:]

        # now parse body:
        offset = 0

        def read_u32():
            nonlocal offset
            val = struct.unpack_from('<I', body, offset)[0]
            offset += 4
            return val

        # 1) number of tensors
        count = read_u32()
        arrays = []

        for _ in range(count):
            # 2) rank
            rank = read_u32()
            # 3) dims
            dims = [ read_u32() for _ in range(rank) ]
            # 4) byte length
            byte_count = read_u32()
            # 5) payload
            chunk = body[offset:offset + byte_count]
            offset += byte_count

            # build the numpy array
            arr = np.frombuffer(chunk, dtype=np.float32)
            arr = arr.reshape(dims)
            arrays.append(torch.tensor(arr, dtype=torch.float32))

        # return a tuple of the four arrays
        return tuple(arrays)


def reader_phone_loop(conn, model, poses, trans, unity_conn):
    unpacker = PhoneUnpacker()
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                print("Client disconnected.")
                break
            
            #unpack here
            result = unpacker.unpack_from_phone(data)
            if result is None:
                continue
            pose, pred_joints, vel, contact = result

            # post processing model backbone outputs
            output = model.process_outputs(pose, pred_joints, vel, contact)

            pred_pose = output[0] # [24, 3, 3]
            pred_tran = output[2] # [3]
        
            # convert rotmatrix to axis angle
            pose = rotation_matrix_to_axis_angle(pred_pose.view(1, 216)).view(72)

            if unity_conn is not None:
                s = ','.join(['%g' % v for v in pose]) + '#' + \
                    ','.join(['%g' % v for v in pred_tran]) + '$'
                unity_conn.send(s.encode('utf8'))  

            poses.append(pose)
            trans.append(pred_tran)

    except Exception as e:
        print("Reader thread error:", e)
    finally:
        conn.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--on_device", action='store_true')
    args = parser.parse_args()
    
    # specify device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # setup IMU collection
    imu_set = IMUSet(buffer_len=1)

    # load model
    model = load_model(paths.weights_file)

    n_imus = 5
    accs, oris = [], []
    raw_accs, raw_oris = [], []
    poses, trans = [], []

    model.eval()

    # setup Unity server for visualization
    unity_conn = None
    if args.vis:
        server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_for_unity.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        server_for_unity.bind(('0.0.0.0', 8889))
        server_for_unity.listen(1)
        print('Server start. Waiting for unity3d to connect.')
        unity_conn, unity_addr = server_for_unity.accept()

    if args.on_device:
        server_for_phone = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_for_phone.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        server_for_phone.bind(('10.105.238.137', 8889))
        server_for_phone.listen(1)
        print('Server start. Waiting for phone to connect.')
        phone_conn, phone_addr = server_for_phone.accept()

        reader = threading.Thread(target=reader_phone_loop, args=(phone_conn, model, poses, trans, unity_conn), daemon=True)
        reader.start()

    # align IMU to SMPL body frame
    input('Put imu 1 aligned with your body reference frame (x = Left, y = Up, z = Forward) and then press any key.')
    print('Keep for 3 seconds ...', end='')
    oris = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=40)[0][0]
    smpl2imu = quaternion_to_rotation_matrix(oris).view(3, 3).t()

    # bone and acceleration offset calibration
    input('\tFinish.\nWear all imus correctly and press any key.')
    for i in range(3, 0, -1):
        print('\rStand straight in T-pose and be ready. The calibration will begin after %d seconds.' % i, end='')
        time.sleep(1)
    print('\nStand straight in T-pose. Keep the pose for 3 seconds ...', end='')

    oris, accs = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=40)
    oris = quaternion_to_rotation_matrix(oris)
    device2bone = smpl2imu.matmul(oris).transpose(1, 2).matmul(torch.eye(3))
    acc_offsets = smpl2imu.matmul(accs.unsqueeze(-1))   # [num_imus, 3, 1], already in global inertial frame

    # start streaming data
    print('\tFinished Calibrating.\nEstimating poses. Press q to quit')
    imu_set.start_reading()

    running = True
    clock = Clock()
    is_recording = False
    record_buffer = None

    get_input_thread = threading.Thread(target=get_input)
    get_input_thread.setDaemon(True)
    get_input_thread.start()


    while running:
        # calibration
        clock.tick(datasets.fps)
        ori_raw, acc_raw = imu_set.get_current_buffer() # [buffer_len, 5, 4]
        
        if ori_raw.shape == (0,):
            print(ori_raw.shape)
            print(acc_raw.shape)
            continue
       
        ori_raw = quaternion_to_rotation_matrix(ori_raw).view(-1, n_imus, 3, 3)
        glb_acc = (smpl2imu.matmul(acc_raw.view(-1, n_imus, 3, 1)) - acc_offsets).view(-1, n_imus, 3)
        glb_ori = smpl2imu.matmul(ori_raw).matmul(device2bone)

        # normalization 
        _acc = glb_acc.view(-1, 5, 3)[:, [1, 4, 3, 0, 2]] / amass.acc_scale
        _ori = glb_ori.view(-1, 5, 3, 3)[:, [1, 4, 3, 0, 2]]
        acc = torch.zeros_like(_acc)
        ori = torch.zeros_like(_ori)

        # device combo
        combo = 'rw_rp'
        c = amass.combos[combo]

        if USE_PHONE_AS_WATCH:
            # set watch value to phone
            acc[:, [0]] = _acc[:, [3]]
            ori[:, [0]] = _ori[:, [3]]
        else:
            # filter and concat input
            acc[:, c] = _acc[:, c] 
            ori[:, c] = _ori[:, c]
        
        imu_input = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)
        # imu_input = torch.cat([_acc[:, c].flatten(1), _ori[:, c].flatten(1)], dim=1)
    
        # send input to iphone for on-device model prediction
        if args.on_device:
            imu_input, imu_shape = model.process_inputs(imu_input.squeeze(0))
            send_tensors([imu_input, imu_shape], phone_conn)
        else:

            # predict pose and translation
            with torch.no_grad():
                output = model.forward_online(imu_input.squeeze(0), [imu_input.shape[0]])
                pred_pose = output[0] # [24, 3, 3]
                pred_tran = output[2] # [3]
            
            # convert rotmatrix to axis angle
            pose = rotation_matrix_to_axis_angle(pred_pose.view(1, 216)).view(72)
            tran = pred_tran

        # # keep track of data
        # if args.save:
        #     accs.append(glb_acc)
        #     oris.append(glb_ori)
        #     raw_accs.append(acc_raw)
        #     raw_oris.append(ori_raw)
        #     poses.append(pred_pose)
        #     trans.append(pred_tran)

            # send pose
            if args.vis:
                s = ','.join(['%g' % v for v in pose]) + '#' + \
                    ','.join(['%g' % v for v in tran]) + '$'
                unity_conn.send(s.encode('utf8'))  
                
                if os.getenv("DEBUG") is not None:
                    print('\r', '(recording)' if is_recording else '', 'Sensor FPS:', imu_set.clock.get_fps(),
                            '\tOutput FPS:', clock.get_fps(), end='')

    # save data to file for viewer
    if args.save:
        data = {
            'raw_acc': torch.cat(raw_accs, dim=0),
            'raw_ori': torch.cat(raw_oris, dim=0),
            'acc': torch.cat(accs, dim=0),
            'ori': torch.cat(oris, dim=0),
            'pose': torch.cat(poses, dim=0),
            'tran': torch.cat(trans, dim=0),
            'calibration': {
                'smpl2imu': smpl2imu,
                'device2bone': device2bone
            }
        }
        torch.save(data, paths.dev_data / f'dev_{int(time.time())}.pt')
    
    if args.on_device:
        data = {'pose': torch.cat(poses, dim=0),
                'tran': torch.cat(trans, dim=0)}
        torch.save(data, f'phone_{int(time.time())}.pt')

    # clean up threads
    get_input_thread.join()
    imu_set.stop_reading()
    print('Finish.')
