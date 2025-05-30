import os
import time
import socket
import struct
import threading
import torch
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from pygame.time import Clock
import pickle
import time

from articulate.math import *
from mobileposer.models import *
from mobileposer.utils.model_utils import *
from mobileposer.config import *

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--file", required=True, type=str)
    args = parser.parse_args()

    saved_tensors = torch.load(args.file)

    poses = saved_tensors["pose"].reshape((saved_tensors["pose"].shape[0]//72, 72)) # [length, 24, 9]
    trans = saved_tensors["tran"].reshape((saved_tensors["tran"].shape[0]//3, 3)) # [length, 3]

    # setup Unity server for visualization
    server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_for_unity.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    server_for_unity.bind(('0.0.0.0', 8889))
    server_for_unity.listen(1)
    print('Server start. Waiting for unity3d to connect.')
    conn, addr = server_for_unity.accept()

    clock = Clock()
    is_recording = False
    record_buffer = None

    time.sleep(5)

    for i in range(poses.shape[0]):
        pose = poses[i]
        tran = trans[i]
        # calibration
        clock.tick(datasets.fps)

        # send pose
        s = ','.join(['%g' % v for v in pose]) + '#' + \
            ','.join(['%g' % v for v in tran]) + '$'
        conn.send(s.encode('utf8'))  

    print('Finish.')
