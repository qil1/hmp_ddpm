import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.getcwd())
from utils import amass3d
from utils import opt


def draw_pic_single(color, mydata, full_path):
    mydata = mydata[:, [0, 2, 1]]

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.grid(False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-1000, 1000])
    ax.set_ylim3d([-1000, 1000])
    ax.set_zlim3d([-1000, 1000])

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([mydata[I[i], j], mydata[J[i], j]]) for j in range(3)]
        if color == 'gt':
            ax.plot(x, y, z, lw=3, color='#B4B4B4' if LR[i] == 0 else '#FA2828' if LR[i] == 2 else '#F57D7D')
        else:
            ax.plot(x, y, z, lw=3, color='#EED5B7' if LR[i] == 0 else '#EE82EE' if LR[i] == 2 else '#FFC0CB')

    # set grid invisible
    ax.grid(None)

    # set X、Y、Z background color white
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # set axis invisible
    ax.axis('off')

    plt.savefig(full_path, transparent=True, dpi=300)
    plt.close()


DRAW_LINE = [
    (2, 1, 0),
    (1, 4, 0),
    (4, 7, 0),
    (5, 10, 0),
    (10, 13, 0),
    (13, 15, 0),
    (15, 17, 0),

    (2, 0, 2),
    (0, 3, 2),
    (3, 6, 2),
    (5, 9, 2),
    (9, 12, 2),
    (12, 14, 2),
    (14, 16, 2),

    (2, 5, 1),
    (5, 8, 1),
    (8, 11, 1),
]
I, J, LR = [], [], []
for i in range(len(DRAW_LINE)):
    I.append(DRAW_LINE[i][0])
    J.append(DRAW_LINE[i][1])
    LR.append(DRAW_LINE[i][2])

# for i in range(35):
#     pose = dataset[2000][i][dataset.joint_used, :]
#     print(pose.shape)
#     print(pose)
#     # plot(i)
#     draw_pic_single('gt', pose, I, J, LR, i)
