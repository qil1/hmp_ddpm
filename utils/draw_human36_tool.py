import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


DRAW_LINE = [
    (0, 1, 0),
    (1, 2, 0),
    (2, 3, 0),
    (3, 4, 0),
    (3, 5, 0),
    (0, 12, 1),
    (12, 13, 1),
    (13, 14, 1),
    (14, 15, 1),
    (13, 17, 2),
    (17, 18, 2),
    (18, 19, 2),
    (19, 21, 2),
    (19, 22, 2),
    (13, 25, 0),
    (25, 26, 0),
    (26, 27, 0),
    (27, 29, 0),
    (27, 30, 0),
    (0, 6, 2),
    (6, 7, 2),
    (7, 8, 2),
    (8, 9, 2),
    (8, 10, 2)
]
I, J, LR = [], [], []
for i in range(len(DRAW_LINE)):
    I.append(DRAW_LINE[i][0])
    J.append(DRAW_LINE[i][1])
    LR.append(DRAW_LINE[i][2])


def draw_pic_single(color, mydata, full_path):
    # num_joints, 3  # x,y,z dimension
    # I
    # J
    # LR

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

    # for i in range(23):
    #     x, z, y = pose[i]
    #     ax.scatter(x, y, z, c='b', s=2)
    #     ax.text(x, y, z, i, fontsize=4)

    # (250, 40, 40) #FA2828 red
    # (245, 125, 125) #F57D7D pink
    # (11, 11, 11) #0B0B0B black
    # (180, 180, 180) #B4B4B4 gray

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
