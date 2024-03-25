import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


DRAW_LINE = [
    (0, 1, 0),
    (1, 4, 0),
    (4, 7, 0),
    (7, 10, 0),
    (13, 9, 0),
    (13, 16, 0),
    (16, 18, 0),
    (18, 20, 0),
    (20, 22, 0),
    (0, 2, 2), # 2
    (2, 5, 2),
    (5, 8, 2),
    (8, 11, 2),
    (14, 9, 2),
    (14, 17, 2),
    (17, 19, 2),
    (19, 21, 2),
    (21, 23, 2),
    (15, 12, 1),
    (12, 9, 1),
    (6, 3, 1),
    (6, 9, 1),
    (0, 3, 1)
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

    # ax.scatter(x, y, z, c='b')
    # for i in range(len(x)):
    #     ax.text(x[i], y[i], z[i], i, fontsize=2)

    # (250, 40, 40) #FA2828 red
    # (245, 125, 125) #F57D7D pink
    # (11, 11, 11) #0B0B0B black
    # (180, 180, 180) #B4B4B4 gray

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([mydata[I[i], j], mydata[J[i], j]]) for j in range(3)]
        # ax.plot(x, y, z, lw=2, c=color)
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