#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from io import BytesIO
from PIL import Image

def IK(x_target, y_target, z_target):

    l1_sq = l1*l1
    l2_sq = l2*l2
    l3_sq = l3*l3

    theta1 = np.arctan2(y_target-l0*np.sin(theta0), x_target-l0*np.cos(theta0)) - theta0
    theta2 = 0.0
    theta3 = 0.0

    D = np.sqrt( np.square(x_target-l0*np.cos(theta0)) + np.square(y_target-l0*np.sin(theta0)) )

    if (D-l1)>=0:
        d = D - l1
        R_sq = np.square(d) + np.square(z_target) 
        R = np.sqrt( R_sq )
        alpha1 = np.arccos( -1*(z_target)/R)
        alpha2 = np.arccos( (l2_sq - l3_sq + R_sq)/(2*l2*R) )
        theta2 = alpha1 + alpha2 - np.pi/2

    if (D-l1)<0:
        d = l1 - D
        R_sq = np.square(d) + np.square(z_target) 
        R = np.sqrt( R_sq )
        alpha1 = np.arccos(d/R)
        alpha2 = np.arccos( (l2_sq - l3_sq + R_sq)/(2*l2*R) )
        theta2 = alpha1 + alpha2 - np.pi

    theta3 = np.arccos( (l2_sq + l3_sq - np.square(d) - np.square(z_target) ) / (2*l2*l3) ) - np.pi

    print('IK given (x y z) [m]                   : ', x_target, y_target, z_target)
    print('IK solved (theta1 theta2 theta3) [rad] : ', np.rad2deg(theta1), np.rad2deg(theta2), np.rad2deg(theta3))
    print("-----------------------")

    return theta1, theta2, theta3


def FK(theta1, theta2, theta3):

    T_g0 = np.array([ [np.cos(theta0), -1*np.sin(theta0),  0,  0],
                      [np.sin(theta0),  np.cos(theta0),    0,  0],
                      [0             ,  0             ,    1,  0 ],
                      [0             ,  0             ,    0,  1 ] ])

    T_01 = np.array([ [np.cos(theta1), -1*np.sin(theta1),  0,  l0],
                      [np.sin(theta1),  np.cos(theta1),    0,  0],
                      [0             ,  0             ,    1,  0 ],
                      [0             ,  0             ,    0,  1 ] ])

    T_12 = np.array([ [np.cos(theta2),  -1*np.sin(theta2), 0,   l1 ],
                      [0             ,   0               , -1,  0 ],
                      [np.sin(theta2),   np.cos(theta2),   0,   0 ],
                      [0             ,   0               , 0,   1 ] ])

    T_23 = np.array([ [np.cos(theta3), -1*np.sin(theta3),  0,  l2],
                      [np.sin(theta3),  np.cos(theta3),    0,  0],
                      [0             ,  0             ,    1,  0 ],
                      [0             ,  0             ,    0,  1 ] ])

    T_34 = np.array([ [1,  0,  0, l3],
                      [0,  1,  0,  0],
                      [0,  0,  1,  0],
                      [0,  0,  0,  1] ])

    T_g1 = T_g0@T_01
    T_g2 = T_g1@T_12
    T_g3 = T_g2@T_23
    T_g4 = T_g3@T_34

    p0 = np.array([ [0], 
                    [0], 
                    [0],
                    [1] ])

    p1 = T_g1@p0
    #print('p1=T_g0*T_01*p0\n', p1)
    p2 = T_g2@p0
    #print('p2=T_g0*T_01*T_12*p0\n', p2)
    p3 = T_g3@p0
    #print('p3=T_g0*T_01*T_12*T_23*p0\n',p3)
    p4 = T_g4@p0
    #print('p4=T_g0*T_01*T_12*T_23*T_34*\n',p4)

    x = [p0[0][0],  p1[0][0], p2[0][0], p3[0][0], p4[0][0]]
    y = [p0[1][0],  p1[1][0], p2[1][0], p3[1][0], p4[1][0]]
    z = [p0[2][0],  p1[2][0], p2[2][0], p3[2][0], p4[2][0]]

    print('FK given  (theta1 theta2 theta3) [rad] : ', np.rad2deg(theta1), np.rad2deg(theta2), np.rad2deg(theta3))
    print('FK solved (x y z) [m]                  : ', x[4], y[4], z[4]) 
    print("-----------------------")
 
    return x, y, z


def render_frame(angle):

    x, y, z = FK(theta1, theta2, theta3)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim([0, 0.3])
    ax.set_ylim([0, 0.3])
    ax.set_zlim([-0.15, 0.15])
    ax.set_box_aspect([1,1,1])

    ax.plot(x, y, z, "o-", color="#00aa00", ms=4, mew=0.5)
    ax.view_init(30, angle)
    plt.close()

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    buf = BytesIO()
    fig.savefig(buf, bbox_inches='tight', pad_inches=0.0)

    return Image.open(buf)


theta0 = np.pi/2 #[rad]

x_ref = 0.203 #0.25 #0.2536 #[m]
y_ref = 0.2536 #0.05 # 0.2036   #[m]
z_ref = -0.0424 #0.15 #-0.0424 #[m]

l0 = 0.05  #[m]
l1 = 0.09  #[m]
l2 = 0.11  #[m]
l3 = 0.17  #[m]

en_animation = False

theta1, theta2, theta3 = IK(x_ref, y_ref, z_ref)
#a, b, c = IK(x[4], y[4], z[4])


if en_animation:

    images = [render_frame(angle) for angle in range(180)]
    images[0].save('output.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

else:

    #Forward Kinematick for Draw

    """ #FOR DEBUG
    theta1 = np.pi/4 
    theta2 = np.pi/4
    theta3 = -np.pi/2
    """

    x, y, z = FK(theta1, theta2, theta3)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim([0, 0.3])
    ax.set_ylim([0, 0.3])
    ax.set_zlim([-0.15, 0.15])
    ax.set_box_aspect([1,1,1])

    ax.plot(x, y, z, "o-", color="#00aa00", ms=4, mew=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

