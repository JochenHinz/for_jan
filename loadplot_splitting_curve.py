import numpy as np
import scipy as sp
from problem_library import *
from matplotlib import pyplot
import os
import shapely.geometry as sh
from separator import *


def main(frompath = 'splitting_curve/smooth/', frompath_grid = 'xml/smooth/'):
    '''
        Load and plot splitting curves
        
        IMPORTANT: make sure that the value of ``a`` in separator snail is the same as in the main script
    '''
    
    ####### snail point cloud cusp points etc #######
       
    c_l, c_r = np.array([-13.1,0]), np.array([13.1,0])
    left = []
    with open('screwLeft.txt') as fl:
        for l in fl:
            row = l.split()
            left.append(np.array([float(row[i]) for i in range(2)])[:,None])
    left_pc = np.roll(np.concatenate(left, axis = 1),250, axis = 1)
    left_pc = ndspline(rep.discrete_length_param(left_pc), left_pc, k = 3)(np.linspace(0,1,1000))

    right = []
    with open('screwRight.txt') as fl:
        for l in fl:
            row = l.split()
            right.append(np.array([float(row[i]) for i in range(2)])[:,None])
    right_pc = np.roll(np.concatenate(right, axis = 1),250, axis = 1)
    right_pc = ndspline(rep.discrete_length_param(right_pc), right_pc, k = 3)(np.linspace(0,1,1000))
    
    c1,c2 = cusp(16,16,26.2)
    c1,c2 = [c - np.array([13.1,0]) for c in [c1,c2]]
    
    ###### functions to generate grid object corresponding to angle ######
    
    def rotate_snails(x,y,angle):
        x_, y_ = [rotate(x, angle, center = c_l), rotate(y, -angle, center = c_r)]
        i,j = [np.argmax(norm(pc)) for pc in [x_,y_]]
        x_ = np.roll(x_, -i, axis = 1)
        y_ = np.roll(y_, -j, axis = 1)
        return x_, y_


    def separator_snail(angle_, a = 4/5):
        ''' 
            generate the bottom, left, right and top point cloud of the snail separator 
            ``angle_`` denotes the rotational angle of the rotors and ``a`` tunes the contraction / expansion
            of the x-axis of the ellipsiod used in the separator 
        '''
        c_1, c_0 = [i - np.array([13.1,0]) for i in cusp(16,16,26.2)]
        theta_0, theta_1 = [theta(c[:,None], center = c_l)[0] for c in [c_1,c_0]]   ## CUSP angles from [-13.1,0]
        theta_2, theta_3 = [theta(c[:,None], center = c_r)[0] for c in [c_1,c_0]]    ## CUSP angles from [13.1,0]


        male, female = left_pc.copy(), right_pc.copy()

        cusp_center = (c_1 + c_0)/2.0
        circle = circle_point(np.linalg.norm((c_1-c_0)/2),  - np.pi + np.linspace(0,2*np.pi,1001), a = a) + cusp_center[:,None]
        cuspolygon = sh.Polygon([p for p in circle.T])

        def cut_paste(x_, y_):
            lists = [[sh.Point(P).within(cuspolygon) for P in z.T] for z in [x_,y_]]  ## list of indices inside and outside of CUSP
            lists = [False_True(l) for l in lists]
            list_0, list_1 = [lists[i][np.argmax([np.diff(j) for j in lists[i]])] for i in range(2)]
            i_0, i_1 = [np.argmin(distance(circle,j[:,None])) for j in [x_.T[list_0[0]].T,y_.T[list_1[0]].T]]
            i_3, i_2 = [np.argmin(distance(circle,j[:,None])) for j in [x_.T[list_0[1]].T,y_.T[list_1[1]].T]]
            return circle.T[i_0:i_1+1].T, circle.T[i_2:i_3+1].T, x_.T[list_0[0]:list_0[1]+1].T, y_.T[list_1[0]:list_1[1]+1].T

        male_, female_ = rotate_snails(male, female, angle_)
        bottom, top, left, right = cut_paste(male_,female_)
        top = top.T[::-1].T
        left = np.concatenate([bottom[:,0][:,None], left.T[1:-1].T, top[:,0][:,None]], axis = 1)
        right = np.concatenate([bottom[:,-1][:,None], right.T[1:-1].T, top[:,-1][:,None]], axis = 1)

        return bottom, right, top, left
    
    
    c1,c2 = cusp(16,16,26.2)
    c1,c2 = [c - np.array([13.1,0]) for c in [c1,c2]]
    
    names = sorted(os.listdir(frompath))
    
    for i in range(len(names)):
        filename = names[i]
        log.info('Plotting splitting-curve corresponding to ' + filename)
        func = ut.tensor_grid_object.fromxml(frompath + filename)
        go = ut.tensor_grid_object.fromxml(frompath_grid + filename)
        plt.scatter(*func.toscipy(np.linspace(0,1,1000)), s = 0.5)
        plt.scatter(*np.hstack(separator_snail(float(filename[6:16]))), s = 0.5)
        plt.scatter(*c1)
        plt.scatter(*c2)
        
        plt.show()
                    
        
   
        
        
        
if __name__ == '__main__':
    main()

    
    
    