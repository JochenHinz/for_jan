import numpy as np
import scipy as sp
from problem_library import *
import shapely.geometry as sh
import reparam as rep
from separator import *
import preprocessor as prep
import utilities as ut


def main(angle = np.linspace(0,np.pi,200), extrapolate = True, repair_defects = True, save = False):
    
    
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
        rotate_all = lambda x,y,angle: [pl.rotate(x, angle, center = c_l), pl.rotate(y, -angle, center = c_r)]
        cusp_center = (c_1 + c_0)/2.0
        circle = pl.circle_point(np.linalg.norm((c_1-c_0)/2),  - np.pi + np.linspace(0,2*np.pi,1001), a = a) + cusp_center[:,None]
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


    def separator_go(go, angle_, reparam = False, a= 4/5):
        bottom, right, top, left = separator_snail(angle_, a = a)
        minimum = np.min(distance(left,right))
        if reparam and minimum < 0.5:
            left_verts, right_verts = rep.constrained_arc_length(left, right, fac = 1, smooth = True)
        else:
            left_verts, right_verts = [rep.discrete_length_param(j) for j in [left,right]]
        bottom_verts, top_verts = [rep.discrete_length_param(j) for j in [bottom,top]]
        goal_boundaries = dict(left = lambda g: ut.interpolated_univariate_spline(left_verts, left, g[1]))
        goal_boundaries.update(right = lambda g: ut.interpolated_univariate_spline(right_verts, right, g[1]))
        goal_boundaries.update(bottom = lambda g: ut.interpolated_univariate_spline(bottom_verts, bottom, g[0]))
        goal_boundaries.update(top = lambda g: ut.interpolated_univariate_spline(top_verts, top, g[0]))

        corners = {(0,0): bottom.T[0], (1,0): bottom.T[-1], (0,1): top.T[0], (1,1): top.T[-1]}
        return go, goal_boundaries, corners
    
    
    ###### generate empty grid object ######
    
    n,m = 13,36  ## amount of elements
    p = 3  ## order (same in both directions, can be changed)
    knots = np.prod([ut.nonuniform_kv(p, knotvalues = np.linspace(0,1,j)) for j in [n,m]])  ## create tensor knot-vector
    go_ = ut.tensor_grid_object(knots = knots)  ## empty grid object
    
    ###### generate geometries ######
    
    extp_grids = [] ## previous grid objects used for extrapolation
    extp_angles = []  ## corresponding angles
    
    for ang in angle:
        log.info('Computing grid corresponding to angle %.5f' %ang)
        if len(extp_grids) == 7:  ##  remove grids that are not needed for <= 5-th order extrapolation
            del extp_angles[0]
            del extp_grids[0]
        go = go_.empty_copy()  ## empty grid object
        go, goal_boundaries, corners = separator_go(go,ang)  ## compute goal_boundaries and corners corresponding to angle
        go.set_cons(goal_boundaries,corners)  ## set constraints resulting from goal_boundaries and corners

        
        ###### generate initial guess ######
        
        if extrapolate and len(extp_grids) > 0:  ## initial guess via extrapolation
            ''' 
                the vector ``s`` corresponds to the weights of the mapping with respect to the basis utilized
                here it is set to a constrained extrapolation of the weights corresponding to the previous
                grids (constraining the boundary control points to the least-squares projection of the basis 
                onto the point cloud).
            '''
            try: ## it'll only work if all grids use the same knot-vector, implementation for arbitrary knots forthcoming
                assert all([go._knots == grid.knots for grids in extp_grids])
                go.s = go.cons | go.grid_interpolation(extp_angles, extp_grids)(ang)
            except:
                go.set_initial_guess(goal_boundaries, corners)
        else:  ## initial guess via transfinite-interpolation
            go.set_initial_guess(goal_boundaries, corners)
            
        ###### solve, detect/repair defects & save ######
            
        go.quick_solve()
        
        extp_grids.append(go)
        extp_angles.append(ang)
        
        if repair_defects:
            log.info('Checking for defects')
            it = 1
            while True:
                if go.detect_defects_discrete(ischeme = 20, thresh_ratio = 1e-3) or it > 2:  
                    break  ## no defect detected or more than ``it`` correction attempts
                log.warning('Warning, defect detected in converged mapping at angle %.3f' %ang)
                go = go.ref(1)
                go.quick_solve()
                it += 1
        
        if save:
            go.toxml('xml/smooth/theta_%.8f_smooth' %ang + '_' + str(go.ndims[0]) + '_' + str(go.ndims[1]))
        
        
   ###### END ######
        
        
        
if __name__ == '__main__':
    main()

    
    
    
