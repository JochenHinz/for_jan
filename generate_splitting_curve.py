import numpy as np
import scipy as sp
from problem_library import *
import os


def main(frompath = 'xml/smooth/', topath = 'splitting_curve/smooth/'):
    '''
        Generate univariate splitting curves from bivariate grid objects and save them as .xml
    '''
    
    
    c1,c2 = cusp(16,16,26.2)
    c1,c2 = [c - np.array([13.1,0]) for c in [c1,c2]]
    
    for filename in os.listdir(frompath):
        log.info('Generating splitting-curve corresponding to ' + filename)
        go = ut.tensor_grid_object.fromxml(frompath + filename)
        pc = go.splitting_curve(c2,c1)
        sgo = ut.tensor_grid_object.from_pointcloud(pc)
        sgo.toxml(topath+filename[:-4])
        
   
        
        
        
if __name__ == '__main__':
    main()

    
    
    