#!/usr/bin/python

import numpy as np
import time
import random
import os
import math

#import cec2013lsgo
#from cec2013lsgo.cec2013 import Benchmark
import matplotlib

from optproblems import *
from optproblems.continuous import *
#matplotlib.use('PS')
import matplotlib.pyplot as plt


# helper functions
def Ackley(x):
    firstSum = 0.0
    secondSum = 0.0
    chromosome = [xi - 2.317 for xi in x]
    for c in chromosome:
        firstSum += c**2.0
        secondSum += math.cos(2.0*math.pi*c)
    n = float(len(x))
    return -20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e


# Large Scale Problem
class LSProblem():
    '''
    A class for large-scale problems
    '''
    def __init__(self, id = 0, isBenchmark = True, dim = None, dim_e = None):
        '''
        id : benchmark function id
        isBenchmark : flag to use functions from the cec2013lsgo package if True, else
        use optproblems
        dim : extrinsic dimension
        dim_e : effective dimension where dim_e << dim
        All problems are in a minimization setting.
        '''
        if isBenchmark:
            assert id < 15  # "ID should be in [0-14]")
            bench = Benchmark()
            self.fctn = bench.get_function(id)
            self.info = bench.get_info(id)
        else:
            # use scalable benchmark functions
            # from https://ls11-www.cs.tu-dortmund.de/people/swessing/optproblems/doc/index.html
            assert id < 9
            if dim is None:
                dim = 100
            if dim_e is None:
                dim_e = dim
            bounds = ([-5.0] * dim_e, [5.0] * dim_e)
            preprocessor = BoundConstraintsChecker(bounds)
            if id == 0:
                fctn = LunacekTwoRastrigins(num_variables=dim_e, phenome_preprocessor=preprocessor)
            elif id == 1:
                fctn = lambda x: sum(np.array([10**5*(x[idx]-2.317)**2 if idx == 0 else (x[idx]-2.317)**2 for idx in xrange(len(x)) ]))
            elif id == 2:
                bounds = ([-np.pi] * dim_e, [np.pi] * dim_e)
                preprocessor = BoundConstraintsChecker(bounds)
                fctn = FletcherPowell(num_variables=dim_e, phenome_preprocessor=preprocessor)
            elif id == 3:
                bounds = ([-500.0] * dim_e, [500.0] * dim_e)
                preprocessor = BoundConstraintsChecker(bounds)
                fctn = Schwefel(num_variables=dim_e, phenome_preprocessor=preprocessor)
            elif id == 4:
                bounds = ([0.25] * dim_e, [10.0] * dim_e)
                preprocessor = BoundConstraintsChecker(bounds)
                fctn = Vincent(num_variables=dim_e, phenome_preprocessor=preprocessor)
            elif id == 5:
                fctn = LunacekTwoSpheres(num_variables=dim_e, phenome_preprocessor=preprocessor)
            elif id == 6:
                fctn = Rosenbrock(num_variables=dim_e, phenome_preprocessor=preprocessor)
            elif id == 7:
                # spherical easy to go
                x_opt = 10.0 * np.random.random(dim_e) - 5.0
                fctn = lambda x: sum((np.array(x)-x_opt)**2)
                #fctn = lambda x: sum((np.array(x)-2.317)**2)
            elif id == 8:
                fctn = lambda x: Ackley(x)
            else:
                raise Exception()
            #y = np.array([-3.2]*dim)
            #self.fctn = lambda x: sum((x[0:10]-y[0:10])**2)
            #x = np.array([0.22]*dim_e)
            #print fctn(x)
            #print "opt sol :", fctn.get_optimal_solutions(max_number=1)[-1].phenome[0:5]
            #print "opt val:", fctn(fctn.get_optimal_solutions(max_number=1)[-1].phenome)
            eff_dims = np.random.choice(dim, dim_e)
            self.fctn = lambda x : fctn([x[eff_dim] for eff_dim in eff_dims])
            self.info= {'dimension': dim, 'lower':bounds[0][0], 'upper': bounds[1][0]}
        self.best_val = float("inf")
        self.num_evals = 0
        self.profiles = []
        self.id = id
        self.isRecording = False

    def init_profile(self):
        '''
        initialize a new recorder
        '''
        self.isRecording = True
        self.best_val = float("inf")
        self.num_evals = 0
        self.profiles.append([])

    def init_best_val(self):
        '''
        reset best val to infinity
        '''
        self.best_val = float("inf")

    def get_dimension(self):
        return self.info['dimension']

    def get_lower_range(self):
        return self.info['lower']

    def get_upper_range(self):
        return self.info['upper']

    def get_best_val(self):
        return self.best_val

    def f(self, x):
        '''
        returns the value of f at x
        x: numpy array normalized to [0,1]
        '''
        # evaluate
        #y = self.fctn(self.info['lower'] + x * (self.info['upper'] - self.info['lower']))
        #print x
        y = self.fctn(x)
        #print y
        self.num_evals += 1
        # record if enabled
        if self.isRecording and y < self.best_val:
            self.best_val = y
            #print y
            self.profiles[-1].append({'at': self.num_evals, 'val': y})
        elif y < self.best_val:
            self.best_val = y
        return y

    def profile2text(self, xlimit = None, legend = None):
        '''
        print profile to text in a tabular format
        '''
        if xlimit is None:
            xlimit = max([profile[-1]['at'] for profile in self.profiles])
        if legend is None:
            legend = ["alg-%d" % i for i in xrange(len(self.profiles))]
        dir_name = "./fig"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        for idx, profile in enumerate(self.profiles):
            x = [data['at']  for data in profile if data['at'] <= xlimit]
            y = [data['val'] for data in profile if data['at'] <= xlimit]
            if not xlimit == x[-1]:
                x.append(xlimit)
                y.append(y[-1])
            fname = dir_name + "/" + legend[idx] + ".table"
            data = np.vstack((np.array(x),np.array(y))).T
            np.savetxt(fname, data, comments='',header='x\ty',delimiter='\t',fmt='%d\t%.15e')

    def plot_profiles(self, xlimit = None, legend = None):
        '''
        plot profiles
        '''
        if xlimit is None:
            xlimit = max([profile[-1]['at'] for profile in self.profiles])
        for idx, profile in enumerate(self.profiles):
            x = [data['at']  for data in profile if data['at'] <= xlimit]
            if self.id != 3:
                y = [np.log(data['val']+1) for data in profile if data['at'] <= xlimit]
            else:
                y = [data['val']+1 for data in profile if data['at'] <= xlimit]
            x.append(xlimit)
            y.append(y[-1])
            plt.step(x, y, label='Algorithm %d ' % self.id, where = 'post')
        plt.ylabel('$f_{best}$')
        plt.xlabel(r'$f-evals$')
        if legend is not None:
            plt.legend(legend)
        plt.title('problem %d'% self.id)
        #plt.show()

    def get_profiles(self):
        return self.profiles




def main():
  print "I am a routine to be used by other codes, below is some testing"
  start_time = time.time()
  print "Testing some benchmark problems"
  problem = LSProblem(8, isBenchmark = False, dim_e = 10)
  dim = problem.get_dimension()
  num_runs = 1
  num_evals = 10
  for run in xrange(num_runs):
      problem.init_profile()
      print "Run: %d" % run
      for feval in xrange(num_evals):
          pt = random.random()
          pt = (problem.get_upper_range() - problem.get_lower_range()) * np.random.random((dim,)) + problem.get_lower_range() #np.array([pt]*dim)
          print "\t f(x)=%f" % (problem.f(pt))
  print "Saving profiles to text files"
  problem.profile2text()
  #print "Plotting the profile"
  #problem.plot_profiles()
  end_time = time.time()
  print "It took %f seconds." % (end_time - start_time)


if __name__ == '__main__':
  main()
