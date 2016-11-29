#--------------------------------------------------------------------------
#Copyright (c) 2016 by Abdullah Al-Dujaili
#
#This file is part of EmbeddedHunter - large-scale black-box solver
#EmbeddedHunter is free software: you can redistribute it and/or modify it under
#the terms of the GNU General Public License as published by the Free
#Software Foundation, either version 3 of the License, or (at your option)
#any later version.
#
#EmbeddedHunter is distributed in the hope that it will be useful, but WITHOUT
#ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
#more details.
#
#You should have received a copy of the GNU General Public License along
#with EmbeddedHunter.  If not, see <http://www.gnu.org/licenses/>.
#--------------------------------------------------------------------------
#Author information
#Abdullah Al-Dujaili
#ash.aldujaili@gmail.com
#--------------------------------------------------------------------------


# BENCHMARKING
from Benchmark import LSProblem
# Deep Larning

__author__ = "Abdullah Al-Dujaili"
# Algorithms
from Algorithms import EmbeddedHunter
from Algorithms import RESOO
from Algorithms import SRESOO
# Other stuff
import time
import pickle
import os
import numpy as np
import random
import sys


def experiments_to_tables(fname):
    '''
    take a list experiment ds and converts into tables by reporting the mean of the results
    '''
    # read the dictionary
    if type(fname) == str:
        with open(fname,'rb') as handle:
            exprmnt_result = pickle.load(handle)
    else:
        exprmnt_result = fname
    # print out the experiements one by one in this fashion
    # a file for each problem-experiment pair
    dir_name = "./fig"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    #for exid,exprmnt in enumerate(exprmnts):
    exid = exprmnt_result['id']
    # set the header
    header = 'at'
    fmt = '%d'
    for alg in exprmnt_result['algs']:
        header = header + '\t' + alg
        fmt = fmt + '\t%.15e'
    for pid in exprmnt_result['data'].keys():
        # set the data holder
        data = np.array(sorted(exprmnt_result['at']))
        for alg in exprmnt_result['algs']:
            alg_data = np.array([np.mean(exprmnt_result['data'][pid][alg][step]) for step in sorted(exprmnt_result['at'])])
            data = np.vstack((data, alg_data))
        fname = dir_name + "/" + "exp-%d-pid-%d"   + ".table"
        np.savetxt(fname % (exid, pid) , data.T, comments='',header=header,delimiter='\t',fmt=fmt)

def main(exid = 0):
    '''
    A routine for AAAI'16 submitted paper:
    Bandits Attack Large-Scale Optimization's scalable benchmark functions.
    '''

    if os.path.exists('experiemnt-%d.pickle' % exid):
        print "Experiment file exists, delete the experiment file *.pickle if you want to redo the experiment, it will take around a week"
        experiments_to_tables('experiemnt-%d.pickle' % exid)
        return
    start_time = time.time()
    # set a seed
    random.seed(123)
    np.random.seed(123)
    #------------------------------------------------
    print "Empirical Performance Evaluation on Scalable Black-Box Problems"
    # set up the benchmark settings
    num_runs = 20
    PROBLEMS = [1,2,6,8]
    NUM_EMBEDS =  [1, 2, 5, 8, 10, 20]
    E_DIMS = [2, 5, 10, 20, 50, 75]
    KE_DIMS = [2, 5, 8, 25, 75, 250]
    DIMS = [ int(d) for d in [1e2,5e2, 1e3, 1e4, 5e4, 1e5]]
    BUDGETS = [ int(v) for v in [1e1,5e1,1e2, 1e3, 1e4, 5e4, 1e5]]
    ALGS = [RESOO,  SRESOO, EmbeddedHunter]
    ALG_NAMES = [alg.__name__ for alg in ALGS]
    # ========================================================================================================
    if exid == 0:
        #  On the number of random embeddings:
        print "-On the number of random embeddings"
        n = int(1e4)
        v = int(1e4)
        n_e = int(1e1)
        exprmnt_result={'id':exid,'name':'number-of-embeddings', \
                            'settings':{'n':n,'v':v,'n_e':n_e,'num_runs':num_runs, 'problems':PROBLEMS},\
                            'algs': ALG_NAMES, \
                            'at': NUM_EMBEDS, \
                            'data':{pid:{alg:{m:[] for m in NUM_EMBEDS} for alg in ALG_NAMES} for pid in PROBLEMS}
                            }
        for m in NUM_EMBEDS:
            for pid in PROBLEMS:
                print "problem", pid,":",
                problem = LSProblem(pid, isBenchmark = False, dim = n, dim_e = n_e) # False for optproblems
                # the problem space is rescaled to [-1,1] in line with the assumption made in RE literature
                fctn = lambda x: problem.f((problem.get_upper_range() - problem.get_lower_range()) * (x + 1.0) / 2.+ problem.get_lower_range())
                for algid, alg in enumerate(ALGS):
                    print  ALG_NAMES[algid],
                    for rid in xrange(num_runs):
                        problem.init_best_val()
                        alg(n, fctn, v, M = m, n_e = n_e)
                        exprmnt_result['data'][pid][ALG_NAMES[algid]][m].append(problem.get_best_val())
                print
        # ========================================================================================================
    elif exid == 1:
        # ========================================================================================================
        #  On the effective dimension:
        print
        print "-On the effective dimension"
        n = int(1e4)
        v = int(1e4)
        m = 5
        exprmnt_result={'id':exid,'name':'effective-dimension', \
                            'settings':{'n':n,'v':v,'m':m,'num_runs':num_runs, 'problems':PROBLEMS},\
                            'algs': ALG_NAMES, \
                            'at': E_DIMS, \
                            'data':{pid:{alg:{n_e:[] for n_e in E_DIMS} for alg in ALG_NAMES} for pid in PROBLEMS}
                            }
        for n_e in E_DIMS:
            for pid in PROBLEMS:
                print "problem", pid,":",
                problem = LSProblem(pid, isBenchmark = False, dim = n, dim_e = n_e) # False for optproblems
                # the problem space is rescaled to [-1,1] in line with the assumption made in RE literature
                fctn = lambda x: problem.f((problem.get_upper_range() - problem.get_lower_range()) * (x + 1.0) / 2.+ problem.get_lower_range())
                for algid, alg in enumerate(ALGS):
                    print  ALG_NAMES[algid],
                    for rid in xrange(num_runs):
                        problem.init_best_val()
                        alg(n, fctn, v, M = m, n_e = n_e)
                        exprmnt_result['data'][pid][ALG_NAMES[algid]][n_e].append(problem.get_best_val())
                print
        # ========================================================================================================
    elif exid == 2:
        # ========================================================================================================
        #  On Scalability
        print
        print "-On scalability"
        v = int(1e4)
        m = 5
        n_e = 10
        exprmnt_result={'id':exid,'name':'scalability', \
                            'settings':{'n_e':n_e,'v':v,'m':m,'num_runs':num_runs, 'problems':PROBLEMS},\
                            'algs': ALG_NAMES, \
                            'at': DIMS, \
                            'data':{pid:{alg:{n:[] for n in DIMS} for alg in ALG_NAMES} for pid in PROBLEMS}
                            }
        for n in DIMS:
            for pid in PROBLEMS:
                print "problem", pid,":",
                problem = LSProblem(pid, isBenchmark = False, dim = n, dim_e = n_e) # False for optproblems
                # the problem space is rescaled to [-1,1] in line with the assumption made in RE literature
                fctn = lambda x: problem.f((problem.get_upper_range() - problem.get_lower_range()) * (x + 1.0) / 2.+ problem.get_lower_range())
                for algid, alg in enumerate(ALGS):
                    print  ALG_NAMES[algid],
                    for rid in xrange(num_runs):
                        problem.init_best_val()
                        alg(n, fctn, v, M = m, n_e = n_e)
                        exprmnt_result['data'][pid][ALG_NAMES[algid]][n].append(problem.get_best_val())
                print
        # ========================================================================================================
    elif exid == 3:
        # ========================================================================================================
        #  On Convergence Rate
        print
        print "-On convergence"
        m = 5
        n_e = 10
        n = int(1e4)
        v = max(BUDGETS)
        exprmnt_result={'id':exid,'name':'convergence', \
                            'settings':{'n':n,'n_e':n_e,'m':m,'num_runs':num_runs, 'problems':PROBLEMS},\
                            'algs': ALG_NAMES, \
                            'at': BUDGETS, \
                            'data':{pid:{alg:{v:[] for v in BUDGETS} for alg in ALG_NAMES} for pid in PROBLEMS}
                            }
        #for v in BUDGETS:
        # different from other experiments run by the maximum budget and ping their results
        # at other budgets
        for pid in PROBLEMS:
            print "problem", pid,":",
            problem = LSProblem(pid, isBenchmark = False, dim = n, dim_e = n_e) # False for optproblems
            # the problem space is rescaled to [-1,1] in line with the assumption made in RE literature
            fctn = lambda x: problem.f((problem.get_upper_range() - problem.get_lower_range()) * (x + 1.0) / 2.+ problem.get_lower_range())
            for algid, alg in enumerate(ALGS):
                print  ALG_NAMES[algid],
                for rid in xrange(num_runs):
                    #problem.init_best_val()
                    problem.profiles = []
                    problem.init_profile()
                    alg(n, fctn, v, M = m, n_e = n_e)
                    #exprmnt_result['data'][pid][ALG_NAMES[algid]][v].append(problem.get_best_val())
                    #process the profile and record the best values at v in BUDGETS
                    for v_idx, val in enumerate(BUDGETS):
                        exprmnt_result['data'][pid][ALG_NAMES[algid]][BUDGETS[v_idx]].append(0.0)
                        for item in problem.profiles[-1]:
                            if item['at'] < BUDGETS[v_idx]:
                                exprmnt_result['data'][pid][ALG_NAMES[algid]][BUDGETS[v_idx]][-1] = item['val']
                                continue
                            elif item['at'] == BUDGETS[v_idx]:
                                exprmnt_result['data'][pid][ALG_NAMES[algid]][BUDGETS[v_idx]][-1]= item['val']
                                break
                            else:
                                exprmnt_result['data'][pid][ALG_NAMES[algid]][BUDGETS[v_idx]][-1] = item['val']
                                break
            print
        # ========================================================================================================
    elif exid == 4:
        #  On the effective dimension:
        print
        print "-On the knowledge of  effective dimension"
        n = int(1e4)
        v = int(1e4)
        m = 5
        alg_n_e = int(1e1)
        exprmnt_result={'id':exid,'name':'effective-dimension-knowledge', \
                            'settings':{'n':n,'v':v,'m':m,'num_runs':num_runs, 'problems':PROBLEMS},\
                            'algs': ALG_NAMES, \
                            'at': KE_DIMS, \
                            'data':{pid:{alg:{n_e:[] for n_e in KE_DIMS} for alg in ALG_NAMES} for pid in PROBLEMS}
                            }
        for n_e in KE_DIMS:
            for pid in PROBLEMS:
                print "problem", pid,":",
                problem = LSProblem(pid, isBenchmark = False, dim = n, dim_e = n_e) # False for optproblems
                # the problem space is rescaled to [-1,1] in line with the assumption made in RE literature
                fctn = lambda x: problem.f((problem.get_upper_range() - problem.get_lower_range()) * (x + 1.0) / 2.+ problem.get_lower_range())
                for algid, alg in enumerate(ALGS):
                    print  ALG_NAMES[algid],
                    for rid in xrange(num_runs):
                        problem.init_best_val()
                        alg(n, fctn, v, M = m, n_e = alg_n_e)
                        exprmnt_result['data'][pid][ALG_NAMES[algid]][n_e].append(problem.get_best_val())
                print
    else:
        print "no such experiment"
        return
    # save data
    print exprmnt_result
    with open('experiemnt-%d.pickle' % exid,'wb') as handle:
        pickle.dump(exprmnt_result, handle)
    # write data to tables: enable this later
    experiments_to_tables('experiemnt-%d.pickle' % exid)
    #------------------------------------------------
    end_time = time.time()
    print "It took %f seconds." % (end_time - start_time)


if __name__ == '__main__':
    print "########### AAAI DEMO ##########"
    #main_debug()
    if len(sys.argv) < 2:
        print "synatx: `python run_aaai_demo.py d` where d is the number of experiment of interest in [0,1,2,3,4].\n Set -1 to loop over all the experiments (takes a long time/days!)."
    else:
        exp_num = int(sys.argv[1])
        if exp_num == -1:
            for exp_id in range(5):
                main(exid = exp_id)
        else:
            main(exid = exp_num)
    print "################################"
