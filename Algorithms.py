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



import numpy as np
import time
import random
import copy


# Helper functions & data structures========================================================
class Node:
    '''
    A class of node, which of depth D and base point X.
    WLOG, it is assumed the support to be in [0,1]^n and in a minimization setting and
    a partition factor of 3.
    '''
    def __init__(self, base, depth):
        self.best_val = float("inf")
        self.num_visits = 0
        self.lcb = self.best_val
        self.base = base
        self.depth = depth

    def sample(self, f):
        '''
        pull an evaluation from the current Node
        '''

        #rnd_pt = (random.random() - 0.5) * (1. / 3.0)**self.depth
        x = self.base
        #print "alg input ", x
        y = f(x)
        self.best_val = min(y, self.best_val)
        self.lcb = self.best_val
        self.num_visits += 1
        return self.lcb

    def get_lcb(self):
        return self.lcb

    def get_depth(self):
        return self.depth

    def get_best_val(self):
        return self.best_val

    def get_lambda(self):
        return np.linalg.norm(self.base-0.5)

    def get_base_pt(self):
        return self.base

    def get_lc_vector(self):
        return [self.best_val, -np.linalg.norm(self.base-0.5), self.depth]

    def get_num_visits(self):
        return self.num_visits

    def isExpandable(self):
        '''
        returns True if expandable, False otherwise
        '''
        #visit_threshold = np.linalg.norm(self.base-0.5) * 0 * 2.0 / np.sqrt(len(self.base))
        visit_threshold = 0
        if self.num_visits > visit_threshold:
            return True
        else:
            return False

    def expand(self, isReset = False):
        '''
        Expand into a 3-ary tree, by returning two other child nodes and incrementing
        the current node's depth and reseting its visitis.
        '''
        #step = 0.25 * (1.0 / 2.0)**self.depth
        step = (1.0 / 3.0)**(self.depth + 1)
        split_dim = self.depth % len(self.base)
        base1 = copy.deepcopy(self.base)
        base2 = copy.deepcopy(self.base)
        base1[split_dim] -= step
        base2[split_dim] += step
        # increment its depth
        self.depth += 1
        if isReset:
            self.num_visits = 0
        return [Node(base1, self.depth), Node(base2, self.depth)]

# for projection with random embedding
def euclidean_projection(x):
    '''
    map x into the feasible original space [-1.0,1.0]^n
    for more details, refer to Eq.(1) in (Qian and Yu, 2016)
    How to use it: with random embedding f(euclidean_projection(np.dot(A,x)))
    '''
    x_projected = np.array([1. if i > 1.0 else -1. if i < - 1.0 else float(i) for i in x])
    return x_projected

def SOO(f, n, fctn_evals):
    '''
    The Simultaneous Optimistic Optimization, return the best solution in the normalized space
    [0,1]^n
    '''
    # initialize
    max_depth = np.sqrt(fctn_evals)
    num_evals = 0
    Tree = [ Node(np.array([0.5] * n), 0) ]
    Tree[-1].sample(f)
    depth = -1
    # some bookkeeprs to increase efficiency
    min_depth = 0
    rchd_depth = 0
    v_min = float("inf")
    # the main routine
    while num_evals < fctn_evals:
        # set the depth
        #print "hi"
        depth += 1
        # sweep the tree
        if depth > max_depth:
            depth = min_depth
            v_min = float("inf")
        # get the optimistic node
        cur_idc = [idx for idx,node in enumerate(Tree) if node.depth == depth]
        cur_nodes = [Tree[idx] for idx in cur_idc]#[node for node in Tree if node.depth == depth]
        if len(cur_nodes) == 0:
            if depth > rchd_depth:
                depth = min_depth - 1
                v_min = float("inf")
            continue
        best_node_idx = np.argmin([node.get_best_val() for node in cur_nodes])
        best_node = cur_nodes[best_node_idx]
        if best_node.get_best_val() > v_min:
            continue
        else:
            v_min = best_node.get_best_val()
        # expand with some book-keeping
        if len(cur_nodes) == 1:
            min_depth += 1
        #new_nodes = best_node.expand()
        #if depth < max_depth:
        Tree = Tree + best_node.expand()
        Tree[-1].sample(f)
        Tree[-2].sample(f)
        # update the rchd_depth
        rchd_depth = max(rchd_depth, depth + 1)
        num_evals += 2

    best_node_idx = np.argsort([node.get_best_val() for node in Tree])[0]
    best_sol = Tree[best_node_idx].get_base_pt()
    return best_sol

##########################################################################################################


def EmbeddedHunter(n, fctn, fctn_evals, eta = 0.2, n_e = 10, M = 5, isRecordTree = False):
    '''
    Embedded Hunter minimizing on [-1,1]^n
    '''
    l_e = - n_e / eta
    u_e = + n_e / eta
    # for record and visualization
    record = []
    if isRecordTree:
        record.append({'l_e':l_e,'u_e':u_e}) # first element in the record keeps the boundaries of the low-dimensional space Y
    y_map = lambda y : (u_e - l_e) * y + l_e # ymap to -1,1 since we are encoding it in [0.,1.]
    g = lambda y : fctn(euclidean_projection(np.dot(1. / np.sqrt(n) * np.random.randn(n,n_e), y_map(y))))
    max_depth = np.sqrt(fctn_evals)
    # initialize
    num_evals = 0
    Tree = [ Node(np.array([0.5] * n_e), 0) ]
    Tree[-1].sample(g)
    depth = -1
    # some bookkeeprs to increase efficiency
    min_depth = 0
    rchd_depth = 0
    v_min = float("inf")
    # the main routine
    while num_evals < fctn_evals:
        # set the depth
        depth += 1
        # sweep the tree
        if depth > max_depth:
            depth = min_depth
            v_min = float("inf")
        # get the nodes at the current depth
        cur_idc = [idx for idx,node in enumerate(Tree) if node.depth == depth]
        cur_nodes = [Tree[idx] for idx in cur_idc]#[node for node in Tree if node.depth == depth]
        if len(cur_nodes) == 0:
            if depth > rchd_depth:
                depth = min_depth - 1
                v_min = float("inf")
            continue
        # expand nodes by their depth
        cur_nodes = sorted(cur_nodes, key = lambda x: (-x.get_lambda(),x.get_best_val()))
        #print "hi"
        # sweep through this level
        for idx, cur_node in enumerate(cur_nodes):
            # skip nodes of the same lambda as they are sorted and no need to compare them
            if idx > 0:
                if cur_nodes[idx-1].get_lambda() <= cur_node.get_lambda():
                    continue
            if cur_node.get_best_val() > v_min:
                continue
            else:
                #[node.get_base_pt() for node in cur_nodes]
                v_min = cur_node.get_best_val()
                # expand then and evaluate the kids
                #print "expanding"
                Tree = Tree + cur_node.expand()
                Tree[-1].sample(g)
                Tree[-2].sample(g)
                if cur_node.get_lambda() >= 1e-7 and cur_node.get_num_visits() <= M * cur_node.get_lambda(): # for zero lambda the function is determininstic in A given y
                    cur_node.sample(g)
                    num_evals += 3
                else:
                    num_evals += 2
                if num_evals >= fctn_evals:
                    return record
                if isRecordTree:
                    # record for visualization and understanding of the Algorithm
                    # h : depth, y : base point of the node in the low-dimensional space,
                    # y_p: base point of the parent node, v: number of visits of this node (number of evaluations done at its base point so far)
                    record.append({'h': depth + 1, 'y': y_map(Tree[-2].get_base_pt()), 'y_p' : y_map(cur_node.get_base_pt()), 'v': Tree[-2].get_num_visits()})
                    record.append({'h': depth + 1, 'y': y_map(cur_node.get_base_pt()), 'y_p' : y_map(cur_node.get_base_pt()), 'v': cur_node.get_num_visits()})
                    record.append({'h': depth + 1, 'y': y_map(Tree[-1].get_base_pt()), 'y_p' : y_map(cur_node.get_base_pt()), 'v': Tree[-1].get_num_visits()})
                # update the rchd_depth
                rchd_depth = max(rchd_depth, depth + 1)
        #  some book-keeping
        if len(cur_nodes) == 1:
            min_depth += 1

    return record


def RESOO(n, fctn, fctn_evals, eta = 0.2, n_e = 10, M = 5):
    '''
    Scalable Random Embedding Simultaneous Optimistic Optimization minimizing on [-1,1]^n
    AAAI'16
    '''
    l_e = - n_e / eta
    u_e = + n_e / eta
    f = lambda A,y : fctn(euclidean_projection(np.dot(A, (u_e - l_e) * y + l_e)))
    fctn_evals_per_run = (fctn_evals + M - 1) // M
    for m in xrange(M):
        A = 1. / np.sqrt(n) * np.random.randn(n,n_e)
        g = lambda y: f(A,y)
        SOO(g, n_e, fctn_evals_per_run)

def SRESOO(n, fctn, fctn_evals, eta = 0.2, n_e = 10, M = 5):
    '''
    Sequential Random Embedding with Simultaneous Optimistic Optimization minimizing on [-1,1]^n
    IJCAI'16
    '''
    l_e = -1 #- n_e / eta as in paper IJCIA
    u_e = 1 # + n_e / eta
    f = lambda A,y, alpha, x : fctn(euclidean_projection(np.dot(A, (u_e - l_e) * y + l_e) + alpha * x))
    fctn_evals_per_run = (fctn_evals + M - 1) // M
    x = np.array([0.0] * n)
    for m in xrange(M):
        # a random matrix
        A = 1. / np.sqrt(n) * np.random.randn(n,n_e)
        # build the auxiliary optimization function
        g = lambda y,alpha: f(A,y,alpha,x)
        g_bar = lambda y_bar: g(y_bar[:-1],2. * y_bar[-1] - 1.)
        norm_best_pt = SOO(g_bar, n_e+1, fctn_evals_per_run)
        x = (2. *norm_best_pt[-1] - 1)*x + np.dot(A, (u_e - l_e) * norm_best_pt[:-1] + l_e)







if __name__ == '__main__':
    print "I am supposed to be called by others!"
