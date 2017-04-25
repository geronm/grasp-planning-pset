import numpy as np

class POMDP(object):
    def __init__(self, stuff):
        pass
    
    def prob_obs_given_bs_a(self, b_s, a, o):
        raise NotImplementedError('Not implemented.')
    
    def bayes_update(self, b_s, a, o):
        raise NotImplementedError('Not implemented.')
    
    def cost(self, b_s, actions, observations):
        raise NotImplementedError('Not implemented.')
    
    def is_terminal_belief(self, b_s):
        raise NotImplementedError('Not implemented.')
    
    def heuristic(self, b_s):
        raise NotImplementedError('Not implemented.')
    
    def get_possible_actions(self, b_s):
        raise NotImplementedError('Not implemented.')
    
    def get_possible_observations(self, b_s, a):
        raise NotImplementedError('Not implemented.')
    
    def solve(self, b_s_init, depth=3):
        root = ObservationNode(self, b_s_init, None, None)
        
        out, a0_best = self.alphabeta_observation_node(root, depth, float('-Inf'), float('+Inf'))
        
        return out, a0_best
    
    def alphabeta_action_node(self, act_node, depth, alpha, beta):
        # action nodes are never terminal
        # Iterate over obs nodes
        v = 0.0
        for child_obs in act_node.get_children():
            prob_obs = self.prob_obs_given_bs_a(act_node.b_s, act_node.a, child_obs.o)
            if prob_obs > 0:
                value, a_best = self.alphabeta_observation_node(child_obs, depth, alpha, beta)
                v += prob_obs*value
                #beta = max(beta, v) # TODO: LOOK CLOSELY AT THE LOGIC HERE
                #if beta <= alpha:
                #    break # beta cut-off
                if v < alpha:
                    break # alpha cut-off, PERHAPS THE ONLY TYPE OF CUT-OFF
        return v

    def alphabeta_observation_node(self, obs_node, depth, alpha, beta):
        # obs nodes can be terminal
        if depth == 0 or self.is_terminal_belief(obs_node.b_s):
            # Return negative cost so higher value is better
            return -self.cost(obs_node.b_s, obs_node.get_actions(), obs_node.get_observations()), None
        # If not terminal, iterate over next action nodes
        v = float('-Inf');
        a_best = None
        for child_act in obs_node.get_children():
            value = self.alphabeta_action_node(child_act, depth-1, alpha, beta)
            #if depth == 2:
            #    print 'act: %s value: %s' % (str(child_act.a), str(value))
            if v < value:
                a_best = child_act.a
            v = max(v, value)
            alpha = max(v, alpha)
            #if beta <= alpha:
            #    # alpha cut-off
            #    break
        return v, a_best
            
        

                
class ActionNode(object):
    def __init__(self, pomdp, b_s, a, parent_o):
        self.pomdp = pomdp
        self.b_s = b_s
        self.a = a
        self.parent_o = parent_o
        
    def get_children(self):
        return [ObservationNode(self.pomdp, self.b_s, o, self) for o in self.pomdp.get_possible_observations(self.b_s, self.a)]

class ObservationNode(object):
    def __init__(self, pomdp, b_s, o, parent_a):
        self.pomdp = pomdp
        self.b_s = b_s
        self.o = o
        self.parent_a = parent_a
        
    def get_children(self):
        if self.parent_a is not None:
            return [ActionNode(self.pomdp,
                               self.pomdp.bayes_update(self.b_s, self.parent_a.a, self.o),
                               a,
                               self) for a in self.pomdp.get_possible_actions(self.b_s)]
        

        return [ActionNode(self.pomdp,
                           self.b_s,
                           a,
                           self) for a in self.pomdp.get_possible_actions(self.b_s)]

    def get_observations(self):
        if self.parent_a.parent_o is None or self.parent_a.parent_o.o is None:
            return [self.o]
        return [self.o]          + self.parent_a.parent_o.get_observations()
    
    def get_actions(self):
        if self.parent_a.parent_o is None or self.parent_a.parent_o.o is None:
            return [self.parent_a.a]
        return [self.parent_a.a] + self.parent_a.parent_o.get_actions()


class BattleshipProblem(POMDP):
    # state is the location of a 1-by-3 battleship in a 5-by-5 grid (ie. 5-by-3 possible locations)
    # actions are guessing a square (r, c)
    # observations are whether hit/miss that guessed square: (hit?)
    #
    # b_s actually has both certain and uncertain state.
    #   b_s = (p_s, guessed_set)
    #        p_s - distribution over grid, stored as 5-by-3 numpy array.
    #        guessed_set - a python set() which tells which squares have been guessed so far (guess outcomes encoded in p_s)
    def __init__(self):
        pass
    
    def prob_obs_given_bs_a(self, b_s, a, o):
        # for a given observation o, we need to give the probability of that observation
        # under the belief b_s
        row, col = a
        did_hit = o
        p_s, guessed_set = b_s
        
        # compute the probability that the ship occupied square (row, col)
        p_occd = 0.0
        for test_col in range(3):
            test_col_tru = test_col + 1 # test_col is in width-3 state space; test_col_true is in the width-5 grid
            if abs(test_col_tru - col) <= 1:
                p_occd += p_s[row][test_col]
        
        if did_hit:
            return p_occd
        else:
            return 1-p_occd

    def bayes_update(self, b_s, a, o):
        # we update our belief ; essentially states were either possible or impossible given the observation
        row, col = a
        did_hit = o
        p_s, guessed_set = b_s
        
        possible_states = np.zeros((5,3))
        
        # compute the probability that the ship occupied square (row, col)
        p_occd = 0.0
        for test_col in range(3):
            test_col_tru = test_col + 1 # test_col is in width-3 state space; test_col_true is in the width-5 grid
            if abs(test_col_tru - col) <= 1: # possible!
                possible_states[row][test_col] = 1.0
                
        if not did_hit:
            possible_states = 1.0 - possible_states
        
        # now, possible_states holds a mask of which
        # beliefs should survive
        p_s_new = np.multiply(possible_states, p_s)
        p_s_new_total = np.sum(p_s_new)
        
        if p_s_new_total == 0.0:
            raise Exception("Performed Bayesian update on an observation of 0 probability")
        
        p_s_new = p_s_new / p_s_new_total
        
        # also, guessed set got bigger
        guessed_set_new = frozenset(list(guessed_set) + [a])
        
        # return updated belief
        return (p_s_new, guessed_set_new)
     
    def cost(self, b_s, actions, observations):
        # cost in Battleship is how many moves it took to sink the ship.
        # guessed_states = np.zeros((5,3))
        if self.board_is_done(b_s):
            return len(actions) # can score
            # return 0
        
        # otherwise, heuristics
        return len(actions) + self.heuristic(b_s, actions, observations)  # score so far, plus heuristic cost-to-go
    
    def heuristic(self, b_s, actions, observations):
        # heuristic in Battleship is min of
        #         2 + 2 * number of rows with nonzero belief,
        #  and
        #         num with nonzero belief.
        #
        p_s, guessed_set = b_s
        
        p_s_nonzero = (p_s > 0) + 0.0
        
        row_sums = np.sum(p_s, 1)
        row_count_nonzero = np.sum( (row_sums > 0) + 0.0 )
        
        return min( np.sum(p_s_nonzero) , 2 + 2 * np.sum(row_count_nonzero) )

    def get_possible_actions(self, b_s):
        p_s, guessed_set = b_s
        
        actions = []
        for i in range(len(p_s)):
            for j in range(2 + len(p_s[0])):
                if (i,j) not in guessed_set:
                    actions.append((i,j))
        
        return actions
    
    def get_possible_observations(self, b_s, a):
        # True or False, depending on b_s
        p_o_true = self.prob_obs_given_bs_a(b_s, a, True)
        p_o_false = self.prob_obs_given_bs_a(b_s, a, False)
        
        possible_o = []
        if p_o_true:
            possible_o.append(True)
        if p_o_false:
            possible_o.append(False)
        
        return possible_o
    
    def is_terminal_belief(self, b_s):
        return self.board_is_done(b_s)
    
    def board_is_done(self, b_s):
        p_s, guessed_set = b_s
        
        certain = np.sum( (p_s == 1) + 0.0 ) == 1
        
        if not certain:
            return False
        
        # find the location of certainty in the belief
        ship_r = None
        ship_c = None
        for r in range(len(p_s)):
            for c in range(len(p_s[0])):
                if p_s[r][c] > 0:
                    ship_r = r
                    ship_c = c+1 # tru coordinates
        
        # make sure the ship was sunk for the believed state
        if (ship_r, ship_c-1) in guessed_set and \
                (ship_r, ship_c  ) in guessed_set and \
                (ship_r, ship_c+1) in guessed_set:
            return True
        
        # More sinkin' to do
        return False
    
    def get_uniform_belief(self):
        return (np.ones((5,3)) / np.sum(np.ones((5,3))), frozenset())
    
    

        