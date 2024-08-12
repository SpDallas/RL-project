from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Optional, Hashable

from rl2024.constants import EX1_CONSTANTS as CONSTANTS
from rl2024.exercise1.mdp import MDP, Transition, State, Action


class MDPSolver(ABC):
    """Base class for MDP solvers

    **DO NOT CHANGE THIS CLASS**

    :attr mdp (MDP): MDP to solve
    :attr gamma (float): discount factor gamma to use
    :attr action_dim (int): number of actions in the MDP
    :attr state_dim (int): number of states in the MDP
    """

    def __init__(self, mdp: MDP, gamma: float):
        """Constructor of MDPSolver

        Initialises some variables from the MDP, namely the state and action dimension variables

        :param mdp (MDP): MDP to solve
        :param gamma (float): discount factor (gamma)
        """
        self.mdp: MDP = mdp
        self.gamma: float = gamma

        self.action_dim: int = len(self.mdp.actions)
        self.state_dim: int = len(self.mdp.states)

    def decode_policy(self, policy: Dict[int, np.ndarray]) -> Dict[State, Action]:
        """Generates greedy, deterministic policy dict

        Given a stochastic policy from state indeces to distribution over actions, the greedy,
        deterministic policy is generated choosing the action with highest probability

        :param policy (Dict[int, np.ndarray of float with dim (num of actions)]):
            stochastic policy assigning a distribution over actions to each state index
        :return (Dict[State, Action]): greedy, deterministic policy from states to actions
        """
        new_p = {}
        for state, state_idx in self.mdp._state_dict.items():
            new_p[state] = self.mdp.actions[np.argmax(policy[state_idx])]
        return new_p

    @abstractmethod
    def solve(self):
        """Solves the given MDP
        """
        ...


class ValueIteration(MDPSolver):
    """MDP solver using the Value Iteration algorithm
    """

    def _calc_value_func(self, theta: float) -> np.ndarray:
        """Calculates the value function

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        **DO NOT ALTER THE MDP HERE**

        Useful Variables:
        1. `self.mpd` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :param theta (float): theta is the stop threshold for value iteration
        :return (np.ndarray of float with dim (num of states)):
            1D NumPy array with the values of each state.
            E.g. V[3] returns the computed value for state 3
        """
        
        # Initialize the state value
        V = np.zeros(self.state_dim)
        
        # Retrieve discount factor gamma
        gamma_VI = self.gamma

        # Retrieve number of states and actions
        S_cap = self.state_dim
        A_cap = self.action_dim

        # Retrieve transition probabilities and transition rewards matrices
        r = self.mdp.R
        p = self.mdp.P

        # Repeat until convergence
        while True:

            # Initialize convergence error and save the old value vector
            delta_VI = 0
            V_old = V.copy()

            # For each state, find the maximum expected reward over the actions space
            # and save it as the state value
            for s in range(S_cap):  
                max_value = 0
                for a in range(A_cap):
                    value_a = 0
                    for s_new in range(S_cap):
                        value_a += p[s, a, s_new] * (r[s, a, s_new] + gamma_VI * V_old[s_new])
                    max_value = max(max_value, value_a)
                V[s] = max_value

                # Calculate and save convergence parameter
                delta_VI = max(delta_VI, abs(V_old[s] - V[s]))

            # Break the loop if the state value has converged
            if delta_VI < theta:
                break
 
        return V

    def _calc_policy(self, V: np.ndarray) -> np.ndarray:
        """Calculates the policy

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        :param V (np.ndarray of float with dim (num of states)):
            A 1D NumPy array that encodes the computed value function (from _calc_value_func(...))
            It is indexed as (State) where V[State] is the value of state 'State'
        :return (np.ndarray of float with dim (num of states, num of actions):
            A 2D NumPy array that encodes the calculated policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        """
        # Initialize the policy
        policy = np.zeros([self.state_dim, self.action_dim])
        
        # Retrieve discount factor gamma
        gamma_VI = self.gamma

        # Retrieve number of states and actions
        S_cap = self.state_dim
        A_cap = self.action_dim

        # Retrieve transition probabilities and transition rewards matrices
        r = self.mdp.R
        p = self.mdp.P

        # Iterate over all states
        for s in range(S_cap):

            # Initialize array to store the expected returns for each action
            expected_returns = np.zeros(A_cap)

            # Iterate over all actions
            for a in range(A_cap):

                # Calculate the expected return for the current action
                for s_new in range(S_cap):
                    expected_returns[a] += p[s, a, s_new] * (r[s, a, s_new] + gamma_VI * V[s_new])

            # Choose the action with the maximum expected return as the optimal action for the current state
            best_action = np.argmax(expected_returns)

            # Set probability of best action to 1
            policy[s, best_action] = 1.0  

        return policy

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        Compiles the MDP and then calls the calc_value_func and
        calc_policy functions to return the best policy and the
        computed value function

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        V = self._calc_value_func(theta)
        policy = self._calc_policy(V)

        return policy, V


class PolicyIteration(MDPSolver):
    """MDP solver using the Policy Iteration algorithm
    """

    def _policy_eval(self, policy: np.ndarray) -> np.ndarray:
        """Computes one policy evaluation step

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        :param policy (np.ndarray of float with dim (num of states, num of actions)):
            A 2D NumPy array that encodes the policy.
            It is indexed as (STATE, ACTION) where policy[STATE, ACTION] has the probability of
            taking action 'ACTION' in state 'STATE'.
            REMEMBER: the sum of policy[STATE, :] should always be 1.0
            For deterministic policies the following holds for each state S:
            policy[S, BEST_ACTION] = 1.0
            policy[S, OTHER_ACTIONS] = 0
        :return (np.ndarray of float with dim (num of states)): 
            A 1D NumPy array that encodes the computed value function
            It is indexed as (State) where V[State] is the value of state 'State'
        """

        # Initialize values vector
        V = np.zeros(self.state_dim)
        
        # Retrieve parameters theta (convergence) and delta (discount)
        theta_PE = self.theta
        gamma_PE = self.gamma

        # Retrieve number of states and actions
        S_cap = self.state_dim
        A_cap = self.action_dim

        # Retrieve transition probabilities and transition rewards matrices
        r = self.mdp.R
        p = self.mdp.P

        # Repeat until the state value converges
        while True:

            # Initialize delta and save the old value state
            delta_PE = 0
            V_old = V.copy()

            # For every state, save as state value function the expected
            # reward over the actions space given the policy
            for s in range(S_cap):
                sum = 0
                for a in range(A_cap):
                    for s_new in range(S_cap):
                        sum += policy[s, a] * p[s, a, s_new] * (r[s, a, s_new] + gamma_PE * V_old[s_new])
                V[s] = sum

                # Save the maximum value of convergence parameter delta
                delta_PE = max(delta_PE, abs(V_old[s] - V[s]))

            # Break the loop once the state value has converged
            if delta_PE < theta_PE:
                break      

        return np.array(V)

    def _policy_improvement(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes policy iteration until a stable policy is reached

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q1**

        Useful Variables (As with Value Iteration):
        1. `self.mpd` -- Gives access to the MDP.
        2. `self.mdp.R` -- 3D NumPy array with the rewards for each transition.
            E.g. the reward of transition [3] -2-> [4] (going from state 3 to state 4 with action
            2) can be accessed with `self.R[3, 2, 4]`
        3. `self.mdp.P` -- 3D NumPy array with transition probabilities.
            *REMEMBER*: the sum of (STATE, ACTION, :) should be 1.0 (all actions lead somewhere)
            E.g. the transition probability of transition [3] -2-> [4] (going from state 3 to
            state 4 with action 2) can be accessed with `self.P[3, 2, 4]`

        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)):
            Tuple of calculated policy and value function
        """
        
        # Initialize policy, state value function and action value function
        policy = np.zeros([self.state_dim, self.action_dim])

        # Retrieve transition probabilities and transition rewards matrices
        r = self.mdp.R
        p = self.mdp.P

        # Retrieve number of states and actions
        S_cap = self.state_dim
        A_cap = self.action_dim

        # Retrieve discount parameter
        gamma_PI = self.gamma

        # Repeat until convergence
        while True:

            # Save the old policy
            old_policy = policy.copy()

            # Can the policy evaluation function
            V = self._policy_eval(old_policy)

            # Start with an empty policy array, as we will fill it with 
            # 1 (deterministic) for the actions that maximize expected return
            policy = np.zeros([self.state_dim, self.action_dim])

            # For every state, find the action that maximizes expected returns
            # and set its probability to 1 (deterministic)
            for s in range(S_cap):

                # Initialize array to store the expected returns for each action
                expected_returns = np.zeros(A_cap)

                # Find the expected return for each action
                for a in range(A_cap):
                    for s_new in range(S_cap):
                        expected_returns[a] += p[s, a, s_new] * (r[s, a, s_new] + gamma_PI * V[s_new])

                # Choose the action with the maximum expected return as the 
                # optimal action for the current state
                best_action = np.argmax(expected_returns)

                # Set probability of best action to 1
                policy[s, best_action] = 1.0  

            # If all elements of the policy are the same with the old policy, 
            # policy iteration has converged, so break the loop.
            if np.array_equal(policy, old_policy):
                break

        return policy, V

    def solve(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the MDP

        This function compiles the MDP and then calls the
        policy improvement function that the student must implement
        and returns the solution

        **DO NOT CHANGE THIS FUNCTION**

        :param theta (float, optional): stop threshold, defaults to 1e-6
        :return (Tuple[np.ndarray of float with dim (num of states, num of actions),
                       np.ndarray of float with dim (num of states)]):
            Tuple of calculated policy and value function
        """
        self.mdp.ensure_compiled()
        self.theta = theta
        return self._policy_improvement()


if __name__ == "__main__":
    mdp = MDP()
    mdp.add_transition(
        #         start action end prob reward
        Transition("rock0", "jump0", "rock0", 1, 0),
        Transition("rock0", "stay", "rock0", 1, 0),
        Transition("rock0", "jump1", "rock0", 0.1, 0),
        Transition("rock0", "jump1", "rock1", 0.9, 0),
        Transition("rock1", "jump0", "rock1", 0.1, 0),
        Transition("rock1", "jump0", "rock0", 0.9, 0),
        Transition("rock1", "jump1", "rock1", 0.1, 0),
        Transition("rock1", "jump1", "land", 0.9, 10),
        Transition("rock1", "stay", "rock1", 1, 0),
        Transition("land", "stay", "land", 1, 0),
        Transition("land", "jump0", "land", 1, 0),
        Transition("land", "jump1", "land", 1, 0),
    )

    solver = ValueIteration(mdp, CONSTANTS["gamma"])
    policy, valuefunc = solver.solve()
    print("---Value Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)

    solver = PolicyIteration(mdp, CONSTANTS["gamma"])
    policy, valuefunc = solver.solve()
    print("---Policy Iteration---")
    print("Policy:")
    print(solver.decode_policy(policy))
    print("Value Function")
    print(valuefunc)
