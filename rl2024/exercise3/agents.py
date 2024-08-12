from abc import ABC, abstractmethod
from copy import deepcopy
import gymnasium as gym
import numpy as np
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List

from rl2024.exercise3.networks import FCNetwork
from rl2024.exercise3.replay import Transition


class Agent(ABC):
    """Base class for Deep RL Exercise 3 Agents

    **DO NOT CHANGE THIS CLASS**

    :attr action_space (gym.Space): action space of used environment
    :attr observation_space (gym.Space): observation space of used environment
    :attr saveables (Dict[str, torch.nn.Module]):
        mapping from network names to PyTorch network modules

    Note:
        see https://gymnasium.farama.org/api/spaces/ for more information on Gymnasium spaces
    """

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def update(self):
        ...


class DQN(Agent):
    """The DQN agent for exercise 3

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**

    :attr critics_net (FCNetwork): fully connected DQN to compute Q-value estimates
    :attr critics_target (FCNetwork): fully connected DQN target network
    :attr critics_optim (torch.optim): PyTorch optimiser for DQN critics_net
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr update_counter (int): counter of updates for target network updates
    :attr target_update_freq (int): update frequency (number of iterations after which the target
        networks should be updated)
    :attr batch_size (int): size of sampled batches of experience
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay_strategy: str = "constant",
        epsilon_decay: float = None,
        exploration_fraction: float = None,
        **kwargs,
        ):
        """The constructor of the DQN agent class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param target_update_freq (int): update frequency (number of iterations after which the target
            networks should be updated)
        :param batch_size (int): size of sampled batches of experience
        :param gamma (float): discount rate gamma
        :param epsilon_start (float): initial value of epsilon for epsilon-greedy action selection
        :param epsilon_min (float): minimum value of epsilon for epsilon-greedy action selection
        :param epsilon_decay (float, optional): decay rate of epsilon for epsilon-greedy action. If not specified,
                                                epsilon will be decayed linearly from epsilon_start to epsilon_min.
        """
        super().__init__(action_space, observation_space)

        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.critics_net = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
            )

        self.critics_target = deepcopy(self.critics_net)

        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=learning_rate, eps=1e-3
            )

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min

        self.epsilon_decay_strategy = epsilon_decay_strategy
        if epsilon_decay_strategy == "constant":
            assert epsilon_decay is None, "epsilon_decay should be None for epsilon_decay_strategy == 'constant'"
            assert exploration_fraction is None, "exploration_fraction should be None for epsilon_decay_strategy == 'constant'"
            self.epsilon_exponential_decay_factor = None
            self.exploration_fraction = None
        elif self.epsilon_decay_strategy == "linear":
            assert epsilon_decay is None, "epsilon_decay is only set for epsilon_decay_strategy='exponential'"
            assert exploration_fraction is not None, "exploration_fraction must be set for epsilon_decay_strategy='linear'"
            assert exploration_fraction > 0, "exploration_fraction must be positive"
            self.epsilon_exponential_decay_factor = None
            self.exploration_fraction = exploration_fraction
        elif self.epsilon_decay_strategy == "exponential":
            assert epsilon_decay is not None, "epsilon_decay must be set for epsilon_decay_strategy='exponential'"
            assert exploration_fraction is None, "exploration_fraction is only set for epsilon_decay_strategy='linear'"
            self.epsilon_exponential_decay_factor = epsilon_decay
            self.exploration_fraction = None
        else:
            raise ValueError("epsilon_decay_strategy must be either 'linear' or 'exponential'")
        # ######################################### #
        self.saveables.update(
            {
                "critics_net"   : self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim"  : self.critics_optim,
                }
            )

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**
        ** Implement both epsilon_linear_decay() and epsilon_exponential_decay() functions **
        ** You may modify the signature of these functions if you wish to pass additional arguments **

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """

        def epsilon_linear_decay(epsilon_start, epsilon_min, exploration_fraction, timestep, max_timestep):
            """
            A function that decreases epsilon linearly for each timestep,
            until it reaches the minimum value of epsilon (epsilon_min), in the number of 
            steps determined by the exploration fraction.

            Parameters:
            - epsilon_start: Initial value of epsilon.
            - epsilon_min: Minimum value of epsilon.
            - exploration_fraction: Time steps out of max_timesteps before reaching epsilon_min.
            - timestep: Current timestep.
            - max_timestep: Maximum number of training timesteps.

            Returns:
            - epsilon: The current decayed epsilon value based on the present timestep.
            """
            # Calculate the decay rate
            slope = (epsilon_min - epsilon_start) / (exploration_fraction * max_timestep)

            # Calculate the current decayed epsilon
            epsilon = slope * timestep + epsilon_start

            # Ensure epsilon does not go below minimum
            epsilon = max(epsilon, epsilon_min)

            # Ensure epsilon does not exceed the initial one
            epsilon = min(epsilon, epsilon_start)
            
            return epsilon

        def epsilon_exponential_decay(epsilon_start, epsilon_min, epsilon_decay_factor, timestep, max_timestep):
            """
            A function that decreases epsilon exponentialy for each timestep, 
            from a starting point to a minimum point, based on a given decay factor.

            Parameters:
            - epsilon_start: Starting value of epsilon.
            - epsilon_min: Minimum value of epsilon after decay.
            - epsilon_decay_factor: The base of the exponential rate at which epsilon decays each step.
            - timestep: Current training timestep.
            - max_timestep: Maximum number of training timesteps.

            Returns:
            - epsilon: The decayed epsilon value based on the current timestep.
            """
            # The recursive formula is: eps_new = eps_old * decay_factor ** (dt / t_max)
            # The non-recursive formula is: eps_new = eps_start * decay_factor ** (sum(i=0,...,t) / tmax)
            # The non-recursive formula was used here.

            # First find sum from 0 to current timestep
            sum_timestep = timestep * (timestep + 1) / 2

            # Then calculate the exponent
            exponent = sum_timestep / max_timestep

            # Calculate the decayed epsilon
            epsilon = epsilon_start * epsilon_decay_factor ** exponent

            # Ensure epsilon is greater than or equal to epsilon_min
            epsilon = max(epsilon, epsilon_min)
            
            return epsilon

        if self.epsilon_decay_strategy == "constant":
            pass
        elif self.epsilon_decay_strategy == "linear":
            # linear decay
            ### PUT YOUR CODE HERE ###
            self.epsilon = epsilon_linear_decay(self.epsilon_start, self.epsilon_min, self.exploration_fraction, timestep, max_timestep)
        
        elif self.epsilon_decay_strategy == "exponential":
            # exponential decay
            ### PUT YOUR CODE HERE ###
            self.epsilon = epsilon_exponential_decay(self.epsilon_start, self.epsilon_min, self.epsilon_exponential_decay_factor, timestep, max_timestep)
        
        else:
            raise ValueError("epsilon_decay_strategy must be either 'constant', 'linear' or 'exponential'")

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        When explore is False you should select the best action possible (greedy). However, during
        exploration, you should be implementing an exploration strategy (like e-greedy). Use
        schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """

        if explore and np.random.uniform(0, 1) < self.epsilon:

            # Select a random action
            action = self.action_space.sample()

        else:

            # Select the greedy action for the current state

            # Convert the states to a tensor
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            # There is no need tp track the gradient
            with torch.no_grad():

                # Retrieve the Q-table for all actions from the network
                q_table = self.critics_net(state_tensor)  

            # Select the greedy action
            action = torch.argmax(q_table).item()

        return action

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your network, update the target network at the given
        target update frequency, and return the Q-loss in the form of a dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """

        # Retrieve the elements of the batch
        states, actions, new_states, rewards, dones = batch

        # Convert tensor actions
        actions = actions.to(torch.long)

        # Calculate Q(a|s), max_a(Q(a|s')) and target
        q_table = self.critics_net(states).gather(1, actions)
        new_state_q_table = self.critics_target(new_states).detach().max(1)[0].unsqueeze(-1)
        target = rewards + self.gamma * new_state_q_table * (1 - dones)

        # Compute the loss function
        q_loss = torch.nn.functional.mse_loss(q_table, target)

        # Zero gradients before backpropagation
        self.critics_optim.zero_grad()

        # Backpropagation
        q_loss.backward()

        # Update the network
        self.critics_optim.step()

        # Update the target network after C steps
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.critics_target.load_state_dict(self.critics_net.state_dict())

        q_loss = q_loss.item()

        return {"q_loss": q_loss}


class Reinforce(Agent):
    """ The Reinforce Agent for Ex 3

    ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **

    :attr policy (FCNetwork): fully connected network for policy
    :attr policy_optim (torch.optim): PyTorch optimiser for policy network
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        gamma: float,
        **kwargs,
        ):
        """
        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.policy = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=torch.nn.modules.activation.Softmax
            )

        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-3)

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.gamma = gamma

        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #

        # ###############################################
        self.saveables.update(
            {
                "policy": self.policy,
                }
            )

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        pass

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        Select an action from the model's stochastic policy by sampling a discrete action
        from the distribution specified by the model output

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
      
        # Convert the states array to a tensor
        state_tensor = torch.tensor(obs, dtype=torch.float32)

        # Retrieve the policy for these states
        act_probs = self.policy(state_tensor)

        if explore:
            # Sample an action based on the policy
            action = torch.distributions.Categorical(act_probs).sample().item()
        else:
            # Select the greedy action
            action = torch.argmax(act_probs).item()

        return action

    def update(
        self, rewards: List[float], observations: List[np.ndarray], actions: List[int],
        ) -> Dict[str, float]:
        """Update function for policy gradients

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        :param rewards (List[float]): rewards of episode (from first to last)
        :param observations (List[np.ndarray]): observations of episode (from first to last)
        :param actions (List[int]): applied actions of episode (from first to last)
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
            losses
        """

        # Initialize reward and policy loss
        G = 0
        p_loss = 0

        # Retrieve length of episode
        T = len(rewards)

        # For t = T-1, ..., 0
        for t in reversed(range(T)):

            # Calculate the updated reward
            G = rewards[t] + self.gamma * G

            # Convert the states array to a tensor
            state_tensor = torch.tensor(observations[t], dtype=torch.float32)

            # Calculate the log policy of present action
            act_probs = self.policy(state_tensor)
            log_policy = torch.log(act_probs[actions[t]])

            # Compute the loss function L_theta, here named p_loss
            p_loss += - G * log_policy

        p_loss = p_loss / T

        # Perform a gradient step and update policy weights
        self.policy_optim.zero_grad()
        p_loss.backward()
        self.policy_optim.step()

        p_loss = p_loss.item()

        return {"p_loss": p_loss}
