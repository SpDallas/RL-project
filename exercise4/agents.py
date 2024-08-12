import os
import gymnasium as gym
import numpy as np
from torch.optim import Adam
from typing import Dict, Iterable
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

from rl2024.exercise3.agents import Agent
from rl2024.exercise3.networks import FCNetwork
from rl2024.exercise3.replay import Transition


class DiagGaussian(torch.nn.Module):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        eps = Variable(torch.randn(*self.mean.size()))
        return self.mean + self.std * eps


class DDPG(Agent):
    """ DDPG

        ** YOU NEED TO IMPLEMENT THE FUNCTIONS IN THIS CLASS **

        :attr critic (FCNetwork): fully connected critic network
        :attr critic_optim (torch.optim): PyTorch optimiser for critic network
        :attr policy (FCNetwork): fully connected actor network for policy
        :attr policy_optim (torch.optim): PyTorch optimiser for actor network
        :attr gamma (float): discount rate gamma
        """

    def __init__(
            self,
            action_space: gym.Space,
            observation_space: gym.Space,
            gamma: float,
            critic_learning_rate: float,
            policy_learning_rate: float,
            critic_hidden_size: Iterable[int],
            policy_hidden_size: Iterable[int],
            tau: float,
            **kwargs,
    ):
        """
        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param gamma (float): discount rate gamma
        :param critic_learning_rate (float): learning rate for critic optimisation
        :param policy_learning_rate (float): learning rate for policy optimisation
        :param critic_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected critic
        :param policy_hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected policy
        :param tau (float): step for the update of the target networks
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.shape[0]

        self.upper_action_bound = action_space.high[0]
        self.lower_action_bound = action_space.low[0]

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        # self.actor = Actor(STATE_SIZE, policy_hidden_size, ACTION_SIZE)
        self.actor = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=torch.nn.Tanh
        )
        self.actor_target = FCNetwork(
            (STATE_SIZE, *policy_hidden_size, ACTION_SIZE), output_activation=torch.nn.Tanh
        )

        self.actor_target.hard_update(self.actor)
        # self.critic = Critic(STATE_SIZE + ACTION_SIZE, critic_hidden_size)
        # self.critic_target = Critic(STATE_SIZE + ACTION_SIZE, critic_hidden_size)

        self.critic = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target = FCNetwork(
            (STATE_SIZE + ACTION_SIZE, *critic_hidden_size, 1), output_activation=None
        )
        self.critic_target.hard_update(self.critic)

        self.policy_optim = Adam(self.actor.parameters(), lr=policy_learning_rate, eps=1e-3)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_learning_rate, eps=1e-3)


        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.gamma = gamma
        self.critic_learning_rate = critic_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.tau = tau

        # ################################################### #
        # DEFINE A GAUSSIAN THAT WILL BE USED FOR EXPLORATION #
        # ################################################### #
        mean = torch.zeros(ACTION_SIZE)
        std = 0.1 * torch.ones(ACTION_SIZE)
        # Original noise initialisation with custom function DiagGaussian
        # self.noise = DiagGaussian(mean, std)

        # Noise initialisation with pytorch function normal, allowing later to sample.
        self.noise = Normal(mean, std)

        # ############################### #
        # WRITE ANY AGENT PARAMETERS HERE #
        # ############################### #

        self.saveables.update(
            {
                "actor": self.actor,
                "actor_target": self.actor_target,
                "critic": self.critic,
                "critic_target": self.critic_target,
                "policy_optim": self.policy_optim,
                "critic_optim": self.critic_optim,
            }
        )


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


    def restore(self, filename: str, dir_path: str = None):
        """Restores PyTorch models from models file given by path

        :param filename (str): filename containing saved models
        :param dir_path (str, optional): path to directory where models file is located
        """

        if dir_path is None:
            dir_path = os.getcwd()
        save_path = os.path.join(dir_path, filename)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())


    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):
        """Updates the hyperparameters

        **YOU MAY IMPLEMENT THIS FUNCTION FOR Q5**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ### PUT YOUR CODE HERE ###
        pass

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        When explore is False you should select the best action possible (greedy). However, during exploration,
        you should be implementing exporation using the self.noise variable that you should have declared in the __init__.
        Use schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """

        # Convert the state vector into a 1D tensor
        state_tensor = torch.FloatTensor(obs).unsqueeze(0)

        # Get the action vector, based on the state vector
        # No need to track operations for gradient computation (detach)
        action_tensor = self.actor(state_tensor).detach()

        # During exploration phase, make the continuous action vector noisy
        if explore:
            action_tensor += self.noise.sample()

        # Continuous actions should be within [lower bound, upper bound]. But with 
        # the introduction of randomness through the policy and noise, an exploratory
        # action vector might get out of action bounds. Therefore, the action values 
        # need to be adjusted within the upper and lower bounds of the action space.
        action_tensor = torch.clamp(action_tensor, self.lower_action_bound, self.upper_action_bound)

        # Ensure the action tensor is in the cpu and convert it to a NumPy array before returning it.
        action = action_tensor.squeeze(0).cpu().numpy()

        return action

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q4**

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your critic and actor networks, target networks with soft
        updates, and return the q_loss and the policy_loss in the form of a dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        
        # Retrieve the experience batch from the replay buffer
        states, actions, new_states, rewards, dones = batch

        # Initialize critic network and actor network loss functions
        q_loss = 0.0
        p_loss = 0.0

        #----- Calculate the loss function for the critic network -----#

        # Gradient not monitored for new action and Q arrays, as not needed in the update step.
        with torch.no_grad():

            # Retrieve the new actions given the new states, from the target actor network
            new_actions = self.actor_target(new_states)

            # Update the Q table using the target critic network
            new_Q = self.critic_target(torch.cat([new_states, new_actions], dim=1)) #forward?

            # Calculate the target Q table
            Q_target = rewards + (self.gamma * new_Q * (1 - dones))
        
        # Retrieve the present Q, given present states and actions, using the critic network
        current_Q = self.critic(torch.cat([states, actions], dim=1))

        # Calculate the critic network loss function
        q_loss = F.mse_loss(current_Q, Q_target)

        # Update the critic network

        # Reset grads to 0
        self.critic_optim.zero_grad()
        # Calculate the gradient of the network loss function
        q_loss.backward()
        # Update the critic network parameters
        self.critic_optim.step()

        #----- Calculate the loss function for the actor network -----#

        # Given present states, retrieve predicted actions from actor network
        predicted_actions = self.actor(states)

        # Calculate the actor network loss function
        p_loss = -self.critic(torch.cat([states, predicted_actions], dim=1)).mean()

        # Update the actor network, without updating the critic network

        # Reset grads to 0
        self.policy_optim.zero_grad()
        # Calculate the gradient of the actor loss function
        p_loss.backward()
        # Update the actor network parameters
        self.policy_optim.step()

        #----- Soft update the target networks -----#

        # Target critic soft update
        for target_theta, theta in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_theta.data.copy_(self.tau * theta.data + (1 - self.tau) * target_theta.data)

        # Target actor soft update
        for target_phi, phi in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_phi.data.copy_(self.tau * phi.data + (1.0 - self.tau) * target_phi.data)

        return {
            "q_loss": q_loss.item(),
            "p_loss": p_loss.item(),
        }
