
############################################################################################################
##########################            RL2023 Assignment Answer Sheet              ##########################
############################################################################################################

# **PROVIDE YOUR ANSWERS TO THE ASSIGNMENT QUESTIONS IN THE FUNCTIONS BELOW.**

############################################################################################################
# Question 2
############################################################################################################

def question2_1() -> str:
    """
    (Multiple choice question):
    For the Q-learning algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_2() -> str:
    """
    (Multiple choice question):
    For the First-visit Monte Carlo algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_3() -> str:
    """
    (Multiple choice question):
    Between the two algorithms (Q-Learning and First-Visit MC), whose average evaluation return is impacted by gamma in
    a greater way?
    a) Q-Learning
    b) First-Visit Monte Carlo
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_4() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) as to why the value of gamma affects more the evaluation returns achieved
    by [Q-learning / First-Visit Monte Carlo] when compared to the other algorithm.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "The discount factor gamma affects more the evaluation returns of the First-Visit Monte Carlo (FVMC) \
        than the Q-learning (QL), because in FVMC the update of the value function at each time step takes into account \
        the compounding effect of gamma at this step as we go from T-1, to 0. On the other hand, QL as a time difference \
        method, has more localized effect of gamma to the value function, meaning that at each time step, gamma is \
        multiplied with the greedy value of the next step only to find together with the next state reward the effect \
        on the present value."  
    return answer


############################################################################################################
# Question 3
############################################################################################################

def question3_1() -> str:
    """
    (Multiple choice question):
    In Reinforce, which learning rate achieves the highest mean returns at the end of training?
    a) 2e-2
    b) 2e-3
    c) 2e-4
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_2() -> str:
    """
    (Multiple choice question):
    When training DQN using a linear decay strategy for epsilon, which exploration fraction achieves the highest mean
    returns at the end of training?
    a) 0.99
    b) 0.75
    c) 0.01
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_3() -> str:
    """
    (Multiple choice question):
    When training DQN using an exponential decay strategy for epsilon, which epsilon decay achieves the highest
    mean returns at the end of training?
    a) 1.0
    b) 0.5
    c) 1e-5
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_4() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of training when employing an exponential decay strategy
    with epsilon decay set to 1.0?
    a) 0.0
    b) 1.0
    c) epsilon_min
    d) approximately 0.0057
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_5() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of  training when employing an exponential decay strategy
    with epsilon decay set to 0.95?
    a) 0.95
    b) 1.0
    c) epsilon_min
    d) approximately 0.0014
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_6() -> str:
    """
    (Short answer question):
    Based on your answer to question3_5(), briefly  explain why a decay strategy based on an exploration fraction
    parameter (such as in the linear decay strategy you implemented) may be more generally applicable across
    different environments  than a decay strategy based on a decay rate parameter (such as in the exponential decay
    strategy you implemented).
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "With exploration fraction parameter strategies one can directly select the duration of the \
        exploration phase, whereas in decay rate parameter strategies this direct control and tuning convenience \
        is lost. So for diverse environments, where we would need to better understand and control/tune the \
        level of exploration, exploration fraction strategies would offer more transparency and adaptability, \
        where in the case of decay rate strategies, which we saw are very sensitive wrt decay rate, one would \
        need to in addition create a map between exploration timesteps and decay rates to have the same level \
        of control."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


def question3_7() -> str:
    """
    (Short answer question):
    In DQN, explain why the loss is not behaving as in typical supervised learning approaches
    (where we usually see a fairly steady decrease of the loss throughout training)
    return: answer (str): your answer as a string (150 words max)
    """
    answer = "The loss graph presents jumps in every few time steps and the effect of compounding error, \
        explained by the loss's decomposition of (Mnih et al, 2015, p.7/13).  The loss function is decomposed to \
        a term which changes in each iteration and a term of variance. The changing term is due to the fact that \
        the target depends on the neural network weights which are updated as the network learns. The target is \
        also hard updated every few timesteps based on the current network justifying the jumps. \
        The variance term increases with time, due to errors introduced perhaps by use of bootstrapping, to \
        predict future targets based on current rewards estimation, and also because the neural network \
        functions are themselves approximations that introduce compounded overestimation error in each step. \
        In contrast, in standard supervised learning where the target is set before learning starts, the loss \
        function shows a steady decreasing trend."  # TYPE YOUR ANSWER HERE (150 words max)
    return answer


def question3_8() -> str:
    """
    (Short answer question):
    Provide an explanation for the spikes which can be observed at regular intervals throughout
    the DQN training process.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "It is due to the hard update of the target value function with the current value function \
        every C timesteps, determined by the update frequency. This magnifies the first term of the loss \
        function, previously mentioned in 3_7, giving the appearance of spikes."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 5
############################################################################################################

def question5_1() -> str:
    """
    (Short answer question):
    Provide a short description (200 words max) describing your hyperparameter turning and scheduling process to get
    the best performance of your agents
    return: answer (str): your answer as a string (200 words max)
    """
    answer = "Firstly, I noticed that the combination of hyperparameters in Q4 was decent, so I doubled my neuron \
        sizes to [128, 128, 128], and the results showed acceptable time per simulation (~25') and better mean \
        returns (~700). Then I researched into existing literature for the magnitude of the rest of the \
        hyperparameters, using mainly (Lilicrap et al) and (Ashraf et al) papers. I used three sets of hyperparameters \
        for an exhaustive search, two from the papers and one from Q4 constants. The exhaustive search offered \
        combinations with returns above 800, (interestingly different from the selection of each case) and so I \
        I concluded my search, with policy and critic network size [128, 128, 128], policy learning rate 0.0001, \
        critic learning rate 0.0046, tau 0.0067, batch size 50 and gamma0.95, (mean final score: 1086.20 +- 95.17).\
        (Ashraf et al): Optimizing hyperparameters of deep reinforcement learning for autonomous driving based on \
        whale optimization algorithm, PLOS ONE, 2021."  # TYPE YOUR ANSWER HERE (200 words max)
    return answer