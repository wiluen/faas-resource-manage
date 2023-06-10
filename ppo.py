import time
import numpy as np
import gym
import torch
from torch import nn
import matplotlib.pyplot as plt
from util import RLEnvironment

class ForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ForwardNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        # Weichao: maybe reduce one fc layer?
        input = torch.FloatTensor(input)
        output = self.relu(self.fc1(input))
        output = self.relu(self.fc2(output))
        output = self.fc3(output)
        if not flag_continuous_action:
            output = self.softmax(output)

        return output


def calc_action(state):
    if flag_continuous_action:
        mean = actor(state)
        dist = torch.distributions.MultivariateNormal(mean, cov)  #多元正太分布
    else:
        action_probs = actor(state)
        dist = torch.distributions.Categorical(action_probs)

    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.detach().numpy(), log_prob.detach()


def calc_GAE(rewards):
    returns = []
    for episode_rewards in reversed(rewards):
        discounted_return = 0.0
        # Caution: Episodes might have different lengths if stopped earlier
        for reward in reversed(episode_rewards):
            discounted_return = reward + discounted_return * DISCOUNT
            returns.insert(0, discounted_return)

    returns = torch.FloatTensor(returns)
    return returns

def deal_action(action):
    scaled_action = (action + 1) * 4 # 映射到0~8范围
    discrete_action = int(scaled_action) % 8  # 取余数为0-7 
    return discrete_action

# def save_model(actor,critic,iter):
#     torch.save(actor.state_dict(),'actor_'+iter'.pth')
#     torch.save(critic.state_dict(),'actor_'+iter'.pth')

# def load_model(iter):
#     actor=torch.load('actor_'+iter'.pth')
#     critic=torch.load('actor_'+iter'.pth')
#     return actor,critic

if __name__ == '__main__':
    start_time = time.time()

    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = "cpu"

    # env = gym.make('Pendulum-v0')
    # flag_continuous_action = True
    # print("env=Catepole v0")

    env = RLEnvironment()
    env.load_data()
    flag_continuous_action = True

    HIDDEN_SIZE = 64
    lr = 5e-3
    DISCOUNT = 0.99
    TOTAL_ITERATIONS = 400             # 400
    EPISODES_PER_ITERATION = 10           # 20
    EPISODE_LENGTH = 200             # 200
    SGD_EPOCHS = 5
    CLIP = 0.2

    state_size = env.observation_space
    if flag_continuous_action:
        action_size = env.action_space
    else:
        action_size = env.action_space

    actor = ForwardNetwork(state_size, HIDDEN_SIZE, 2)
    critic = ForwardNetwork(state_size, HIDDEN_SIZE, 1)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)       # 居然是一起优化的

    cov = torch.diag(torch.ones(action_size, ) * 0.5)

    # for plots
    iteration_rewards = []
    smoothed_rewards = []

    for iteration in range(TOTAL_ITERATIONS):
        states = []
        actions = []
        rewards = []
        log_probs = []

        for episode in range(EPISODES_PER_ITERATION):
            state = env.reset()
            episode_rewards = []

            for step in range(EPISODE_LENGTH):
                action, log_prob = calc_action(state)
                # print(action,log_prob)
                a1=deal_action(action[0])
                a2=deal_action(action[1])
                new_action=[4,a1,a2,7,4]

                #做一个 固定策略 +  随着训练衰减转为探索s


                # print(new_action)
                next_state, reward, done= env.step(new_action)

                states.append(state)
                episode_rewards.append(reward)
                log_probs.append(log_prob)
                actions.append(action)

                if done:
                    break
                state = next_state

            rewards.append(episode_rewards)   # 应该是每个episode的奖励，二维数组，每行代表了这一个episode每步的奖励
        # print(rewards)       [10*200] [episode*step]
        iteration_rewards.append(np.mean([np.sum(episode_rewards) for episode_rewards in rewards]))
        smoothed_rewards.append(np.mean(iteration_rewards[-10:]))

        # states = torch.FloatTensor(states)
        states = torch.FloatTensor(np.array(states))   #[2000,31]
        if flag_continuous_action:
            actions = torch.FloatTensor(actions)
        else:
            actions = torch.IntTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        # print(rewards)
        average_rewards = np.mean([np.sum(episode_rewards) for episode_rewards in rewards])
        # if iteration%40==0:
        #     print('reward:',episode_rewards)
        print('Iteration:', iteration, '- Average rewards:', average_rewards)

        returns = calc_GAE(rewards)       # [10*200]

        values = critic(states).squeeze()  #[2000]
        advantage = returns - values.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)   #势归一化

        for epoch in range(SGD_EPOCHS):
            # TODO: Use a mini-batch instead of the whole batch to for gradient descent
            values = critic(states).squeeze()
            if flag_continuous_action:
                mean = actor(states)
                dist = torch.distributions.MultivariateNormal(mean, cov)      #多元正态分布，mean均值，cov协方差矩阵
            else:
                action_probs = actor(states)
                dist = torch.distributions.Categorical(action_probs)
            log_probs_new = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = (log_probs_new - log_probs).exp()  #log_prob是做动作时候的概率 new是更新时候的概率  on policy

            surr1 = ratios * advantage
            surr2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * advantage
            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (returns - values).pow(2).mean()   #return大 values小

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
            # loss = actor_loss + 0.5 * critic_loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print("--- %s seconds for training %s iterations ---" % (time.time() - start_time, TOTAL_ITERATIONS))
    
    print('---------------------------test----------------------------')
    state = env.reset()    
    for step in range(EPISODE_LENGTH):
        action, log_prob = calc_action(state)
        a1=deal_action(action[0])
        a2=deal_action(action[1])
        new_action=[4,a1,a2,7,4]
        print(env.get_price(new_action))
        next_state, reward, done= env.step(new_action)
        state = next_state
    # plot
    plt.plot(iteration_rewards, color='darkorange')  # total rewards in an iteration
    plt.plot(smoothed_rewards, color='b')  # moving avg rewards
    plt.xlabel('Iteration')
    plt.show()
    # plt.savefig('final.png')