import time
import numpy as np
import gym
import torch
from torch import nn
import matplotlib.pyplot as plt
from util import RLEnvironment
from torch_geometric.nn import GCNConv,global_mean_pool
import torch.nn.functional as F
class ForwardNetwork(nn.Module):
    def __init__(self, graph_embeddings_size, hidden_size, output_size):
        super(ForwardNetwork, self).__init__()
        self.conv1=GCNConv(6,4)
        self.pool=global_mean_pool
        self.fc1 = nn.Linear(graph_embeddings_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.tanh=F.tanh
        self.dropout=F.dropout
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,edge_index):
        # Weichao: maybe reduce one fc layer?
        # input = torch.FloatTensor(input)
        x=self.conv1(x,edge_index)
        x=self.tanh(x)
        x=self.dropout(x)
        # batch = torch.tensor([1])
        graph_embeddings = global_mean_pool(x,batch=None)   #batch = 1
        output = self.relu(self.fc1(graph_embeddings))
        # output = self.relu(self.fc2(output))
        output = self.fc3(output)
        return output

# 把state换成 x和edge_index
def calc_action(x,edge_index):
    if flag_continuous_action:
        mean = actor(x,edge_index)
        dist = torch.distributions.MultivariateNormal(mean, cov)  #多元正太分布
    else:
        action_probs = actor(x,edge_index)
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

def state2graphinfo(state):
    x = torch.tensor([
    [state[0],state[1],state[2],state[3],state[4],state[5]],  # 节点0的特征
    [state[6],state[7],state[8],state[9],state[10],state[11]],   # 节点1的特征     
    [state[12],state[13],state[14],state[15],state[16],state[17]],   # 节点2的特征 
    [state[18],state[19],state[20],state[21],state[22],state[23]],   # 节点3的特征
    [state[24],state[25],state[26],state[27],state[28],state[29]]    # 节点4的特征
    ])
    return x
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
    TOTAL_ITERATIONS = 500             # 400
    EPISODES_PER_ITERATION = 10           # 10
    EPISODE_LENGTH = 200               # 200
    SGD_EPOCHS = 5
    CLIP = 0.2
    GRAPH_EMBEDDINGS_SIZE=4
    edge_index = torch.tensor([[0, 3, 1, 3, 1, 2, 1, 4],
                               [3, 0, 3, 1, 2, 1, 4, 1]], dtype=torch.long)

    state_size = env.observation_space
    if flag_continuous_action:
        action_size = env.action_space
    else:
        action_size = env.action_space

    actor = ForwardNetwork(GRAPH_EMBEDDINGS_SIZE, HIDDEN_SIZE, 2)
    critic = ForwardNetwork(GRAPH_EMBEDDINGS_SIZE, HIDDEN_SIZE, 1)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)       # 居然是一起优化的

    cov = torch.diag(torch.ones(action_size, ) * 0.5)

    # for plots
    iteration_rewards = []
    smoothed_rewards = []

    for iteration in range(TOTAL_ITERATIONS):
        states = []
        xs= []
        actions = []
        rewards = []
        log_probs = []

        for episode in range(EPISODES_PER_ITERATION):
            state = env.reset()
            x=state2graphinfo(state)
            episode_rewards = []

            for step in range(EPISODE_LENGTH):
                action, log_prob = calc_action(x,edge_index)   #当前的action和prob
                # print(action[0][0],action[0][1],log_prob)
                a1=deal_action(action[0][0])
                a2=deal_action(action[0][1])
                new_action=[4,a1,a2,7,4]

                #做一个 固定策略 +  随着训练衰减转为探索s

                # print(new_action)
                next_state, reward, done= env.step(new_action)
                next_x=state2graphinfo(next_state)
                xs.append(x)
                episode_rewards.append(reward)
                log_probs.append(log_prob)
                actions.append(action)

                if done:
                    break
                x = next_x

            rewards.append(episode_rewards)   # 应该是每个episode的奖励，二维数组，每行代表了这一个episode每步的奖励
        # print(rewards)  [10*200]
        iteration_rewards.append(np.mean([np.sum(episode_rewards) for episode_rewards in rewards]))
        smoothed_rewards.append(np.mean(iteration_rewards[-10:]))

        # states = torch.FloatTensor(states)
        # xs = torch.FloatTensor(np.array(xs))  #[2000,31]
        
        if flag_continuous_action:
            actions = torch.FloatTensor(actions)
        else:
            actions = torch.IntTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        # print(rewards)   
        average_rewards = np.mean([np.sum(episode_rewards) for episode_rewards in rewards])
        print('Iteration:', iteration, '- Average rewards:', average_rewards)
        value_tensor = torch.empty(2000)
        returns = calc_GAE(rewards)    # [2000]
        for i in range(2000):
            values = critic(xs[i],edge_index).squeeze()  # [2000]
            value_tensor[i]=values
        advantage = returns - value_tensor.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)   #势归一化

        for epoch in range(SGD_EPOCHS):            # 更新参数！！！
            # TODO: Use a mini-batch instead of the whole batch to for gradient descent
            log_probs_new_tensor = torch.empty(2000)
            values_tensor = torch.empty(2000)
            
            for i in range(2000):
                values = critic(xs[i],edge_index).squeeze()
                values_tensor[i]=values
                mean = actor(xs[i],edge_index)
                dist = torch.distributions.MultivariateNormal(mean, cov)      #多元正态分布，mean均值，cov协方差矩阵
                log_probs_new = dist.log_prob(actions[i])
                # entropy = dist.entropy().mean()
                log_probs_new_tensor[i]=log_probs_new

            ratios = (log_probs_new_tensor - log_probs).exp()

            surr1 = ratios * advantage
            surr2 = torch.clamp(ratios, 1 - CLIP, 1 + CLIP) * advantage
            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (returns - values).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print("--- %s seconds for training %s iterations ---" % (time.time() - start_time, TOTAL_ITERATIONS))

    # plot
    plt.plot(iteration_rewards, color='darkorange')  # total rewards in an iteration
    plt.plot(smoothed_rewards, color='b')  # moving avg rewards
    plt.xlabel('Iteration')
    plt.show()
    # plt.savefig('final.png')