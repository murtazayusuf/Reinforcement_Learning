"""
creating a deep Q neural network to play space invaders using Pytorch
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, ALPHA):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128*19*8, 512)
        self.fc2 = nn.Linear(512, 6)

        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.optimizer)

    def forward(self, observation):
        observation = T.tensor(observation).to(self.device)
        observation = observation.view(-1, 1, 185, 95)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        observation = observation.view(-1, 128*19*8)
        observation = F.relu(self.fc1(observation))

        actions = self.fc2(observation)

        return actions

class Agent(object):
    def __init__(self, gamma, epsilon, alpha, 
                 maxMemorySize, epsEnd=0.05, 
                 replace=10000, actionspace=[1, 2, 3, 4, 5]):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.actionSpace = actionSpace
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCtr = 0
        self.replace_target_cnt = replace
        self.Q_eval = DeepQNetwork(alpha)
        self.Q_next = DeepQNetwork(alpha)

    def storeTransition(self, state, action, reward, state_):
        if self.memCtr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCtr%self.memSize] = [state, action, reward, state_]
        self.memCtr +=1

    def chooseAction(self, observation):
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        if rand < 1 - self.EPSILON:
            action = T.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.actionSpace)
        self.steps += 1
        return actions

    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()
        if self.replace_target_cnt is not None and \
            self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict)

        if self.memCtr + batch_size < self.memSize:
            memStart = int(np.random.choice(range(self.memCtr)))
        else:
            memStart = int(np.random.choice(range(self.memCtr - batch_size - 1)))
        miniBatch  = self.memory[memStart:memStart+batch_size]
        memory = np.array(miniBatch)

        Qpred = self.Q_eval.forward(list(memory[:, 0][:])).to(self.Q_eval.device)
        Qnext = self.Q_next.forward(list(memory[:, 3][:])).to(self.Q_eval.device)

        maxA = T.argmax(Qnext, dim=1).to(self.Q_eval.device)
        rewards = T.Tensor(list(memory[:,2])).to(self.Q_eval.device)
        Qtarget = Qpred
        Qtarget[:, maxA] = rewards + self.GAMMA*T.max(Qnext[1])

        if self.steps > 500:
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END

        loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1