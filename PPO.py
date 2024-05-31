import numpy as np
import random
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import vessl
from torch.distributions import Categorical
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
device = 'cuda'
print(device)

class MLP(nn.Module):
    def __init__(self, state_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256,output_size)
        self._initialize_weights()

    def _initialize_weights(self, init_min=-0.08, init_max=0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class GNN(nn.Module):
    def __init__(self, epoch):
        super(GNN, self).__init__()
        self.K = epoch
        self.f1 = MLP(8, 8)
        self.f2 = MLP(8, 8)
        self.f3 = MLP(8, 8)
        self.f4 = MLP(48, 8)

    def forward(self, x, MM, PM):
        MM = np.array(MM)
        PM = np.array(PM)
        max_step = np.max(np.array(MM), axis=1)
        # node job,step
        update = x.clone()
        init = x.clone()
        for _ in range(self.K):
            for i in range(MM.shape[0]):
                for j in range(MM.shape[1]):
                    job_step = MM[i][j]
                    if job_step == 0:
                        a1 = F.relu(self.f1(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32).to(device)))
                    else:
                        a1 = F.relu(self.f1(x[i][list(MM[i]).index(job_step - 1)]))
                    if job_step == max_step[i]:
                        a2 = F.relu(self.f2(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32).to(device)))
                    else:
                        a2 = F.relu(self.f2(x[i][list(MM[i]).index(job_step + 1)]))
                    a3 = F.relu(self.f3(x[:, j, :].sum(axis=0) - x[i, j, :]))
                    a4 = F.relu(x.sum(axis=0).sum(axis=0))
                    a5 = x[i, j]
                    a6 = init[i, j]
                    update[i][j] = self.f4(torch.cat([a1, a2, a3, a4, a5, a6], axis=0))
            x = update.clone()
        return x

class PPO(nn.Module):
    def __init__(self, K=3, learning_rate=0.001, lmbda=0.95, gamma=1, alpha=0.5, beta=0.01, clipping_parameter=0.2,
                 discount_factor=1):
        super(PPO, self).__init__()
        self.gnn = GNN(K)
        self.pi = MLP(8, 1)
        self.v = MLP(8, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.lmbda = lmbda
        self.gamma = gamma
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.beta = beta
        self.clipping_parameter = clipping_parameter

    def calculate_GNN(self, x, MM, PM):
        return self.gnn(x, MM, PM)

    def calculate_pi(self, x, job, machine):
        return self.pi(x[job][machine])

    def calculate_v(self, x):
        return self.v(x)

    def update(self, data, MM_list, PM_list):
        num = 0
        ave_loss = 0
        en_loss = 0
        v_loss = 0
        L_loss = 0
        for episode in data:
            tr = 0
            for transition in episode:
                if tr == 0:
                    state_GNN_finished = self.calculate_GNN(
                        torch.tensor(transition[0], dtype=torch.float32).to(device), MM_list[num],
                        PM_list[num]).unsqueeze(0)
                    next_state_GNN_finished = self.calculate_GNN(
                        torch.tensor(transition[5], dtype=torch.float32).to(device), MM_list[num],
                        PM_list[num]).unsqueeze(0)
                else:
                    state_GNN_finished = torch.cat([state_GNN_finished, self.calculate_GNN(
                        torch.tensor(transition[0], dtype=torch.float32).to(device), MM_list[num],
                        PM_list[num]).unsqueeze(0)])
                    next_state_GNN_finished = torch.cat([next_state_GNN_finished, self.calculate_GNN(
                        torch.tensor(transition[5], dtype=torch.float32).to(device), MM_list[num],
                        PM_list[num]).unsqueeze(0)])
                a = []
                for i in transition[2]:
                    a.append(self.calculate_pi(state_GNN_finished[tr], i[0], i[1]))

                pi = torch.softmax(torch.cat(a, dim=0), dim=0)[transition[3]]
                ratio = pi / torch.tensor(transition[4][transition[3]], dtype=torch.float32).to(device)
                entropy = -torch.sum(a * torch.log(a))
                if tr == 0:
                    ratio_list = ratio.unsqueeze(0)
                    entropy_error_list = entropy.unsqueeze(0)
                else:
                    ratio_list = torch.cat([ratio_list, ratio.unsqueeze(0)])
                    entropy_error_list = torch.cat([entropy_error_list, entropy.unsqueeze(0)])

                tr += 1
            advantage = 0.0
            v_target = 0.0
            rewards = torch.tensor([[transition[1]] for transition in episode], dtype=torch.float32).to(device)
            next_v = self.calculate_v(torch.sum(next_state_GNN_finished, dim=(1, 2)))
            current_v = self.calculate_v(torch.sum(state_GNN_finished, dim=(1, 2)))  # mini batch 떄문에

            delta = next_v - current_v + rewards

            v_target_lst = np.zeros(len(delta))
            advantage_lst = torch.tensor(np.zeros(len(delta)), dtype=torch.float32).to(device)
            advantage = torch.tensor(advantage, dtype=torch.float32).to(device)

            for t in reversed(range(0, len(delta))):
                v_target = rewards[t] + v_target
                v_target_lst[t] = v_target
                advantage = self.discount_factor * self.lmbda * advantage + delta[t][0]
                advantage_lst[t] = advantage
            v_target_lst = torch.tensor(v_target_lst, dtype=torch.float32).to(device)
            surr1 = ratio_list * advantage_lst

            surr2 = torch.clamp(ratio_list, 1 - self.clipping_parameter,
                                1 + self.clipping_parameter) * advantage_lst
            value_function_error = (current_v.view(-1) - v_target_lst) ** 2

            loss = -torch.min(surr1, surr2) + value_function_error * self.alpha - self.beta * entropy_error_list
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            en_loss += entropy_error_list.mean().detach() * self.beta
            v_loss += value_function_error.mean().detach() * self.alpha
            L_loss += torch.min(surr1, surr2).mean().detach()
            ave_loss += loss.mean().detach()
            num += 1
        en_loss = en_loss / len(data)
        L_loss = L_loss / len(data)
        v_loss = v_loss / len(data)
        ave_loss = ave_loss / len(data)
        return ave_loss.item(), L_loss.item(), v_loss.item(), en_loss.item()
