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

class Simulator():
    def __init__(self):
        self.ppo = PPO()
        self.ppo.to(device)

    def jssp_sampling(self, machine_min=5, machine_max=9, job_max=9, processing_time_min=1, processing_time_max=10):
        m = np.random.randint(machine_min, machine_max)
        n = np.random.randint(m, job_max)
        MM = np.ndarray(shape=(n, m))
        PM = np.random.randint(processing_time_min, processing_time_max, size=(n, m))
        for i in range(n):
            MM[i] = np.random.permutation(m)
        return MM, PM

        # graph [1,0,0] 2//[processing time] 3//+[Degree of completion] 4//+[Numberofsuceedingoperation] 5//+[watiting time]+[remaining time]

    def run_episode(self, Matrix1, Matrix2, discount_factor):
        event_list = []
        MM = Matrix1
        PM = Matrix2
        graph = np.zeros((Matrix1.shape[0], Matrix1.shape[1], 8))
        graph[:, :, 0] = 1

        for i in range(Matrix1.shape[0]):
            for j in range(Matrix1.shape[1]):
                graph[i, j, 3] = PM[i][j]
                graph[i, j, 7] = -1
        max_step = np.max(np.array(MM), axis=1)  # 각 job의 max step 반환
        time = 0
        end = False
        episode = []
        reward = 0
        operation_left = max_step.sum() + MM.shape[0]

        while not end:

            if time != 0:
                reward = reward * discount_factor - len(
                    np.where((graph[:, :, 1].sum(axis=1) == 0) & (graph[:, :, 0].sum(axis=1) > 0))[
                        0])  # operating 중이지 않은 job

            graph[:, :, 7] = np.where(graph[:, :, 7] > -1, graph[:, :, 7] - 1, graph[:, :, 7])  # remaining time을 -1
            graph[:, :, 4] = np.where(graph[:, :, 7] > -1, 1 - (graph[:, :, 7] / graph[:, :, 3]),
                                      graph[:, :, 4])  # degree of comp;letion

            for i in range(MM.shape[0]):
                job_succeded = int(graph[i, 0, 5])  # 현재 해야할 job
                if time != 0:
                    if graph[i, :, 1].sum() == 0 and job_succeded <= max_step[i]:
                        ind = list(MM[i]).index(job_succeded)
                        graph[i, ind, 6] += 1

            graph[:, :, 5] = np.where(graph[:, :, 7] == 0, graph[:, :, 5] + 1,
                                      graph[:, :, 5])  # number of succeding operation

            for i in range(MM.shape[0]):
                graph[i, :, 5] = np.max(graph[i, :, 5])
            graph[:, :, 1] = np.where(graph[:, :, 7] == 0, 0, graph[:, :, 1])
            graph[:, :, 2] = np.where(graph[:, :, 7] == 0, 1, graph[:, :, 2])

            operation_left -= len(np.where(graph[:, :, 7] == 0)[0])

            if operation_left == 0:
                make_span = time
                end = True

            else:
                idle_list = []
                idle_machine = list(np.where(graph[:, :, 1].sum(axis=0) == 0)[0])  # 비는 machine 탐색
                for i in idle_machine:
                    for k in range(MM.shape[0]):
                        # machine i 가 처리하는 operation중 job step이 현재 해야하는 step과 일치하는 job 찾기
                        if MM[k][i] == graph[k, 0, 5]:
                            idle_list.append([k, i])
                while len(idle_list) != 0:
                    policy = np.array([])
                    for i in idle_list:
                        with torch.no_grad():
                            policy = np.append(policy, self.ppo.calculate_pi(
                                self.ppo.calculate_GNN(torch.tensor(graph, dtype=torch.float32).to(device), MM, PM),
                                i[0], i[1]).item())
                    policy = np.exp(policy - np.max(policy))
                    policy = policy / np.sum(policy)
                    select = random.choices(range(0, len(idle_list)), weights=policy)[0]
                    data = []
                    data.append(graph.copy())
                    event_list.append([time, idle_list[select][0], idle_list[select][1],
                                       PM[idle_list[select][0]][idle_list[select][1]]])
                    graph[idle_list[select][0], idle_list[select][1], 0] = 0
                    graph[idle_list[select][0], idle_list[select][1], 1] = 1
                    graph[idle_list[select][0], idle_list[select][1], 7] = graph[
                        idle_list[select][0], idle_list[select][1], 3]
                    reward = reward * discount_factor - len(
                        np.where((graph[:, :, 1].sum(axis=1) == 0) & (graph[:, :, 0].sum(axis=1) > 0))[0])

                    data.append(reward / 3)
                    data.append(idle_list)
                    data.append(select)
                    data.append(policy)
                    reward = 0
                    data.append(graph.copy())
                    episode.append(data)
                    # state, reward, action_list, select, policy, next_state
                    idle_list = []
                    idle_machine = list(np.where(graph[:, :, 1].sum(axis=0) == 0)[0])  # 비는 machine 탐색
                    for i in idle_machine:
                        for k in range(MM.shape[0]):
                            # machine i 가 처리하는 operation중 job step이 현재 해야하는 step과 일치하는 job 찾기
                            if MM[k][i] == graph[k, 0, 5]:
                                idle_list.append([k, i])

            time += 1
        return episode, event_list, make_span

    def train(self, episode_num=20, update_num=5, epoch=50):
        loss_list = []
        L_loss_list = []
        v_loss_list = []
        en_loss_list = []
        ave_make_span_list = []
        print('start!')
        time1 = time.time()
        num = 0
        for j in range(epoch):
            MM_list = []
            PM_list = []
            epi_loss = 0
            for __ in range(episode_num):
                MM, PM = self.jssp_sampling()
                MM_list.append(MM)
                PM_list.append(PM)
            for k in range(update_num):
                ave_make_span = 0
                data = []
                for i in range(episode_num):
                    episode, event_list, make_span = self.run_episode(MM_list[i], PM_list[i], 1)
                    ave_make_span += make_span
                    data.append(episode)
                ave_make_span = ave_make_span / 20
                print('episode generate:', time.time() - time1)
                ave_loss, L_loss, v_loss, en_loss = self.ppo.update(data, MM_list, PM_list)
                print("epoch:", j, 'update:', k, "ave_loss:", ave_loss)
                print("epoch:", j, 'update:', k, "L_loss:", L_loss)
                print("epoch:", j, 'update:', k, "v_loss:", v_loss)
                print("epoch:", j, 'update:', k, "en_loss:", en_loss)
                print("epoch:", j, 'update:', k, "ave_make_span:", ave_make_span)

                time2 = time.time()

                print("total time of update:", time2 - time1)
                time1 = time2
                loss_list.append(ave_loss)
                L_loss_list.append(L_loss)
                v_loss_list.append(v_loss)
                en_loss_list.append(en_loss)
                ave_make_span_list.append(ave_make_span)
                vessl.log(step=num, payload={'make_span': ave_make_span})

                vessl.log(step=num, payload={'ave_loss': ave_loss})

                vessl.log(step=num, payload={'L_loss': L_loss})

                vessl.log(step=num, payload={'v_loss': v_loss})

                vessl.log(step=num, payload={'en_loss': en_loss})
                num += 1
            if j % 5 == 0 and j > 0:
                name = 'rrf' + str(j) + '.pt'
                torch.save(self.ppo, '/output/' + name)
                self.validate_model()
        return loss_list, L_loss_list, v_loss_list, en_loss_list, ave_make_span_list

    def plot_gantt_chart(self, events, job_max_num, machine_max_num):
        """

        events=[start_time, job_num, machine_num, processing time]

        """

        # variable n below should be number of curves to plot

        # version 1:
        colorset = plt.cm.rainbow(np.linspace(0, 1, machine_max_num))
        # Set up figure and axis
        fig, ax = plt.subplots()

        # Plot Gantt chart bars
        for event in events:
            job_start = event[0]
            job_end = event[0] + event[3]
            ax.barh(y=event[1], width=job_end - job_start, left=job_start, height=0.6, label=f'machine {event[2] + 1}',
                    color=colorset[event[2]])

        # Customize the plot
        ax.set_xlabel('Time')
        ax.set_yticks(range(job_max_num))
        ax.set_yticklabels([f'Job {i + 1}' for i in range(job_max_num)])

        # Show the plot
        plt.show()

    def validate_model(self):
        machine_matrix = np.array([[2, 0, 1, 3, 5, 4],
                                   [1, 2, 4, 5, 0, 3],
                                   [2, 3, 5, 0, 1, 4],
                                   [1, 0, 2, 3, 4, 5],
                                   [2, 1, 4, 5, 0, 3],
                                   [1, 3, 5, 0, 4, 2]])

        processing_time = np.array([[1., 3., 6., 7., 3., 6.],
                                    [8., 5., 10., 10., 10., 4.],
                                    [5., 4., 8., 9., 1., 7.],
                                    [5., 5., 5., 3., 8., 9.],
                                    [9., 3., 5., 4., 3., 1.],
                                    [3., 3., 9., 10., 4., 1.]])
        MM = machine_matrix
        PM = processing_time
        ep1 = []
        ep2 = []
        for i in range(10):
            episode, event_list, make_span = self.run_episode(MM, PM, 1)
            ep1.append(event_list)
            ep2.append(make_span)
        ave_make_span = sum(ep2) / 10.0
        min_index = ep2.index(min(ep2))
        event_list = ep1[min_index]
        print("ave make span is", ave_make_span)


# In[4]:


sim = Simulator()

# In[ ]:


Loss_list, L_loss_list, v_loss_list, en_loss_list, ave_make_span_list = sim.train()