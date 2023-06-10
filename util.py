import random
import pandas as pd

DATA_PATH = 'DRL/dataset_cancel/res.csv'
# output [0,1,2,...,7]
#  goal minimize time*cost


function_list=['calculate-refund','cancel-ticket','drawback','get-order-by-id','save-order-info']


def merge():    
    df1=pd.read_csv('DRL/dataset_cancel/calculate-refund.txt')
    df2=pd.read_csv('DRL/dataset_cancel/cancel-ticket.txt')
    df3=pd.read_csv('DRL/dataset_cancel/drawback.txt')
    df4=pd.read_csv('DRL/dataset_cancel/get-order-by-id.txt')
    df5=pd.read_csv('DRL/dataset_cancel/save-order-info.txt')
    df = pd.concat([df1, df2, df3, df4, df5], axis=1)
    df.to_csv('DRL/dataset_cancel/res.csv',index=False)

class RLEnvironment:
    observation_space=31
    action_space=2
    table={}
    latency_slo=3
    instance_price=[0.00272, 0.00368, 0.00464, 0.0056, 0.00656, 0.00752, 0.00848, 0.00944]

    def action2conf(self,action):
        config=[{"cpu":150,"mem":64},
            {"cpu":200,"mem":96},
            {"cpu":250,"mem":128},
            {"cpu":300,"mem":160},
            {"cpu":350,"mem":192},
            {"cpu":400,"mem":224},
            {"cpu":450,"mem":256},
            {"cpu":500,"mem":288}]
        
        resource_cpu=[config[action[0]]["cpu"],
                  config[action[1]]["cpu"],
                  config[action[2]]["cpu"],
                  config[action[3]]["cpu"],
                  config[action[4]]["cpu"]]
        resource_mem=[config[action[0]]["mem"],
                  config[action[1]]["mem"],
                  config[action[2]]["mem"],
                  config[action[3]]["mem"],
                  config[action[4]]["mem"]]
        return resource_cpu , resource_mem
    
    def load_data(self):
        df = pd.read_csv(DATA_PATH)
        for index, row in df.iterrows():
            tabular_item = {  
                'cpu_util1': row['cpu_util1'],  
                'cpu_thr1': row['cpu_thr1'],
                'mem_util1': row['mem_util1'],
                'net_rec1': row['net_rec1'],
                'net_send1': row['net_send1'],
                'latency1': row['latency1'],                
                'cpu_util2': row['cpu_util2'],
                'cpu_thr2': row['cpu_thr2'],
                'mem_util2': row['mem_util2'],
                'net_rec2': row['net_rec2'],
                'net_send2': row['net_send2'],
                'latency2': row['latency2'],               
                'cpu_util3': row['cpu_util3'],
                'cpu_thr3': row['cpu_thr3'],
                'mem_util3': row['mem_util3'],
                'net_rec3': row['net_rec3'],
                'net_send3': row['net_send3'],
                'latency3': row['latency3'],                
                'cpu_util4': row['cpu_util4'],
                'cpu_thr4': row['cpu_thr4'],
                'mem_util4': row['mem_util4'],
                'net_rec4': row['net_rec4'],
                'net_send4': row['net_send4'],
                'latency4': row['latency4'],
                'cpu_util5': row['cpu_util5'],
                'cpu_thr5': row['cpu_thr5'],
                'mem_util5': row['mem_util5'],
                'net_rec5': row['net_rec5'],
                'net_send5': row['net_send5'],
                'latency5': row['latency5'],
                'latency': row['latency']
            }
            key = (row['cpu1'], row['cpu2'],row['cpu3'], row['cpu4'],row['cpu5'],row['mem1'], row['mem2'],row['mem3'], row['mem4'], row['mem5'])
            self.table[key] = tabular_item
                   

    def get_price(self,action):
        function_per_price=[]
        resource_cpu , resource_mem=self.action2conf(action)
       
        for i in range(5):
            function_per_price.append(self.instance_price[action[i]]) 
        key=(resource_cpu[0],resource_cpu[1],resource_cpu[2],resource_cpu[3],resource_cpu[4],
        resource_mem[0],resource_mem[1],resource_mem[2],resource_mem[3],resource_mem[4])
        
        try:
            p1=(float(self.table[key]['latency1'])*function_per_price[0])
            p2=(float(self.table[key]['latency2'])*function_per_price[1])
            p3=(float(self.table[key]['latency3'])*function_per_price[2])
            p4=(float(self.table[key]['latency4'])*function_per_price[3])
            p5=(float(self.table[key]['latency5'])*function_per_price[4])
            cost=p1+p2+p3+p4+p5   # 在0.018-0.05的范围
        except:
            cost=0.1
        # print("resource cost: ", cost)
        return cost

    def get_rl_states(self,action):
        resource_cpu,resource_mem=self.action2conf(action)
        key=(resource_cpu[0],resource_cpu[1],resource_cpu[2],resource_cpu[3],resource_cpu[4],   
        resource_mem[0],resource_mem[1],resource_mem[2],resource_mem[3],resource_mem[4])
        # try:
        value = self.table[key]
        state=[value['cpu_util1'],value['cpu_thr1']/100,value['mem_util1']/100,value['net_rec1']/1000,value['net_send1']/1000,value['latency1'],
                value['cpu_util2'],value['cpu_thr2']/100,value['mem_util2']/100,value['net_rec2']/1000,value['net_send2']/1000,value['latency2'],
                value['cpu_util3'],value['cpu_thr3']/100,value['mem_util3']/100,value['net_rec3']/10000,value['net_send3']/1000,value['latency3'],
                value['cpu_util4'],value['cpu_thr4']/100,value['mem_util4']/100,value['net_rec4']/1000,value['net_send4']/1000,value['latency4'],
                value['cpu_util5'],value['cpu_thr5']/100,value['mem_util5']/100,value['net_rec5']/1000,value['net_send5']/1000,value['latency5'],
                value['latency']]
        # 注意：这里返回的是一个数组，不是map

        return state
    
    def reset(self):
        action=[4,4,4,7,4]
        state=self.get_rl_states(action)
        return state
    
    def get_reward(self,action):
        # penatly=0
        # if action[1]<3 and action[2]<3:
        #     penatly=1
        alpha=0.3
        beta=0.7
        price=self.get_price(action)
        # print('price:',price)
        try:
            state=self.get_rl_states(action)
            latency_measure=state[30]
        except:
            latency_measure=10
        slo_preservation=self.latency_slo/latency_measure
        # print('slo_preservation:',slo_preservation)
        reward= alpha * slo_preservation - beta * (price*10) 
        
        return reward 
    
    def step(self,action):
        # test的action是2维
        state=self.get_rl_states(action)
        reward = self.get_reward(action)
        done = False
        return state,reward,done


if __name__ == '__main__':
    print('Testing simulated environment...')
    env = RLEnvironment()
    env.load_data()
    # merge()
    

