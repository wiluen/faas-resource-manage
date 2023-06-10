from torch_geometric.data import InMemoryDataset,Data
from tqdm import tqdm
import torch
import pandas as pd
from util import RLEnvironment
DATA_PATH = 'DRL/dataset_cancel/res.csv'


class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self,root, transform=None, pre_transform=None):
        super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform) # transform就是数据强，对每一个数据都执行
        self.data,self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self): #检selfrawdir月灵下是否存在rar#如有文件不存在，则调用dowmload0
        return [] 

    @property
    def processed_file_names(self): 
        return [ 'yoochoose_click binary_1M sess.dataset' ]
    def download(self):
        pass
    def process(self):
        data_list =[]
        df = pd.read_csv(DATA_PATH)
        for index, row in df.iterrows():
            tabular_item = {  
                'cpu1': row['cpu1'],
                'mem1': row['mem1'],
                'cpu_util1': row['cpu_util1'],
                'cpu_thr1': row['cpu_thr1'],
                'mem_util1': row['mem_util1'],
                'net_rec1': row['net_rec1'],
                'net_send1': row['net_send1'],
                'latency1': row['latency1'],
                'cpu2': row['cpu2'],
                'mem2': row['mem2'],               
                'cpu_util2': row['cpu_util2'],
                'cpu_thr2': row['cpu_thr2'],
                'mem_util2': row['mem_util2'],
                'net_rec2': row['net_rec2'],
                'net_send2': row['net_send2'],
                'latency2': row['latency2'],  
                'cpu3': row['cpu3'],
                'mem3': row['mem3'],             
                'cpu_util3': row['cpu_util3'],
                'cpu_thr3': row['cpu_thr3'],
                'mem_util3': row['mem_util3'],
                'net_rec3': row['net_rec3'],
                'net_send3': row['net_send3'],
                'latency3': row['latency3'], 
                'cpu4': row['cpu4'],
                'mem4': row['mem4'],               
                'cpu_util4': row['cpu_util4'],
                'cpu_thr4': row['cpu_thr4'],
                'mem_util4': row['mem_util4'],
                'net_rec4': row['net_rec4'],
                'net_send4': row['net_send4'],
                'latency4': row['latency4'],
                'cpu5': row['cpu5'],
                'mem5': row['mem5'],
                'cpu_util5': row['cpu_util5'],
                'cpu_thr5': row['cpu_thr5'],
                'mem_util5': row['mem_util5'],
                'net_rec5': row['net_rec5'],
                'net_send5': row['net_send5'],
                'latency5': row['latency5'],
                'latency': row['latency']
            }
        
            x=[tabular_item['cpu1'],tabular_item['mem1'],tabular_item['cpu_util1'],tabular_item['cpu_thr1']/100,tabular_item['mem_util1']/100,tabular_item['net_rec1']/1000,tabular_item['net_send1']/1000,tabular_item['latency1'],
                tabular_item['cpu2'],tabular_item['mem2'],tabular_item['cpu_util2'],tabular_item['cpu_thr2']/100,tabular_item['mem_util2']/100,tabular_item['net_rec2']/1000,tabular_item['net_send2']/1000,tabular_item['latency2'],
                tabular_item['cpu3'],tabular_item['mem3'],tabular_item['cpu_util3'],tabular_item['cpu_thr3']/100,tabular_item['mem_util3']/100,tabular_item['net_rec3']/10000,tabular_item['net_send3']/1000,tabular_item['latency3'],
                tabular_item['cpu4'],tabular_item['mem4'],tabular_item['cpu_util4'],tabular_item['cpu_thr4']/100,tabular_item['mem_util4']/100,tabular_item['net_rec4']/1000,tabular_item['net_send4']/1000,tabular_item['latency4'],
                tabular_item['cpu5'],tabular_item['mem5'],tabular_item['cpu_util5'],tabular_item['cpu_thr5']/100,tabular_item['mem_util5']/100,tabular_item['net_rec5']/1000,tabular_item['net_send5']/1000,tabular_item['latency5']
                ]
            y=[tabular_item['latency1'],tabular_item['latency2'],tabular_item['latency3'],tabular_item['latency4'],tabular_item['latency5']]
            edge_index = torch.tensor([[0, 3, 1, 3, 1, 2, 1, 4],
                               [3, 0, 3, 1, 2, 1, 4, 1]], dtype=torch.long)
        # for i in range()
        # y = torch.FloatTensor([group.label.values[o]])
         
        # xy都是tensor
            data = Data(x=x,edge_index=edge_index, y=y)
            data_list.append(data)

        data,slices = self.collate(data_list)
        torch.save((data,slices),self.processed_paths[0])


b=YooChooseBinaryDataset("DRL")

# Edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)

# #每个节点的特征：从0号节点开始。。
# X = torch.tensor([[-1], [0], [1]], dtype=torch.float)
# #每个节点的标签：从0号节点开始-两类0，1
# Y = torch.tensor([[0],[1],[2]],dtype=torch.float)


# data = Data(x=X, edge_index=Edge_index,y=Y)
# print(data)
