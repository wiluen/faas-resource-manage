import os
import datetime
import logging
from run_load_test import run
import time
import random
from request_cancel import single_request_cancel
from request_search import single_request_search
from request_preserve import single_request_preserve
from request_fc import single_request_fc
from request_ml import single_request_ml
import re
from subprocess import Popen, PIPE, STDOUT
import requests
import numpy as np


NUM_RESOURCES=2
CPU_UNIT_COST=0.000173   #0.173/1000
MEM_UNIT_COST=0.000012    #0.0123/1024
PENALTY=10

# image-function{
CPU_MIN=500
CPU_MAX=1000
MEM_MIN=64
MEM_MAX=288
CPU_MIN_RESNET=800
CPU_MAX_RESNET=2000
MEM_MIN_RESNET=400
MEM_MAX_RESNET=1024
SLO=7.5
# }

# search{
# CPU_MIN=200
# CPU_MAX=500
# MEM_MIN=64
# MEM_MAX=288
# SLO=1.5
# }

# function-chain{
# CPU_MIN=50
# CPU_MAX=200
# MEM_MIN=32
# MEM_MAX=64
# SLO=1.5
# }

WARM_TIMES=1  # ms 5,8
TEST_TIMES=5

def data2txt_wo_metric(function_list,x,per_func_latency,end2end_latency,total_price):
    # for f in function_list:
        value=[x,per_func_latency,end2end_latency,total_price]
        with open('/home/user/code/faas-resource/BO_test/newprice/ml/data.txt','a') as f:
            f.write(str(value).strip('[]')+"\n")


def data2txt(function_list,cpu,mem,per_func_latency,end2end_latency):
    #需要一个配置字典，格式{"name":"function","cpu":512,"mem":256}
  
    cpu_container_sql='sum(rate(container_cpu_usage_seconds_total{name=~".+",namespace="openfaas-fn"}[1m])) by (pod) * 100'
    cpu_throttle_sql='sum(increase(container_cpu_cfs_throttled_periods_total{namespace="openfaas-fn"}[1m])) by(pod)'
    mem_container_sql='sum(container_memory_working_set_bytes{namespace="openfaas-fn",container!= "", container!="POD"}) by (pod) / sum(container_spec_memory_limit_bytes{namespace="openfaas-fn",container!= "", container!="POD"}) by (pod) * 100'
    rec_comtainer_sql='sum(rate(container_network_receive_bytes_total{name=~".+",namespace="openfaas-fn"}[1m])) by (pod)'
    transm_container_sql='sum(rate(container_network_transmit_bytes_total{name=~".+",namespace="openfaas-fn"}[1m])) by (pod)'

    url = "http://33.33.33.99:31090/api/v1/query"

    cpures = requests.get(
        url=url,
        params={'query':cpu_container_sql}
        )
    thrres = requests.get(
        url=url,
        params={'query':cpu_throttle_sql}
        )
    memres = requests.get(
        url=url,
        params={'query':mem_container_sql}
        )
    recres = requests.get(
        url=url,
        params={'query':rec_comtainer_sql}
        )
    sendres = requests.get(
        url=url,
        params={'query':transm_container_sql}
        )
    # print(cpures.json()["data"]["result"])  
    # print(thrres.json()["data"]["result"])
     #{'metric': {'pod': 'check-security-69b78cd5d5-h5dz9'}, 'value': [1686813639.653, '0.2119167324999971']}
    cpu_map={}
    thr_map={}
    mem_map={}
    network_rec_map={}
    network_send_map={}

    for value in cpures.json()["data"]["result"]:
        for i in function_list:   
            if str(i) in value["metric"]["pod"]:  
                cpu_map[str(i)]=value["value"][1]
                break

    for value in thrres.json()["data"]["result"]:
        for i in function_list:   
            if str(i) in value["metric"]["pod"]:
                thr_map[str(i)]=value["value"][1]
                break

    for value in memres.json()["data"]["result"]:
        for i in function_list:   
            if str(i) in value["metric"]["pod"]:
                mem_map[str(i)]=value["value"][1]
                break

    for value in recres.json()["data"]["result"]:
        for i in function_list:   
            if str(i) in value["metric"]["pod"]:
                network_rec_map[str(i)]=value["value"][1]
                break

    for value in sendres.json()["data"]["result"]:
        for i in function_list:   
            if str(i) in value["metric"]["pod"]:
                network_send_map[str(i)]=value["value"][1]   
                break
    # print("cpu_map:",cpu_map)
    # print("thr_map:",thr_map)
    # print("mem_map:",mem_map)
    # print("network_rec_map:",network_rec_map)
    # print("network_send_map:",network_send_map)
    
    # 每个有map有13项，分别是每个函数
    result=[]
    header = ['function','cpu','mem','cpu_util','cpu_throttle','mem_util' ,'network_transmit_bytes', 'network_receive_bytes','func_latency','latency']
    #原来，写多个文件
    for i,k in enumerate(cpu_map):      # 会有空的情况，直接报错
        value=[k,int(cpu[k]),int(mem[k]),int(float(cpu_map[k])*100)/100,int(float(thr_map[k])), int(float(mem_map[k])*100)/100, 
               int(float(network_rec_map[k])), int(float(network_send_map[k])),per_func_latency[k],end2end_latency]
        with open('/home/user/code/faas-resource/BO_test/01/'+str(k)+'.txt','a') as f:
            f.write(str(value).strip('[]')+"\n")

   
    

#更新yaml文件，重新部署函数
def update_deploy(function,resource_config):  #list
    for i in range(len(function)):
        # cmd= "/bin/bash /home/user/code/updateOpenfaasYaml.sh " + function[i] + " " + str(cpu[i]) + " " + str(mem[i])
        cmd= "/bin/bash /home/user/code/updateOpenfaasYaml.sh " + function[i] + " " + str(resource_config[i][0]) + " " + str(resource_config[i][1])
        print(cmd)
        os.system(cmd)
        # static_function=['get-price-by-routeid-and-traintype','get-route-by-tripid',
        #       'get-sold-tickets','get-traintype-by-traintypeid','get-traintype-by-tripid',
        #       'query-already-sold-orders','query-config-entity-by-config-name']
        # for f in static_function:
        #     cmd= "/bin/bash /home/user/code/faas-resource/updateself.sh " + f 
        #     os.system(cmd)
    return True

def update_deploy_fc(resource_config):  #list
    # for i in range(len(function)):
        # cmd= "/bin/bash /home/user/code/updateOpenfaasYaml.sh " + function[i] + " " + str(cpu[i]) + " " + str(mem[i])
    cmd= "/bin/bash /home/user/code/faas-resource/update_fc.sh " + str(resource_config[0][0]) + " " + str(resource_config[0][1])+ " " + str(resource_config[1][0]) + " " + str(resource_config[1][1])
    os.system(cmd)      
    return True

def get_per_func_latency(function):
    latency={}
    os.system("rm logs")    #  这个地方的路径是/home/user/code
    for i in function:
        os.system("bash /home/user/code/getLogs.sh " + i +">> logs" )
#   读logs时间就好了
    j=0
    logs = open("/home/user/code/logs",'r')
    for line in logs:
        latency[function[j]]=float(0)
        if line:  #这行存在
            dur=line.strip('()s\n') 
            if dur: 
                latency[function[j]]=float(dur)
        j+=1
    n=len(function)
    if j!=n:
        for rest in range(j,n):
            latency[function[rest]]=float(0)
    return latency

def get_per_func_latency_search():
    latency={}
    # call_times=[2,1,1,8,2,2,4,2,1,2,10,1]
    # lines=[10,5,5,20,6,5,10,5,5,5,40,5]
    call_times=[2,1,8,10,1]
    lines=[4,2,20,40,2]
    # function=['get-left-ticket-of-interval','get-left-trip-tickets','get-price-by-routeid-and-traintype','get-route-by-routeid','get-route-by-tripid',
    #           'get-sold-tickets','get-traintype-by-traintypeid','get-traintype-by-tripid','query-already-sold-orders','query-config-entity-by-config-name',
    #           'query-for-station-id-by-station-name','query-for-travel']
    function=['get-left-ticket-of-interval','get-left-trip-tickets','get-route-by-routeid','query-for-station-id-by-station-name','query-for-travel']
    # os.system("rm funclogs")    #  这个地方的路径是/home/user/code
    perfunc_latency=[]
    perfunc_latency_total=[]
    i=0
    for f in function:
        os.system("bash /home/user/code/faas-resource/getLogs2.sh " + f +" "+ str(lines[i]) +" > /home/user/code/faas-resource/BO_test/01/log/funclogs-" + f )
#   读logs时间就好了
        log =open("/home/user/code/faas-resource/BO_test/01/log/funclogs-"+ f,"r")
        time_pattern = '\(\d+\.\d+s\)'
        for line in log:
            all_times = re.findall(time_pattern, line)   #日志中所有时间
            last_times = all_times[-call_times[i]:]     #属于这次调用的所有时间
            res=0
            for s in last_times:
                num = float(re.search('\d+\.\d+', s).group(0))    #拆(0.001s)
                res += num
            perfunc_latency.append(res)
            latency[function[i]]=float(res) 
        i+=1   
    return latency,perfunc_latency       #一个字典，一个数组

def get_per_func_latency_cancel():
    latency={}
    call_times=[1,1,1,2,1]
    lines=[2,2,2,5,2]
    function=['calculate-refund','cancel-ticket','drawback','get-order-by-id','save-order-info']
    # os.system("rm funclogs")    #  这个地方的路径是/home/user/code
    perfunc_latency=[]
    perfunc_latency_total=[]
    i=0
    for f in function:
        os.system("bash /home/user/code/faas-resource/getLogs2.sh " + f +" "+ str(lines[i]) +" > /home/user/code/faas-resource/function_logs/funclogs-" + f )
#   读logs时间就好了
        log =open("/home/user/code/faas-resource/function_logs/funclogs-"+ f,"r")
        time_pattern = '\(\d+\.\d+s\)'
        for line in log:
            all_times = re.findall(time_pattern, line)
            last_times = all_times[-call_times[i]:]
            res=0
            for s in last_times:
                num = float(re.search('\d+\.\d+', s).group(0))
                res += num
            perfunc_latency.append(res)
            latency[function[i]]=float(res) 
        i+=1   
    return latency

def get_per_func_latency_preserve(function):
    latency={}
    call_times=[1,1,1,1,1,1,1]
    lines=[3,3,3,3,3,3,3]
    function=['check-security-about-order','create-order','dipatch-seat',
                       'get-contacts-by-contactsid','get-trip-all-detai-info','preserve-ticket','check-security']
    # function=['check-security']
    # os.system("rm funclogs")    #  这个地方的路径是/home/user/code
    perfunc_latency=[]
    perfunc_latency_total=[]
    i=0
    for f in function:
        os.system("bash /home/user/code/faas-resource/getLogs2.sh " + f +" "+ str(lines[i]) +" > /home/user/code/faas-resource/function_logs/02/funclogs-" + f )
#   读logs时间就好了
        log =open("/home/user/code/faas-resource/function_logs/02/funclogs-"+ f,"r")
        time_pattern = '\(\d+\.\d+s\)'
        for line in log:
            all_times = re.findall(time_pattern, line)
            last_times = all_times[-call_times[i]:]
            res=0
            for s in last_times:
                num = float(re.search('\d+\.\d+', s).group(0))
                res += num
            perfunc_latency.append(res)
            latency[function[i]]=float(res) 
        i+=1   
    print(latency)
    return latency       
    # j=0
    # logs = open("/home/user/code/logs",'r')
    # for line in logs:
    #     latency[function[j]]=float(0)
    #     if line:  #这行存在
    #         dur=line.strip('()s\n') 
    #         if dur: 
    #             latency[function[j]]=float(dur)
    #     j+=1
    # n=len(function)
    # if j!=n:
    #     for rest in range(j,n):
    #         latency[function[rest]]=float(0)
    # return latency
def get_per_func_latency_ml():
    latency={}
    call_times=[1,1,1,1,1]
    lines=[1,1,1,1,1]
    function=['starter','load','resize','update','resnet']
    # os.system("rm funclogs")    #  这个地方的路径是/home/user/code
    perfunc_latency=[]
    perfunc_latency_total=[]
    i=0
    for f in function:
        os.system("bash /home/user/code/faas-resource/getLogs2.sh " + f +" "+ str(lines[i]) +" > /home/user/code/faas-resource/function_logs/ml/funclogs-" + f )
#   读logs时间就好了
        log =open("/home/user/code/faas-resource/function_logs/ml/funclogs-"+ f,"r")
        # time_pattern = '\(\d+\.\d+s\)'                train ticket中
        time_pattern="Duration:\s*(\d+\.\d+)s"
        for line in log:
            all_times = re.findall(time_pattern, line)
            last_times = all_times[-call_times[i]:]
            res=0
            for s in last_times:
                num = float(re.search('\d+\.\d+', s).group(0))
                res += num
            perfunc_latency.append(res)
            latency[function[i]]=float(res) 
        i+=1   
    return latency,perfunc_latency  # map,array

def get_per_func_latency_fc():
    latency={}
    # call_times=[2,1,1,8,2,2,4,2,1,2,10,1]
    # lines=[10,5,5,20,6,5,10,5,5,5,40,5]
    call_times=[1,1,1]
    lines=[1,1,1]
    # function=['get-left-ticket-of-interval','get-left-trip-tickets','get-price-by-routeid-and-traintype','get-route-by-routeid','get-route-by-tripid',
    #           'get-sold-tickets','get-traintype-by-traintypeid','get-traintype-by-tripid','query-already-sold-orders','query-config-entity-by-config-name',
    #           'query-for-station-id-by-station-name','query-for-travel']
    function=['loadhtml','matchregex']
    # os.system("rm funclogs")    #  这个地方的路径是/home/user/code
    perfunc_latency=[]
    perfunc_latency_total=[]
    i=0
    for f in function:
        os.system("bash /home/user/code/faas-resource/getLogs2.sh " + f +" "+ str(lines[i]) +" > /home/user/code/faas-resource/BO_test/function-chain/log/funclogs-" + f )
#   读logs时间就好了
        log =open("/home/user/code/faas-resource/BO_test/function-chain/log/funclogs-"+ f,"r")
        if f=='loadhtml':
            time_pattern = r'\d+\.\d+ms'
        else:
            time_pattern = r'\d+ms'
        for line in log:
            all_times = re.findall(time_pattern, line)   #日志中所有时间
            last_times = all_times[-call_times[i]:]     #属于这次调用的所有时间
            res=0
            print(last_times)
            for s in last_times:
                num = float(re.search('\d+', s).group(0))   
                res += num
            perfunc_latency.append(float(res/1000))
            latency[function[i]]=float(res) 
        i+=1   
    return latency,perfunc_latency       #一个字典，一个数组

def get_end2end_latency():
    # with open("/home/user/code/faas-resource/locust_log/locustfile_search.log","r") as f:
    #     last_line=f.readlines()[-1]
    # response_time=float(last_line.split('\"')[-1][2:8])  
    with open("/home/user/code/faas-resource/locust_log/locustfile_cancel.log","r") as f:
        last_line=f.readlines()[-1]
    response_time=float(last_line.split('\":')[1][0:7])

          #读取locust文件中的response time 字符处理
    return response_time

    
def launch_test_bo(conf,function_list,resource_config):
    update_deploy_fc(resource_config)
    print('wait function teminating...')
    time.sleep(10)
    latency=[]
    n=TEST_TIMES+WARM_TIMES
    for i in range(n):
        end2end_latency=single_request_fc()  
        if i>=WARM_TIMES:
            latency.append(end2end_latency)
    latency=np.array(latency)
    return latency.mean()



def form_x_to_resource_conf(x):
    num_functions = int(len(x) / NUM_RESOURCES)
    resource_config = [[CPU_MIN, MEM_MIN] for _ in range(num_functions)]
    for i in range(int(len(x) / 2) - 1):
        scaled_cpu = x[i * 2]
        scaled_memory = x[i * 2 + 1]
        resource_config[i][0] = round(scaled_cpu * (CPU_MAX - CPU_MIN) + CPU_MIN, 0)
        resource_config[i][1] = round(scaled_memory * (MEM_MAX - MEM_MIN) + MEM_MIN, 0)
     # resnet资源区间不同
    scaled_cpu_resnet = x[(num_functions-1) * 2]
    scaled_memory_resnet = x[(num_functions-1) * 2 + 1]
    resource_config[(num_functions-1)][0] = round(scaled_cpu_resnet * (CPU_MAX_RESNET - CPU_MIN_RESNET) + CPU_MIN_RESNET, 0)
    resource_config[(num_functions-1)][1] = round(scaled_memory_resnet * (MEM_MAX_RESNET - MEM_MIN_RESNET) + MEM_MIN_RESNET, 0)
    return resource_config

# def form_x_to_resource_conf(x):
#     num_functions = int(len(x) / NUM_RESOURCES)
#     resource_config = [[CPU_MIN, MEM_MIN] for _ in range(num_functions)]
#     for i in range(num_functions):
#         scaled_cpu = x[i * 2]
#         scaled_memory = x[i * 2 + 1]
#         resource_config[i][0] = round(scaled_cpu * (CPU_MAX - CPU_MIN) + CPU_MIN, 0)
#         resource_config[i][1] = round(scaled_memory * (MEM_MAX - MEM_MIN) + MEM_MIN, 0)
#      # resnet资源区间不同
#     return resource_config

def get_latency(x):
    
    # function_list=['starter','load','resize','update','resnet']
    # function_list=['get-left-ticket-of-interval','get-left-trip-tickets','get-price-by-routeid-and-traintype','get-route-by-routeid','get-route-by-tripid',
    #           'get-sold-tickets','get-traintype-by-traintypeid','get-traintype-by-tripid','query-already-sold-orders','query-config-entity-by-config-name',
    #           'query-for-station-id-by-station-name','query-for-travel']
    function_list=['get-left-ticket-of-interval','get-left-trip-tickets','get-route-by-routeid','query-for-station-id-by-station-name','query-for-travel']
    resource_config=form_x_to_resource_conf(x)
    end2end_latency=launch_test_bo(x,function_list,resource_config) 
    # if end2end_latency>=SLO:
    #         total_price=end2end_latency * float(end2end_latency/SLO) * PENALTY  
    latency_map,latency_array=get_per_func_latency_search()
    total_price=0
    for i, config in enumerate(resource_config):
        cpu, mem = config
        duration = latency_array[i]
        if duration==0:
            func_price=1
        else:
            func_price = duration * cpu * CPU_UNIT_COST + duration * mem * MEM_UNIT_COST
        total_price += func_price
    # trick1 SLO penalty
    if end2end_latency>=SLO:
            total_price=total_price * float(end2end_latency/SLO) * PENALTY    
    data2txt_wo_metric(function_list,x,latency_array,end2end_latency,total_price) 
    print("===========================TEST END==============================="+"\n")
    return total_price

def get_price_fc(x):
    function_list=['loadhtml','matchregex','jsonpage']
    # function_list=['get-left-ticket-of-interval','get-left-trip-tickets','get-route-by-routeid','query-for-station-id-by-station-name','query-for-travel']
    # total_price=[]
    resource_config=form_x_to_resource_conf(x)
    end2end_latency=launch_test_bo(x,function_list,resource_config) 
    time.sleep(2)     # wait for log writing
    latency_map,latency_array=get_per_func_latency_fc()
    total_price=0
    for i, config in enumerate(resource_config):
        cpu, mem = config
        duration = latency_array[i]
        if duration==0:
            func_price=1
        else:
            func_price = duration * cpu * CPU_UNIT_COST + duration * mem * MEM_UNIT_COST
        total_price += func_price
    # trick1 SLO penalty
    # if end2end_latency>=SLO:
            # total_price=total_price * float(end2end_latency/SLO) * PENALTY    
    data2txt_wo_metric(function_list,x,latency_array,end2end_latency,total_price)
    print("===========================TEST END==============================="+"\n")
    return total_price

def get_price_ml(x):
    function_list=['starter','load','resize','update','resnet']
    # function_list=['get-left-ticket-of-interval','get-left-trip-tickets','get-route-by-routeid','query-for-station-id-by-station-name','query-for-travel']
    price_truth=[]
    e2e_latency_truth=[]
    resource_config=form_x_to_resource_conf(x)
    update_deploy(function_list,resource_config)
    print('wait function teminating...')
    time.sleep(30)
    for iter in range(8):    #10
        print(f'第{iter}次测试')
        end2end_latency= single_request_ml()
        # time.sleep(2)   
        e2e_latency_truth.append(end2end_latency)
        latency_map,latency_array=get_per_func_latency_ml()
        total_price=0
        for i, config in enumerate(resource_config):
            cpu, mem = config
            duration = latency_array[i]
            if duration==0:
                func_price=1
            else:
                func_price = duration * cpu * CPU_UNIT_COST + duration * mem * MEM_UNIT_COST
            total_price += func_price
       
        price_truth.append(total_price) 

      # average  
    price_truth=np.array(price_truth)
    e2e_latency_truth=np.array(e2e_latency_truth)
    e2e=e2e_latency_truth.mean()
    price=price_truth.mean()
    if e2e>SLO:
        price=price * float(e2e/SLO) * PENALTY
    data2txt_wo_metric(function_list,x,latency_array,e2e,price)
    print("===========================TEST END==============================="+"\n")
    return price

def get_price_ms(x):
    # function_list=['starter','load','resize','update','resnet']
    function_list=['get-left-ticket-of-interval','get-left-trip-tickets','get-route-by-routeid','query-for-station-id-by-station-name','query-for-travel']
    # total_price=[]
    resource_config=form_x_to_resource_conf(x)
    # end2end_latency=launch_test_bo(x,function_list,resource_config) 
    update_deploy(function_list,resource_config)
    time.sleep(10)
    latency=[]
    n=TEST_TIMES+WARM_TIMES
    for i in range(n):
        end2end_latency=single_request_search()  
        if i>=WARM_TIMES:    #开始正式测试
            latency_map,latency_array=get_per_func_latency_search()
            latency.append(latency_array)   #总的，要取平均
    average_price=[]
    for x in range(WARM_TIMES):
        total_price=0
        for i, config in enumerate(resource_config):
            cpu, mem = config
            duration = latency[x][i]
            if duration==0:
                func_price=1
            else:
                func_price = duration * cpu * CPU_UNIT_COST + duration * mem * MEM_UNIT_COST
            total_price += func_price
        average_price.append(total_price)
    average_price=np.array(average_price)
    print(average_price)
    price=average_price.mean()
    # trick1 SLO penalty
    # if end2end_latency>=SLO:
            # total_price=total_price * float(end2end_latency/SLO) * PENALTY    
    data2txt_wo_metric(function_list,x,latency_array,end2end_latency,price)
    print("===========================TEST END==============================="+"\n")
    return price
