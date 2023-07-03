import os
import datetime
import logging
from run_load_test import run
import time
import random
from request_cancel import single_request_cancel
from request_search import single_request_search
import re
from subprocess import Popen, PIPE, STDOUT
import requests

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
    
    cpu_map={}
    thr_map={}
    mem_map={}
    network_rec_map={}
    network_send_map={}

    for value in cpures.json()["data"]["result"]:
        for i in function_list:   
            if str(i) in value["metric"]["pod"]:
                cpu_map[str(i)]=value["value"][1]

    for value in thrres.json()["data"]["result"]:
        for i in function_list:   
            if str(i) in value["metric"]["pod"]:
                thr_map[str(i)]=value["value"][1]

    for value in memres.json()["data"]["result"]:
        for i in function_list:   
            if str(i) in value["metric"]["pod"]:
                mem_map[str(i)]=value["value"][1]

    for value in recres.json()["data"]["result"]:
        for i in function_list:   
            if str(i) in value["metric"]["pod"]:
                network_rec_map[str(i)]=value["value"][1]

    for value in sendres.json()["data"]["result"]:
        for i in function_list:   
            if str(i) in value["metric"]["pod"]:
                network_send_map[str(i)]=value["value"][1]   
    print(cpu_map,network_rec_map,mem_map,thr_map)
    # 每个有map有13项，分别是每个函数
    result=[]
    header = ['function','cpu','mem','cpu_util','cpu_throttle','mem_util' ,'network_transmit_bytes', 'network_receive_bytes','func_latency','latency']
    
    for i,k in enumerate(cpu_map):      # 会有空的情况，直接报错
        value=[k,int(cpu[k]),int(mem[k]),int(float(cpu_map[k])*100)/100,int(float(thr_map[k])), int(float(mem_map[k])*100)/100, 
               int(float(network_rec_map[k])), int(float(network_send_map[k])),per_func_latency[k],end2end_latency]
        with open('/home/user/code/faas-resource/function_metric/search_optimal/'+str(k)+'.txt','a') as f:
            f.write(str(value).strip('[]')+"\n")

#更新yaml文件，重新部署函数
def update_deploy(function,cpu,mem):  #list
    for i in range(len(function)):
        cmd= "/bin/bash /home/user/code/updateOpenfaasYaml.sh " + function[i] + " " + str(cpu[function[i]]) + " " + str(mem[function[i]])
        print(cmd)
        os.system(cmd)
    # except:
        # logger.error("update error")
        # return False
    return True

def get_per_func_latency(function):
    latency={}
    os.system("rm logs")    #  这个地方的路径是/home/user/code
    for i in function:
        os.system("bash /home/user/code/getLogs.sh " + i +" >> log2s" )
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

def get_end2end_latency():
    # with open("/home/user/code/faas-resource/locust_log/locustfile_search.log","r") as f:
    #     last_line=f.readlines()[-1]
    # response_time=float(last_line.split('\"')[-1][2:8])  
    with open("/home/user/code/faas-resource/locust_log/locustfile_cancel.log","r") as f:
        last_line=f.readlines()[-1]
    response_time=float(last_line.split('\":')[1][0:7])

          #读取locust文件中的response time 字符处理
    return response_time
   
def single_test(f0,f1,f2,f3,f4):
    # locustfile="/home/user/code/faas-resource/cancel.py"
    # url="http://33.33.33.99:31112/function/query-orders-for-refresh"
    # users=1
    # spawn_rate=1 
    function_list=['calculate-refund','cancel-ticket','drawback','get-order-by-id','save-order-info']
    config=[{"cpu":150,"mem":64},
            {"cpu":200,"mem":96},
            {"cpu":250,"mem":128},
            {"cpu":300,"mem":160},
            {"cpu":350,"mem":192},
            {"cpu":400,"mem":224},
            {"cpu":450,"mem":256},
            {"cpu":500,"mem":288}]
    # f0=7
    # f1=7
    # f2=0
    # f3=7
    # f4=7
    resource_cpu={function_list[0]:config[f0]["cpu"],
                  function_list[1]:config[f1]["cpu"],
                  function_list[2]:config[f2]["cpu"],
                  function_list[3]:config[f3]["cpu"],
                  function_list[4]:config[f4]["cpu"]}
    resource_mem={function_list[0]:config[f0]["mem"],
                  function_list[1]:config[f1]["mem"],
                  function_list[2]:config[f2]["mem"],
                  function_list[3]:config[f3]["mem"],
                  function_list[4]:config[f4]["mem"]}
    print(f"{f0},{f1},{f2},{f3},{f4},cpu: {resource_cpu},memory:{resource_mem}")
    launch_test(function_list,resource_cpu, resource_mem)
    
def launch_test(function_list,resource_cpu, resource_mem):
    update_deploy(function_list,resource_cpu, resource_mem)
    print('wait function all runing...')
    time.sleep(15)
    for i in range(8):   # 到5差不多稳定   改进下，感觉有些500太多了，到最后的latency比较大，有些一直200，latency非常小
         end2end_latency=single_request_search()   #last one
    print('wait to Prometheus latency collect metircs...')
    time.sleep(16)
        
    per_func_latency=get_per_func_latency(function_list)  # 从k8s log提取每个函数的时间
    print(per_func_latency) 
    print(end2end_latency)
    data2txt(function_list,resource_cpu,resource_mem,per_func_latency,end2end_latency)
    print("===========================TEST END==============================="+"\n")

def get_per_func_latency_search():
    latency={}
    call_times=[2,1,1,8,2,2,4,2,1,2,10,1]
    lines=[10,5,5,20,6,5,10,5,5,5,40,5]
    function=['get-left-ticket-of-interval','get-left-trip-tickets','get-price-by-routeid-and-traintype','get-route-by-routeid','get-route-by-tripid',
              'get-sold-tickets','get-traintype-by-traintypeid','get-traintype-by-tripid','query-already-sold-orders','query-config-entity-by-config-name',
              'query-for-station-id-by-station-name','query-for-travel']
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
    return latency,perfunc_latency  
    
def launch_test(conf,function_list,resource_config):
    update_deploy(function_list,resource_config)
    print('wait function all runing...')
    time.sleep(10)
    end2end_latency=single_request_ml()   #last one
    print('wait to Prometheus latency collect metircs...')
    time.sleep(16)
    per_func_latency=get_per_func_latency_search()  # 从k8s log提取每个函数的时间
    data2txt(function_list,resource_cpu,resource_mem,per_func_latency,end2end_latency)
    return per_func_latency
    
#查询metric，存入文件
def profile(function_list):
    cpu_price=0.000016
    mem_price=0.000005    
    config=[{"cpu":150,"mem":64},
            {"cpu":200,"mem":96},
            {"cpu":250,"mem":128},
            {"cpu":300,"mem":160},
            {"cpu":350,"mem":192},
            {"cpu":400,"mem":224},
            {"cpu":450,"mem":256},
            {"cpu":500,"mem":288}]
    instance_price=[0.00272, 0.00368, 0.00464, 0.0056, 0.00656, 0.00752, 0.00848, 0.00944]
    # max/min=3.47   如果性能可以提升这么多倍，直接可以给最大资源，这样cost也是最小，如果没有这么多倍，则有一定的制衡
    #而且没有这个数的话，如果一个一个的比较，得到的单个time*cost最优，但肯定不是整体最优
    #反而如果speedup碾压，则也一定是整体最优，因为别的业务也会更快 
    f0=7 # 7 and 4
    f1=7
    f2=7
    f3=7 # 7 and 4
    f4=7
    f5=7
    f6=7
    f7=7 # 7 and 4
    f8=7
    f9=7
    f10=7
    for f11 in [0]:
        resource_cpu={function_list[0]:config[f0]["cpu"],
                      function_list[1]:config[f1]["cpu"],
                      function_list[2]:config[f2]["cpu"],
                      function_list[3]:config[f3]["cpu"],
                      function_list[4]:config[f4]["cpu"],
                      function_list[5]:config[f5]["cpu"],
                      function_list[6]:config[f6]["cpu"],
                      function_list[7]:config[f7]["cpu"],
                      function_list[8]:config[f8]["cpu"],
                      function_list[9]:config[f9]["cpu"],
                      function_list[10]:config[f10]["cpu"],
                      function_list[11]:config[f11]["cpu"]}
                      
        resource_mem={function_list[0]:config[f0]["mem"],
                      function_list[1]:config[f1]["mem"],
                      function_list[2]:config[f2]["mem"],
                      function_list[3]:config[f3]["mem"],
                      function_list[4]:config[f4]["mem"],
                      function_list[5]:config[f5]["mem"],
                      function_list[6]:config[f6]["mem"],
                      function_list[7]:config[f7]["mem"],
                      function_list[8]:config[f8]["mem"],
                      function_list[9]:config[f9]["mem"],
                      function_list[10]:config[f10]["mem"],
                      function_list[11]:config[f11]["mem"]}
       
        print(f"{f0},{f1},{f2},{f3},{f4},{f5},{f6},{f7},{f8},{f9},{f10},{f11},cpu: {resource_cpu},memory:{resource_mem}")
        update_deploy(function_list,resource_cpu, resource_mem)

        print('wait function all runing...')
        time.sleep(10)
        for i in range(8):   # 到5差不多稳定   改进下，感觉有些500太多了，到最后的latency比较大，有些一直200，latency非常小
            end2end_latency=single_request_search()   #last one
         
        print('wait to Prometheus latency collect metircs...')
        time.sleep(16)
        
        per_func_latency=get_per_func_latency(function_list)  # 从k8s log提取每个函数的时间
        print(per_func_latency) 
        print(end2end_latency)
        # data2txt(function_list,resource_cpu,resource_mem,per_func_latency,end2end_latency)
        print("===========================TEST END==============================="+"\n")
    # for f1 in range(8):
    #     for f2 in range(8):
    #         for f3 in range(8):
    #             for f4 in range(8):
    #                 for f5 in range(8):
    #                     iter+=1
    #                     resource_cpu=[config[f1]["cpu"],config[f2]["cpu"],config[f3]["cpu"],config[f4]["cpu"],config[f5]["cpu"]]
    #                     resource_mem=[config[f1]["mem"],config[f2]["mem"],config[f3]["mem"],config[f4]["mem"],config[f5]["mem"]]
    #                     print(f"the {iter} round, config:{f1},{f2},{f3},{f4},{f5},cpu: {resource_cpu},memory:{resource_mem}")
    #                     update_deploy(function_cancel,resource_cpu, resource_cpu)
    #                     print('wait function all runing...')
    #                     time.sleep(15)
    #                     #  第一次不知道为什么必失败，时间拉长一点
    #                     run(locustfile,url,users,spawn_rate,10)
    #                     print('wait 20s to real load test...')   #稳定以下
    #                     time.sleep(20)
                        
    #                     run(locustfile,url,users,spawn_rate,10)   #再来一次 成功
    #                     # 这里要停一会，因为metric设置的是一分钟，但瞬间采集有问题
    #                     print('wait to Prometheus latency collect metircs...')
    #                     time.sleep(15)
 
    #                     per_func_latency=get_per_func_latency(function_cancel)  # 从k8s log提取每个函数的时间
    #                     print(per_func_latency)
    #                     end2end_latency=get_end2end_latency()  # 从locust log提取端到端时间
    #                     print(end2end_latency)
    #                     data2txt(function_cancel,resource_cpu,resource_mem,per_func_latency,end2end_latency)
def collect_dataset():
    # function_list=['calculate-refund','cancel-ticket','drawback','get-order-by-id','save-order-info']
    function_list=['get-left-ticket-of-interval','get-left-trip-tickets','get-price-by-routeid-and-traintype','get-route-by-routeid','get-route-by-tripid',
              'get-sold-tickets','get-traintype-by-traintypeid','get-traintype-by-tripid','query-already-sold-orders','query-config-entity-by-config-name',
              'query-for-station-id-by-station-name','query-for-travel']
    config=[{"cpu":150,"mem":64},
            {"cpu":200,"mem":96},
            {"cpu":250,"mem":128},
            {"cpu":300,"mem":160},
            {"cpu":350,"mem":192},
            {"cpu":400,"mem":224},
            {"cpu":450,"mem":256},
            {"cpu":500,"mem":288}]
    # 4 4 4 7 4
    # f0=4,7    4
    # f1=0-7  4
    # f2=0-7  4
    # f3=7    7
    # f4=4,7  4,7
     # 选择性采样    2*8*7*1*2
    iter=0 #  77074 77077
    for f0 in [7]: # calculate-refund
        for f1 in [7]:  # cancel-ticket
            for f2 in [7]:  # drawback (1,8)
                for f3 in [7]: # get-order-by-id   
                    for f4 in [7]:  # save-order-info
                        for f5 in [7]:
                            for f6 in [7]:
                                for f7 in [7]:
                                    for f8 in [7]:
                                        for f9 in [7]:
                                            for f10 in [7]:
                                                for f11 in range(8):
                                                    iter+=1
                                                    if iter > 0:                       #  少一个4,6,3,7,4      
                                                        resource_cpu={function_list[0]:config[f0]["cpu"],
                                                                    function_list[1]:config[f1]["cpu"],
                                                                    function_list[2]:config[f2]["cpu"],
                                                                    function_list[3]:config[f3]["cpu"],
                                                                    function_list[4]:config[f4]["cpu"],
                                                                    function_list[5]:config[f5]["cpu"],
                                                                    function_list[6]:config[f6]["cpu"],
                                                                    function_list[7]:config[f7]["cpu"],
                                                                    function_list[8]:config[f8]["cpu"],
                                                                    function_list[9]:config[f9]["cpu"],
                                                                    function_list[10]:config[f10]["cpu"],
                                                                    function_list[11]:config[f11]["cpu"]}
                                                        resource_mem={function_list[0]:config[f0]["mem"],
                                                                    function_list[1]:config[f1]["mem"],
                                                                    function_list[2]:config[f2]["mem"],
                                                                    function_list[3]:config[f3]["mem"],
                                                                    function_list[4]:config[f4]["mem"],
                                                                    function_list[5]:config[f5]["mem"],
                                                                    function_list[6]:config[f6]["mem"],
                                                                    function_list[7]:config[f7]["mem"],
                                                                    function_list[8]:config[f8]["mem"],
                                                                    function_list[9]:config[f9]["mem"],
                                                                    function_list[10]:config[f10]["mem"],
                                                                    function_list[11]:config[f11]["mem"]}
                        # print(f"{f0},{f1},{f2},{f3},{f4},{f5},{f6},{f7},{f8},{f9},cpu: {resource_cpu},memory:{resource_mem}")
                                                        print(f"{f0},{f1},{f2},{f3},{f4},{f5},{f6},{f7},{f8},{f9},{f10},{f11},cpu: {resource_cpu},memory:{resource_mem}")
                                                        launch_test(function_list,resource_cpu, resource_mem)


    
    
    


if __name__ == "__main__":
    # # execute only if run as a script
    # function=['calculate-refund','cancel-ticket', 'create-third-party-payment-and-pay', 'drawback','get-order-by-id', 
    #   'get-stationid-list-by-name-list' ,'modify-order' ,'pay-for-the-order','query-orders-for-refresh' ,'save-order-info']
    function_search=['get-left-ticket-of-interval','get-left-trip-tickets','get-price-by-routeid-and-traintype','get-route-by-routeid','get-route-by-tripid',
              'get-sold-tickets','get-traintype-by-traintypeid','get-traintype-by-tripid','query-already-sold-orders','query-config-entity-by-config-name',
              'query-for-station-id-by-station-name','query-for-travel']
    function_list=['calculate-refund','cancel-ticket','drawback','get-order-by-id','save-order-info']
    # cpu_conf=["250","500","750","1000","1250","1500","1750","2000"]
    # mem_conf=["64","128","192","256","320","384","448","512"]    
    #刚好按比例分配  这个比例和google cloud function，azure，aws同
    function_list2=['check-security','check-security-about-order','create-new-contacts','create-order','dipatch-seat','find-contacts-by-accountid',
                       'get-contacts-by-contactsid','get-trip-all-detai-info','get-user-by-userid','preserve-ticket']
    collect_dataset()
