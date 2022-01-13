import os
import json

"""
    This functions sets the necessary values in the TF_CONFIG environment variable. It contains information on the 
    cluster architectures, i.e, which workers are allocated for the job, and the worker for the current task. This has 
    been specifically been developed for using the SLURM task manager, and kind of hardcoded for the ouput given using 
    the Berzelius supercomputer at NSC. It may be the case that this funciton does not work on other clusters without
    some changes. 

    Here, the outputted string s from the call  to os.environ["SLURM_JOB_NODELIST"], contains all the allocated 
    workers for the job. 

        Examples: 
            s = "Node021"
            s = "Node[036-039]"
            s = "Node[009-012,047]"

        We need to translate this format into a separated list of strings representing the nodes to be able to 
        describe the cluster in a way that tf.distribute.MultiWorkerMirroredStrategy can interpret the cluster. That 
        is we want: 
            s = "Node021"             -> ["Node021"]
            s =" Node[036-039]"       -> ["Node036", "Node037", "Node038", "Node039"]
            s = "Node[009-012,047]"   -> ["Node009","Node010","Node011","Node012", "Node047"]

        This is what is done below.
        An example for the case s = Node[009-012,047] is followed within the comments.
"""


def set_tf_config():
    s = os.environ["SLURM_JOB_NODELIST"]  #example:  s = "Node[009-012,047]"

    if s.find("[") == -1:  # The case with only one node, has no brackets. Finds the node number.
        s4 = [str(s[s.find("e") + 1:])]

    else:
        s2 = s[s.find("[") + 1:  s.find("]")]  # s2 = "009-012,047"

        s3 = s2.split(",")  # s3 = ["009-012","047"]
        s4 = []
        for i in s3:
            j = i.find("-")
            if j != -1:
                s5 = i.split("-")
                a = int(s5[0])
                while a < int(s5[1]) + 1:
                    s4.append(str(a))
                    a += 1
            else:
                s4.append(i)  # s3 = ["009","010","011","012","047"]
        #print(s4)

    # The node numbering is done using three digits, padded with zeros if necessary.
    number_of_zeros = [3 - len(i) for i in s4]
    clust = ["node" + "0" * i[0] + i[1] for i in zip(number_of_zeros, s4)]  # Clust =  ["Node009","Node010","Node011","Node012", "Node047"]
    num_workers = len(clust)
    port ="8888" # Choose a port number to use

    # In order to communicate, the nodes need to be supplied with port numbers (This is something that I do not 
    # really understand). 
    clust_with_ports = [s + ":"+port for s in
                        clust]  # = ["Node009:8888","Node010:8888","Node011:8888","Node012:8888", "Node047:8888"]

    # This outputs the node used for the specific task, where most likely we want to have 1 node corresponding to 1 
    # task. Use this to check if it is the first worker. The first worker is usually appointed some extra tasks in 
    # addition to training. This can for example be printing stuff etc, just using print() will print using all 
    # tasks, an we will just get extra print statements. 
    t = os.environ["SLURMD_NODENAME"]
    # Find at which index the current Node is, if it is the first node in the job, this is appointed chief status. 
    # This is also used as an output from this function 
    ind = clust.index(t)
    
    if ind == 0:
        role = "worker"
        chief = t
    else:
        role = "worker"
        ind = ind
        chief = 0

    """
    If we explicitly appoint a worker as the chief, it seems to not take part in training. This can be done in this manner:

        cfg = {'cluster': {'chief': [clust_with_ports[0]], 'worker' : clust_with_ports[1:] },
                #'cluster': {'worker': clust},
                'task': {'type': role,'index': ind,},
                'rpc_layer': 'grpc',}

    Here I say that the first node is the chief, and the rest are workers. This is most likely nto what I want.
    I want worker with index 0 to be responsible for printing etc.
    """

    cfg = {
        'cluster': {'worker': clust_with_ports},
        'task': {'type': role, 'index': ind, },
        'rpc_layer': 'grpc' }

    # These addresses with the  "grpc://" prefixes are needed when doing profiling (I think)- profiling multiple 
    # workers seems hard. 
    addresses = [",".join(["grpc://" + c + ":"+port for c in clust])]
    print(addresses)
    # Now we have the full tf config variable, write it to the os.environ to set it as a environment variable.
    os.environ['TF_CONFIG'] = json.dumps(cfg)

    return addresses, chief, num_workers
