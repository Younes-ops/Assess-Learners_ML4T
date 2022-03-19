""""""  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			     			  	 
All Rights Reserved  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			     			  	 
or edited.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			     			  	 
GT honor code violation.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import math  		  	   		  	  			  		 			     			  	 
import sys  		  	   		  	  			  		 			     			  	 
import matplotlib.pyplot as plt			  	   		  	  			  		 			     			  	 
import numpy as np  
import time		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import DTLearner as lrl1
import RTLearner as lrl2 
import BagLearner as bag_lr 
import random  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    if len(sys.argv) != 2:  		  	   		  	  			  		 			     			  	 
        print("Usage: python testlearner.py <filename>")  		  	   		  	  			  		 			     			  	 
        sys.exit(1)  		  	   		  	  			  		 			     			  	 
    inf = open(sys.argv[1]) 

    data = np.array(  		  	   		  	  			  		 			     			  	 
        np.array([list(map(str, s.strip().split(","))) for s in inf.readlines()])[1:,1:].astype(float)		  	   		  	  			  		 			     			  	 
    )  		  	   		  	  			  		 			     			  	 

    
    np.random.seed(903738099) 
    	   		  	  			  		 			     			  	 		  	   		  	  			  		 			     			  	 
	  	   		  	  			  		 			     			  	 
    
    def generating_test_train_data(data,shuffle=False):
        if shuffle: np.random.shuffle(data)
        # compute how much of the data is training and testing  
        train_rows = int(0.6 * data.shape[0])  		  	   		  	  			  		 			     			  	 
        test_rows = data.shape[0] - train_rows  
        # separate out training and testing data 		  	   		  	  			  		 			     			  	  		  	   		  	  			  		 			     			  	 
        train_x = data[:train_rows, 0:-1]  		  	   		  	  			  		 			     			  	 
        train_y = data[:train_rows, -1]  		  	   		  	  			  		 			     			  	 
        test_x = data[train_rows:, 0:-1]  		  	   		  	  			  		 			     			  	 
        test_y = data[train_rows:, -1] 
        return train_x, train_y, test_x, test_y


    def experiment(learner_name):
        rmse_array_train=np.empty((0,))
        rmse_array_test=np.empty((0,))
        leaf_size_array=np.arange(1,51)
        for i in leaf_size_array:
            if (learner_name=='DT') :learner = lrl1.DTLearner(leaf_size=i,verbose=False) 
            elif (learner_name =='Bag'):learner = bag_lr.BagLearner(lrl1.DTLearner,{'leaf_size':i},bags=20)
            else :
                print('invalid learner')
                break
            #train_x, train_y, test_x, test_y = generating_test_train_data(data,shuffle)
            learner.add_evidence(train_x, train_y)
            pred_y_train = learner.query(train_x)
            pred_y_test = learner.query(test_x)
            rmse_train = math.sqrt(((train_y - pred_y_train) ** 2).sum() / train_y.shape[0])
            rmse_test = math.sqrt(((test_y - pred_y_test) ** 2).sum() / test_y.shape[0])  
            rmse_array_train=np.append(rmse_array_train,rmse_train)
            rmse_array_test=np.append(rmse_array_test,rmse_test)
        
        return leaf_size_array , rmse_array_train , rmse_array_test

    train_x, train_y, test_x, test_y = generating_test_train_data(data,shuffle=False)
    print(f"{test_x.shape}")  		  	   		  	  			  		 			     			  	 
    print(f"{test_y.shape}")  
    
    print('------- Experiment 1: -------')
    all_rmse_array_train = np.empty((0,0))
    all_rmse_array_test=np.empty((0,0))


    for i in range(10):
        train_x, train_y, test_x, test_y = generating_test_train_data(data,shuffle=True)
        leaf_size_array , rmse_array_train , rmse_array_test = experiment('DT')
        all_rmse_array_train=np.append(all_rmse_array_train,rmse_array_train)
        all_rmse_array_test=np.append(all_rmse_array_test,rmse_array_test)

    all_rmse_array_train=all_rmse_array_train.reshape(10,-1)
    all_rmse_array_test=all_rmse_array_test.reshape(10,-1)
    rmse_array_train =all_rmse_array_train.mean(axis=0)
    rmse_array_test = all_rmse_array_test.mean(axis=0)
    fig=plt.figure(1)
    plt.plot(leaf_size_array,rmse_array_train,"-", color="r", label="in sample error")
    plt.plot(leaf_size_array,rmse_array_test,"-", color="g", label="out of sample error") 
    plt.grid(True) 
    plt.xlabel('leaf size')
    plt.ylabel('Rmse')
    plt.legend(loc="best")
    plt.savefig('images/{}.png'.format(str('Experiment 1 : learning curves')))
    best_leaf_size=leaf_size_array[rmse_array_test==rmse_array_test.min()]
    print('best leaf size exp1 = ',best_leaf_size)
    
    print('------- Experiment 2: -------')

    all_rmse_array_train = np.empty((0,0))
    all_rmse_array_test=np.empty((0,0))


    for i in range(10):
        train_x, train_y, test_x, test_y = generating_test_train_data(data,shuffle=True)
        leaf_size_array , rmse_array_train , rmse_array_test = experiment('Bag')
        all_rmse_array_train=np.append(all_rmse_array_train,rmse_array_train)
        all_rmse_array_test=np.append(all_rmse_array_test,rmse_array_test)

    all_rmse_array_train=all_rmse_array_train.reshape(10,-1)
    all_rmse_array_test=all_rmse_array_test.reshape(10,-1)
    rmse_array_train =all_rmse_array_train.mean(axis=0)
    rmse_array_test = all_rmse_array_test.mean(axis=0)
    fig=plt.figure(2)
    plt.plot(leaf_size_array,rmse_array_train,"-", color="r", label="in sample error")
    plt.plot(leaf_size_array,rmse_array_test,"-", color="g", label="out of sample error") 
    plt.grid(True) 
    plt.xlabel('leaf size')
    plt.ylabel('Rmse')
    plt.legend(loc="best")
    plt.savefig('images/{}.png'.format(str('Experiment 2 : learning curves')))
    best_leaf_size=leaf_size_array[rmse_array_test==rmse_array_test.min()]
    print('best leaf size exp1 = ',best_leaf_size)
    


    print('------- Experiement 3: ------')
    DT_mae_array_train=np.empty((0,))
    DT_mae_array_test=np.empty((0,))
    leaf_size_array=np.arange(1,51)
    DT_fit_time_array=np.empty((0,))
    DT_querry_time_array=np.empty((0,))

    RT_mae_array_train=np.empty((0,))
    RT_mae_array_test=np.empty((0,))
    leaf_size_array=np.arange(1,51)
    RT_fit_time_array=np.empty((0,))
    RT_querry_time_array=np.empty((0,))

    for i in leaf_size_array:
        #DT learner
        start_time_DT = time.time()
        learner_DT = lrl1.DTLearner(leaf_size=i,verbose=False) 
        learner_DT.add_evidence(train_x, train_y)
        end_time_DT = time.time()
        DT_fit_time_array=np.append(DT_fit_time_array,end_time_DT-start_time_DT)   
        pred_y_train = learner_DT.query(train_x)
        start_time_DT = time.time()
        pred_y_test = learner_DT.query(test_x)
        end_time_DT = time.time()
        DT_querry_time_array=np.append(DT_querry_time_array,end_time_DT-start_time_DT)   

        mae_train = abs(train_y - pred_y_train).sum() / train_y.shape[0]
        mae_test = abs(test_y - pred_y_test).sum() / test_y.shape[0]
        DT_mae_array_train=np.append(DT_mae_array_train,mae_train)
        DT_mae_array_test=np.append(DT_mae_array_test,mae_test)


        #RT learner
        start_time_RT = time.time()
        learner_RT = lrl2.RTLearner(leaf_size=i,verbose=False) 
        learner_RT.add_evidence(train_x, train_y)
        end_time_RT = time.time()
        RT_fit_time_array=np.append(RT_fit_time_array,end_time_DT-start_time_DT)   
        pred_y_train = learner_RT.query(train_x)

        start_time_RT = time.time()
        pred_y_test = learner_RT.query(test_x)
        end_time_RT = time.time()
        RT_querry_time_array=np.append(RT_fit_time_array,end_time_RT-start_time_RT)  
        mae_train = abs(train_y - pred_y_train).sum() / train_y.shape[0]
        mae_test = abs(test_y - pred_y_test).sum() / test_y.shape[0]
        RT_mae_array_train=np.append(RT_mae_array_train,mae_train)
        RT_mae_array_test=np.append(RT_mae_array_test,mae_test)
    fig=plt.figure(3)
    plt.plot(leaf_size_array,DT_fit_time_array,"-", color="r", label="DT fitting time")
    #plt.plot(leaf_size_array,DT_querry_time_array,"-", color="g", label="DT querry time") 
    plt.plot(leaf_size_array,RT_fit_time_array,"--", color="g", label="RT fitting time")
    #plt.plot(leaf_size_array,RT_fit_time_array,"--", color="g", label="RT  querry time") 
    plt.grid(True) 
    plt.xlabel('leaf size')
    plt.ylabel('time')
    plt.legend(loc="best")
    plt.savefig('images/{}.png'.format(str('Experiment 3 : learning curves time')))

    fig=plt.figure(4)
    plt.plot(leaf_size_array,DT_mae_array_train,"-", color="r", label="DT in sample error")
    plt.plot(leaf_size_array,DT_mae_array_test,"-", color="g", label="DT out of sample error") 
    plt.plot(leaf_size_array,RT_mae_array_train,"--", color="r", label="RT in sample error")
    plt.plot(leaf_size_array,RT_mae_array_test,"--", color="g", label="RT out of sample error") 
    plt.grid(True) 
    plt.xlabel('leaf size')
    plt.ylabel('Mae')
    plt.legend(loc="best")
    plt.savefig('images/{}.png'.format(str('Experiment 3 : learning curves MAE')))

	  	 
    # create a learner and train it  		  	   		  	  			  		 			     			  	 
    learnerDT = lrl1.DTLearner(leaf_size=10,verbose=True)  # create a DT learner  
    learnerRT = lrl2.RTLearner(leaf_size=10,verbose=True)  # create a RT learner  			  	   		  	  			  		 			     			  	 
    learnerDT.add_evidence(train_x, train_y)  # train it  	
    learnerRT.add_evidence(train_x, train_y)  # train it  	  	   		  	  			  		 			     			  	 
    # print(learner.author())  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # evaluate in sample  		  	   		  	  			  		 			     			  	 
    pred_y1 = learnerDT.query(train_x)  # get the predictions  	
    pred_y2 = learnerRT.query(train_x)	  	   		  	  			  		 			     			  	 
    rmse1 = math.sqrt(((train_y - pred_y1) ** 2).sum() / train_y.shape[0])  
    rmse2 = math.sqrt(((train_y - pred_y2) ** 2).sum() / train_y.shape[0])		  	   		  	  			  		 			     			  	 
    print()  		  	   		  	  			  		 			     			  	 
    print("In sample results")  		  	   		  	  			  		 			     			  	 
    print(f"RMSE1: {rmse1}")  	
    print(f"RMSE2: {rmse2}")  	 	   		  	  			  		 			     			  	 
    c1 = np.corrcoef(pred_y1, y=train_y)  	
    c2 = np.corrcoef(pred_y2, y=train_y)  	  	   		  	  			  		 			     			  	 
    print(f"corr1: {c1[0,1]}")  
    print(f"corr2: {c2[0,1]}") 		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # evaluate out of sample  		  	   		  	  			  		 			     			  	 
    pred_y1 = learnerDT.query(test_x)  # get the predictions  	
    pred_y2 = learnerRT.query(test_x)	  	   		  	  			  		 			     			  	 
    rmse1 = math.sqrt(((test_y - pred_y1) ** 2).sum() / test_y.shape[0])  
    rmse2 = math.sqrt(((test_y - pred_y2) ** 2).sum() / test_y.shape[0])			  	   		  	  			  		 			     			  	 
    print()  		  	   		  	  			  		 			     			  	 
    print("Out of sample results")  		  	   		  	  			  		 			     			  	 
    print(f"RMSE1: {rmse1}")  	
    print(f"RMSE2: {rmse2}")   	  	   		  	  			  		 			     			  	 
    c1 = np.corrcoef(pred_y1, y=test_y)  	
    c2 = np.corrcoef(pred_y2, y=test_y)  	  	   		  	  			  		 			     			  	 
    print(f"corr1: {c1[0,1]}")  
    print(f"corr2: {c2[0,1]}") 	 		  	   		  	  			  		 			     			  	 
