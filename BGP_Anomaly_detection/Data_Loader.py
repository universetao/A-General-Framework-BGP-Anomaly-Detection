import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Data_Loader(object):
    data_cols=[]
    INPUT_SIZE =83
    def __init__(self,Event_name,Event_num,TIME_STEP=1,WINDOW_SIZE=30,Hijack_Window=39, Legitimate_Window=1440,AddNoramalWin=90,Anomaly_Timestep=15,Legitimate_TIME_STEP=2):

        self.Event_name = Event_name
        self.Event_num = Event_num
        self.TIME_STEP = TIME_STEP
        self.WINDOW_SIZE = WINDOW_SIZE
        self.Hijack_Window=Hijack_Window
        self.Legitimate_Window=Legitimate_Window
        self.AddNoramalWin=AddNoramalWin
        self.Anomaly_Timestep=Anomaly_Timestep
        self.Legitimate_TIME_STEP=Legitimate_TIME_STEP

        self.__loadDataCol(read_from_file=True)
    def loadDataSet(self,read_from_file=False,include_MANRS_data=True,baseline=False):
    # 数据的目录\

        if read_from_file:
            train_x = np.load(
                '../time_step_data/TimeStep_Orign_train_' + self.Event_name + '_' + str(self.TIME_STEP) + '_WINDOWSIZE_' + str(
                    self.WINDOW_SIZE) + '_X'  +str(include_MANRS_data)+ '.npy')
            train_y = np.load(
                '../time_step_data/TimeStep_Orign_train_' + self.Event_name + '_' + str(self.TIME_STEP) + '_WINDOWSIZE_' + str(
                    self.WINDOW_SIZE) + '_Y' +str(include_MANRS_data)+  '.npy')

            test_x = np.load(
                '../time_step_data/TimeStep_Orign_test_' + self.Event_name + '_' + str(self.TIME_STEP) + '_WINDOWSIZE_' + str(
                    self.WINDOW_SIZE) + '_X'  +str(include_MANRS_data)+  '.npy')
            test_y = np.load(
                '../time_step_data/TimeStep_Orign_test_' + self.Event_name + '_' + str(self.TIME_STEP) + '_WINDOWSIZE_' + str(
                    self.WINDOW_SIZE) + '_Y'  +str(include_MANRS_data)+  '.npy')
            self.INPUT_SIZE = train_x.shape[2]

        else:
            if self.Event_num!=0:
                x,y0,hijack_event_len,legitimate_event_len=self.__load_pd_dataset(include_MANRS_data=include_MANRS_data,baseline=baseline)

            self.INPUT_SIZE=x.shape[1]
            print('self.INPUT_SIZE',self.INPUT_SIZE)
            
            train_x, test_x, train_y, test_y,eval_x,eval_y = self.__train_test_split_events(x, y0, hijack_event_len,
                                                                    legitimate_event_len, test_size=0.1,
                                                                   random_state=42)

            np.save('../time_step_data/TimeStep_Orign_train_' + self.Event_name + '_' + str(self.TIME_STEP) + '_WINDOWSIZE_' + str(
                self.WINDOW_SIZE) + '_X' +str(include_MANRS_data), train_x)
            np.save('../time_step_data/TimeStep_Orign_train_' + self.Event_name + '_' + str(self.TIME_STEP) + '_WINDOWSIZE_' + str(
                self.WINDOW_SIZE) + '_Y'+str(include_MANRS_data) , np.array(train_y))

            np.save('../time_step_data/TimeStep_Orign_test_' + self.Event_name + '_' + str(self.TIME_STEP) + '_WINDOWSIZE_' + str(
                self.WINDOW_SIZE) + '_X'+str(include_MANRS_data) ,
                    test_x)
            np.save('../time_step_data/TimeStep_Orign_test_' + self.Event_name + '_' + str(self.TIME_STEP) + '_WINDOWSIZE_' + str(
                self.WINDOW_SIZE) + '_Y'+str(include_MANRS_data) ,
                    np.array(test_y))

            np.save(
                '../time_step_data/TimeStep_Orign_test_' + self.Event_name + '_' + str(self.TIME_STEP) + '_WINDOWSIZE_' + str(
                    self.WINDOW_SIZE) + '_X' + str(include_MANRS_data),
                eval_x)
            np.save(
                '../time_step_data/TimeStep_Orign_test_' + self.Event_name + '_' + str(self.TIME_STEP) + '_WINDOWSIZE_' + str(
                    self.WINDOW_SIZE) + '_Y' + str(include_MANRS_data),
                np.array(eval_y))
        # train_x, train_y = random_undersampler(train_x, train_y, p=0.05)
        train_x = torch.tensor(train_x, dtype=torch.float32)
        train_y = torch.tensor(train_y)

        test_x = torch.tensor(test_x, dtype=torch.float32)
        test_y = torch.tensor(np.array(test_y))

        eval_x = torch.tensor(eval_x, dtype=torch.float32)
        eval_y = torch.tensor(np.array(eval_y))
        return train_x, train_y, test_x, test_y,eval_x,eval_y

    def loadDataSet_route_leak(self):
        pass
    def __train_test_split_events(self,x, y, hijack_event_len,legitimate_event_len, test_size=0.1, random_state=42):
        event_x = x.iloc[0:hijack_event_len].values
        event_y = y.iloc[0:hijack_event_len].values
        legitimate_x = x.iloc[hijack_event_len:legitimate_event_len].values
        legitimate_y = y.iloc[hijack_event_len:legitimate_event_len].values

        event_num = (int)(len(event_x) / self.Hijack_Window)
        Y = []

        legitimate_num = (int)(len(legitimate_x) / self.Legitimate_Window)
        legitimate_Y = []
       
        for event in tqdm(range(0, event_num, 1)):
            bottom = event * self.Hijack_Window

            if bottom == 0:
                tempx = event_x[bottom:bottom + self.Hijack_Window, :]
                tempx = tempx[np.newaxis, :]
                X = tempx
                Y.append(event_y[bottom + self.Anomaly_Timestep])
            else:
                tempx = event_x[bottom:bottom + self.Hijack_Window, :]
                tempx = tempx[np.newaxis, :]
                X = np.concatenate((X, tempx), axis=0)
                Y.append(event_y[bottom + self.Anomaly_Timestep])
        for event in tqdm(range(0, legitimate_num, 1)):
            bottom = event * self.Legitimate_Window

            if bottom == 0:
                tempx = legitimate_x[bottom:bottom + self.Legitimate_Window, :]
                tempx = tempx[np.newaxis, :]
                legitimate_X = tempx
                legitimate_Y.append(legitimate_y[bottom + self.Anomaly_Timestep])
            else:
                tempx = legitimate_x[bottom:bottom + self.Legitimate_Window, :]
                tempx = tempx[np.newaxis, :]
                legitimate_X = np.concatenate((legitimate_X, tempx), axis=0)
                legitimate_Y.append(legitimate_y[bottom + self.Anomaly_Timestep])
        

        train_event_X, test_event_x, train_event_Y, test_event_y = train_test_split(X, Y, test_size=test_size,
                                                                                    random_state=random_state)
        train_le_x, test_legitimate_x, train_le_y, test_legitimate_y = train_test_split(legitimate_X,
                                                                                                        legitimate_Y,
                                                                                                        test_size=test_size,
                                                                                                        random_state=random_state)
        train_event_x, eval_event_x, train_event_y, eval_event_y = train_test_split(train_event_X, train_event_Y, test_size=test_size,random_state=random_state)

        train_legitimate_x, eval_legitimate_x, train_legitimate_y, eval_legitimate_y = train_test_split(train_le_x,train_le_y,test_size=0.2,random_state=random_state)


     
        train_event_x = train_event_x.reshape([-1, self.INPUT_SIZE])
        test_event_x = test_event_x.reshape([-1, self.INPUT_SIZE])
        eval_event_x=eval_event_x.reshape([-1, self.INPUT_SIZE])

        train_legitimate_x = train_legitimate_x.reshape([-1, self.INPUT_SIZE])
        test_legitimate_x = test_legitimate_x.reshape([-1, self.INPUT_SIZE])
        eval_legitimate_x=eval_legitimate_x.reshape([-1, self.INPUT_SIZE])
        

        train_event_len = train_event_x.shape[0]
        test_event_len = test_event_x.shape[0]
        eval_event_len=eval_event_x.shape[0]

        print('train_event_len', train_event_len)

        train_event_x = np.concatenate((train_event_x, train_legitimate_x), axis=0)
        test_event_x = np.concatenate((test_event_x, test_legitimate_x), axis=0)
        eval_event_x=np.concatenate((eval_event_x,eval_legitimate_x),axis=0)

        train_legitimate_len = train_event_x.shape[0]
        test_legitimate_len= test_event_x.shape[0]

    

        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import Normalizer
        from sklearn.preprocessing import StandardScaler

        scaler = MinMaxScaler()
        x = scaler.fit_transform(train_event_x)
        test_x = scaler.transform(test_event_x)
        eval_x=scaler.transform(eval_event_x)
        import pickle
        pickle.dump(scaler, open('../params/' + self.Event_name + '_scaler.pkl', 'wb'))
        print(len(X))
        print(X.shape)
        # train
        hijack_timestep_X, hijack_timestep_y = self.to_timestep_dataset(x[0:train_event_len],
                                                                    train_event_y, self.Hijack_Window,self.TIME_STEP)
        legitimate_timestep_X, legitimate_timestep_y = self.to_timestep_dataset(x[train_event_len:train_legitimate_len],
                                                                            train_legitimate_y,
                                                                            self.Legitimate_Window,self.Legitimate_TIME_STEP)

        
        timestep_X = np.concatenate((hijack_timestep_X, legitimate_timestep_X), axis=0)
        hijack_timestep_y.extend(legitimate_timestep_y)
        # hijack_timestep_y.extend(add_timestep_y)
        # test
        test_hijack_timestep_X, test_hijack_timestep_y = self.to_timestep_dataset(test_x[0:test_event_len],
                                                                              test_event_y,
                                                                              self.Hijack_Window,self.TIME_STEP)
        test_legitimate_timestep_X, test_legitimate_timestep_y = self.to_timestep_dataset(test_x[test_event_len:test_legitimate_len],
                                                                                      test_legitimate_y,
                                                                                      self.Legitimate_Window,self.Legitimate_TIME_STEP)
    
        test_timestep_X = np.concatenate((test_hijack_timestep_X, test_legitimate_timestep_X),
                                         axis=0)
        test_hijack_timestep_y.extend(test_legitimate_timestep_y)
  
        print(test_timestep_X.shape)

        eval_hijack_timestep_X, eval_hijack_timestep_y = self.to_timestep_dataset(eval_x[0:eval_event_len],
                                                                                  eval_event_y,
                                                                                  self.Hijack_Window, self.TIME_STEP)
        eval_legitimate_timestep_X, eval_legitimate_timestep_y = self.to_timestep_dataset(
            eval_x[test_event_len:test_legitimate_len],
            eval_legitimate_y,
            self.Legitimate_Window, self.Legitimate_TIME_STEP)
        eval_timestep_X = np.concatenate((eval_hijack_timestep_X, eval_legitimate_timestep_X),
                                         axis=0)
        eval_hijack_timestep_y.extend(eval_legitimate_timestep_y)


        return timestep_X, test_timestep_X, hijack_timestep_y, test_hijack_timestep_y, eval_timestep_X, eval_hijack_timestep_y
 
    def __load_pd_dataset(self,include_MANRS_data=True,baseline=False):
        datasets_path = '../datasets/datasets/'
        # load files path
        datasets_files = self.loadDataSet_path(datasets_path)
        # spilt files to test and train

        data_all = pd.DataFrame()

        half_window = (int)((self.Hijack_Window + 1) / 2)
        self.Anomaly_Timestep = half_window
        print(half_window)
        # train_data
        count=0
        for data_file in datasets_files:

            temp = pd.read_csv(datasets_path + data_file, index_col=0)
            if (temp.iloc[120]['label_0'] != self.Event_num):
                continue
            temp.iloc[120:120 + half_window]['label_0'] = 1
            data_all = data_all.append(temp.iloc[120 - half_window + 1:120 + half_window])
            count += 1
     

        hijack_event_len = data_all.shape[0]
        if include_MANRS_data:
            datasets_path2 = '../datasets/legitimate/'
            datasets_files2 = self.loadDataSet_path(datasets_path2)
            for data_file in datasets_files2:
                try:
                    temp = pd.read_csv(datasets_path2 + data_file, index_col=0)
                    data_all = data_all.append(temp)
                except:
                    print(datasets_path2 + data_file)
        legitimate_event_len = data_all.shape[0]
        
        # data prepocessing
        data_all = data_all.drop(
            columns=['time', 'new_sub_prefix', 'MOAS_AS', 'Victim_AS', 'MOAS', 'withdraw_unique_prefix'],
            axis=1)
        data_all.fillna(0, inplace=True)

        self.__add_count(data_all, 14, 21, 11, 11)
        data_all = pd.DataFrame(data_all, columns=self.data_cols)
        data_all.fillna(0, inplace=True)
        print(data_all)

        # change test features to train features

        x, y0, y1, y2, y3 = data_all.drop(columns=['label_0', 'label_1', 'label_2', 'label_3'], axis=1), data_all[
            'label_0'], data_all['label_1'], data_all['label_2'], data_all['label_3']
        if baseline:
            x=x[['MOAS_Ann_num','own_Ann_num','withdraw_num','duplicate_ann','withdraw_unique_prefix_num','Diff_Ann','peer_num']]

        return x, y0, hijack_event_len, legitimate_event_len

    @staticmethod
    def load(self,datasets_path, window):
        # load files path
        datasets_files = self.loadDataSet_path(datasets_path)
        # spilt files to test and train
        data_all = pd.DataFrame()
        # train_data

        for data_file in datasets_files:
            try:
                temp = pd.read_csv(datasets_path + data_file, index_col=0)
                temp['label_0'].iloc[719:window] = 1
                data_all = data_all.append(temp.iloc[0:window])
                print(datasets_path + data_file + '正常')
            except:
                print(traceback.print_exc())
                print(datasets_path + data_file)
        # data prepocessing
        data_all = data_all.drop(
            columns=['time', 'new_sub_prefix', 'MOAS_AS', 'Victim_AS', 'MOAS', 'withdraw_unique_prefix'],
            axis=1)

        data_all.fillna(0, inplace=True)

        self.__add_count(data_all, 14, 21, 11, 11)
        data_all = pd.DataFrame(data_all, columns=self.data_cols)
        data_all.fillna(0, inplace=True)
        # print(data_all)

        # change test features to train features

        x, y0, y1, y2, y3 = data_all.drop(columns=['label_0', 'label_1', 'label_2', 'label_3'], axis=1), \
                            data_all['label_0'], data_all['label_1'], data_all['label_2'], data_all['label_3']
        return x, y0

    def __loadDataCol(self,read_from_file=False):
        if read_from_file:
            data_all = pd.read_csv('../result_doc/data_all.csv', index_col=0)
            self.data_cols = data_all.columns
        else:
            datasets_path = '../datasets/datasets/'
            datasets_path2 = '../datasets/legitimate/'
            # load files path
            datasets_files = self.loadDataSet_path(datasets_path)
            datasets_files2 = self.loadDataSet_path(datasets_path2)
            # spilt files to test and train

            data_all = pd.DataFrame()

            half_window = (int)((self.Hijack_Window + 1) / 2)
            print(half_window)
            # train_data
            for data_file in datasets_files:
                try:
                    temp = pd.read_csv(datasets_path + data_file, index_col=0)
                    data_all = data_all.append(temp.iloc[120 - half_window + 1:120 + half_window])
                except:
                    print(datasets_path + data_file)
            for data_file in datasets_files2:
                try:
                    temp = pd.read_csv(datasets_path2 + data_file, index_col=0)
                    data_all = data_all.append(temp)
                except:
                    print(datasets_path2 + data_file)
            # data prepocessing
            data_all = data_all.drop(
                columns=['time', 'new_sub_prefix', 'MOAS_AS', 'Victim_AS', 'MOAS', 'withdraw_unique_prefix'],
                axis=1)

            data_all.fillna(0, inplace=True)

            self.__add_count(data_all, 14, 21, 11, 11)

            data_all.fillna(0, inplace=True)
            # drop useless features
            cols = data_all.columns
            all_len = data_all.shape[0]
            for i in cols:
                is_0 = data_all[data_all[i] == 0].shape[0]
                if is_0 == all_len:
                    print(i)
                    data_all.drop(columns=i, axis=1, inplace=True)
            # change test features to train features
            data_all.to_csv('../result_doc/data_all.csv')
            self.data_cols = data_all.columns

    def to_timestep_dataset(self,x, y, event_len,time_step):  # x,y are array
        event_num = (int)(len(x) / event_len)
        length = event_len - self.WINDOW_SIZE+1
        print(length)
        print(x.shape)
        if event_num != 0:
            bottom = 0
            tempx = x[0:self.WINDOW_SIZE, :]
            tempx = tempx[np.newaxis, :]
            temp_X = tempx
            temp_Y = [y[0]]
            for step in range(1, length, time_step):
                now = bottom + step
                tempx = x[now:now + self.WINDOW_SIZE, :]
                tempx = tempx[np.newaxis, :]
                temp_X = np.concatenate((temp_X, tempx), axis=0)
                temp_Y.append(y[0])
            X = temp_X
            Y = temp_Y
        for event in tqdm(range(1, event_num, 1)):
            bottom = event * event_len
            tempx = x[bottom:bottom + self.WINDOW_SIZE, :]
            tempx = tempx[np.newaxis, :]
            temp_X = tempx
            temp_Y = [y[event]]
            for step in range(1, length, time_step):
                now = bottom + step
                tempx = x[now:now + self.WINDOW_SIZE, :]
                tempx = tempx[np.newaxis, :]
                temp_X = np.concatenate((temp_X, tempx), axis=0)
                temp_Y.append(y[event])
            X = np.concatenate((X, temp_X), axis=0)
            Y.extend(temp_Y)

        print(X.shape)
        return X, Y  # return array

    def to_timestep(self,x, y, event_len):  # x,y are array
        event_num = (int)(len(x) / event_len)
        length = event_len - self.WINDOW_SIZE + 1
        print(length)
        # X=np.array()
        Y = []  # type: List[Any]
        print(x.shape)
        for event in tqdm(range(0, event_num, 1)):
            bottom = event * event_len
            for step in range(0, length, self.TIME_STEP):
                now = bottom + step
                if now == 0:
                    tempx = x[now:now + self.WINDOW_SIZE, :]
                    tempx = tempx[np.newaxis, :]
                    X = tempx
                    Y.append(y[now + self.WINDOW_SIZE - 1])
                else:
                    tempx = x[now:now + self.WINDOW_SIZE, :]
                    tempx = tempx[np.newaxis, :]
                    X = np.concatenate((X, tempx), axis=0)
                    Y.append(y[now + self.WINDOW_SIZE - 1])
        # print(len(X))
        print(X.shape)

        # print(x.head)
        return X, Y  # return array

    def random_undersampler(self,x, y, p):
        import random
        X = x[0]
        X = X[np.newaxis, :]
        Y = [y[0]]
        for i in range(1, len(x)):
            if y[i] == 0:
                if random.random() < p:
                    temp_x = x[i]
                    temp_x = temp_x[np.newaxis, :]
                    X = np.concatenate((X, temp_x), axis=0)
                    Y.append(0)
            else:
                temp_x = x[i]
                temp_x = temp_x[np.newaxis, :]
                X = np.concatenate((X, temp_x), axis=0)
                Y.append(1)
            # print(X.shape)
        print('after_sampler_X: ', X.shape)
        # print('after_sampler_Y: ',Y.shape)
        return X, np.array(Y)

    def loadroute_leak(self):
        datasets_path = '../test/'
        # load files path
        datasets_files = self.loadDataSet_path(datasets_path)
        # spilt files to test and train
        data_all = pd.DataFrame()
        # train_data

        for data_file in datasets_files:
            try:
                temp = pd.read_csv(datasets_path + data_file, index_col=0)
                data_all = data_all.append(temp)
                print(datasets_path + data_file)
            except:
                print(traceback.print_exc())
                print(datasets_path + data_file)
        # data prepocessing
        data_all = data_all.drop(
            columns=['time', 'new_sub_prefix', 'MOAS_AS', 'Victim_AS', 'MOAS', 'withdraw_unique_prefix'],
            axis=1)

        data_all.fillna(0, inplace=True)

        self.__add_count(data_all, 14, 21, 11, 11)
        data_all = pd.DataFrame(data_all, columns=self.data_cols)
        data_all.fillna(0, inplace=True)
        print(data_all)

        # change test features to train features

        x, y0, y1, y2, y3 = data_all.drop(columns=['label_0', 'label_1', 'label_2', 'label_3'], axis=1), \
                            data_all['label_0'], data_all['label_1'], data_all['label_2'], data_all['label_3']
        return x, y0

    def loadroute_leak_train_test(self,scheme='A'):
        datasets_path = '../test/'
        # load files path
        datasets_files = self.loadDataSet_path(datasets_path)
        # spilt files to test and train
        data_all = pd.DataFrame()
        test_all= pd.DataFrame()
        if scheme=='A':
            train_set=[datasets_files[0],datasets_files[1],datasets_files[2]]
            test_set=[datasets_files[3]]
        elif scheme=='B':
            train_set = [datasets_files[0], datasets_files[2], datasets_files[3]]
            test_set = [datasets_files[1]]
            drop_MOAS_prefix_num=True # reduce the high confidence caused by MOSA_prefix_num, for re-training

        elif scheme=='C':
            train_set = [datasets_files[0], datasets_files[1], datasets_files[3]]
            test_set = [datasets_files[2]]
        elif scheme=='D':
            train_set = [datasets_files[1], datasets_files[2], datasets_files[3]]
            test_set = [datasets_files[0]]
        # train_data

        for data_file in train_set:# you can change the file index in training set to build the scheme
            try:
                temp = pd.read_csv(datasets_path + data_file, index_col=0)
                data_all = data_all.append(temp)
                print(datasets_path + data_file)
            except:
                print(traceback.print_exc())
                print(datasets_path + data_file)
        for data_file in test_set:
            try:
                temp = pd.read_csv(datasets_path + data_file, index_col=0)
                test_all = test_all.append(temp)
                print(datasets_path + data_file)
            except:
                print(traceback.print_exc())
                print(datasets_path + data_file)

        # data prepocessing
        data_all = data_all.drop(
            columns=['time', 'new_sub_prefix', 'MOAS_AS', 'Victim_AS', 'MOAS', 'withdraw_unique_prefix'],
            axis=1)



        data_all.fillna(0, inplace=True)

        self.__add_count(data_all, 14, 21, 11, 11)
        data_all = pd.DataFrame(data_all, columns=self.data_cols)

        if drop_MOAS_prefix_num:
            data_all = data_all.drop(
                columns=['MOAS_prefix_num'],
                axis=1)
        data_all.fillna(0, inplace=True)
        print(data_all)

        test_all = test_all.drop(
            columns=['time', 'new_sub_prefix', 'MOAS_AS', 'Victim_AS', 'MOAS', 'withdraw_unique_prefix'],
            axis=1)

        test_all.fillna(0, inplace=True)

        self.__add_count(test_all, 14, 21, 11, 11)
        test_all = pd.DataFrame(test_all, columns=self.data_cols)

        test_all.fillna(0, inplace=True)
        # change test features to train features

        x, y0, y1, y2, y3 = data_all.drop(columns=['label_0', 'label_1', 'label_2', 'label_3'], axis=1), \
                            data_all['label_0'], data_all['label_1'], data_all['label_2'], data_all['label_3']

        testx, testy0, testy1, testy2, testy3 = test_all.drop(columns=['label_0', 'label_1', 'label_2', 'label_3'], axis=1), \
                            test_all['label_0'], test_all['label_1'], test_all['label_2'], test_all['label_3']

        return x, y0,testx,testy0

    def loadDataSet_path(self,datasets_path):  # 0 means all
        import os

        datasets = os.listdir(datasets_path)
        return datasets

    def __add_count(self,data_all, edit_threshold, pl_threshold, longer_threshold, shorter_threshold):
        import re

        ls = data_all.columns

        diff_group = []
        len_group = []
        ann_longer_group = []
        ann_shorter_group = []
        drop_edit = set()
        drop_pl = set()
        drop_longer = set()
        drop_shorter = set()
        ls = data_all.columns
        for col in ls:
            if re.search("diff_\d+", col) != None:
                # print(re.search("diff_\d+", col).string)
                diff_group.append(re.search("diff_\d+", col).string)
            elif re.search("len_path\d+", col) != None:
                # print(re.search("len_path\d+", col).string)
                len_group.append(re.search("len_path\d+", col).string)
            elif re.search("ann_longer_\d+", col) != None:
                # print(re.search("ann_longer_\d+", col).string)
                ann_longer_group.append(re.search("ann_longer_\d+", col).string)
            elif re.search("ann_shorter_\d+", col) != None:
                ann_shorter_group.append(re.search("ann_shorter_\d+", col).string)
        data_all['sum_diff'] = 0
        data_all['sum_diff_num'] = 0
        data_all['PL_sum'] = 0
        data_all['sum_len_num'] = 0
        data_all['sum_ann_longer'] = 0
        data_all['sum_ann_longer_num'] = 0
        data_all['sum_ann_shorter'] = 0
        data_all['sum_ann_shorter_num'] = 0
        data_all['avg_diff'] = 0
        data_all['avg_longer'] = 0
        data_all['avg_shorter'] = 0
        data_all['avg_len'] = 0
        edit = 'edit_bigger_' + str(edit_threshold)
        ppl = 'PL_bigger_' + str(pl_threshold)
        longer = 'longer_bigger_' + str(longer_threshold)
        shorter = 'shorter_bigger_' + str(shorter_threshold)
        data_all[edit] = 0
        data_all[ppl] = 0
        data_all[longer] = 0
        data_all[shorter] = 0
        for diff in diff_group:
            num = int(diff.split('_')[1])
            data_all['sum_diff'] += num * data_all[diff]
            data_all['sum_diff_num'] += data_all[diff]
            if num >= edit_threshold:
                data_all[edit] += data_all[diff]
                drop_edit.add(diff)
        for PL in len_group:
            num = int(PL.split('h')[1])
            data_all['PL_sum'] += num * data_all[PL]
            data_all['sum_len_num'] += data_all[PL]
            if num >= pl_threshold:
                drop_pl.add(PL)
                data_all[ppl] += data_all[PL]
        for al in ann_longer_group:
            num = int(al.split('_')[2])
            data_all['sum_ann_longer'] += num * data_all[al]
            data_all['sum_ann_longer_num'] += data_all[al]
            if num >= longer_threshold:
                data_all[longer] += data_all[al]
                drop_longer.add(al)
        for ann_shorter in ann_shorter_group:
            num = int(ann_shorter.split('_')[2])
            data_all['sum_ann_shorter'] += num * data_all[ann_shorter]
            data_all['sum_ann_shorter_num'] += data_all[ann_shorter]

            if num >= shorter_threshold:
                data_all[shorter] += data_all[ann_shorter]
                drop_shorter.add(ann_shorter)

            data_all['avg_diff'] = data_all['sum_diff'] / data_all['sum_diff_num']

            data_all['avg_longer'] = data_all['sum_ann_longer'] / data_all['sum_ann_longer_num']

            data_all['avg_shorter'] = data_all['sum_ann_shorter'] / data_all['sum_ann_shorter_num']

            data_all['avg_len'] = data_all['PL_sum'] / data_all['sum_len_num']
        data_all.drop(columns=list(drop_edit), inplace=True)
        data_all.drop(columns=list(drop_pl), inplace=True)
        data_all.drop(columns=list(drop_longer), inplace=True)
        data_all.drop(columns=list(drop_shorter), inplace=True)
    def __divid_into_group(self,data_all,):
        import re

        ls = data_all.columns

        diff_group = []
        len_group = []
        ann_longer_group = []
        ann_shorter_group = []
        ls = data_all.columns
        for col in ls:
            if re.search("diff_\d+", col) != None:
                # print(re.search("diff_\d+", col).string)
                diff_group.append(re.search("diff_\d+", col).string)
            elif re.search("len_path\d+", col) != None:
                # print(re.search("len_path\d+", col).string)
                len_group.append(re.search("len_path\d+", col).string)
            elif re.search("ann_longer_\d+", col) != None:
                # print(re.search("ann_longer_\d+", col).string)
                ann_longer_group.append(re.search("ann_longer_\d+", col).string)
            elif re.search("ann_shorter_\d+", col) != None:
                ann_shorter_group.append(re.search("ann_shorter_\d+", col).string)
        return diff_group,len_group,ann_longer_group,ann_shorter_group

  
    def bigger(self,i, j,flag,num):
        return (int(i.split(flag)[num]) - int(j.split(flag)[num])) > 0

    def sort_group(self,group,flag,num):
        for i in range(len(group)):
            for j in range(0, len(group) - i - 1):
                if self.bigger(group[j], group[j + 1],flag,num):
                    temp = group[j]
                    group[j] = group[j + 1]
                    group[j + 1] = temp
        return group
# for i in ['prefix_hijack', 'route_leak', 'breakout', 'edge', 'defcon']:
#     d=Data_Loader(Event_name=i,Event_num=1)
#     d.plot_tsne()
# train_x, train_y, test_x, test_y = d.loadDataSet(read_from_file=False,include_MANRS_data=True)
#d=Data_Loader(Event_name='prefix_hijack',Event_num=1)
#d.measure_prefix()
#d.measure_Volume()
