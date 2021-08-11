
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from LSTM_model import LSTM
from Self_Attention_LSTM import SA_LSTM
from Data_Loader import Data_Loader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
setup_seed(20)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class Detector(object):
    Event_list = ['prefix_hijack', 'route_leak', 'breakout', 'edge', 'defcon']
    event_nums = [1, 2, 3, 4, 5]
    models=[]
    EPOCH = 10
    LSTM_NUM = 1
    BATCH_SIZE = 8
    Hidden_SIZE = 128
    LSTM_layer_NUM = 1
    LR=0.001
    WINDOW_SIZE = 10
    INPUT_SIZE = 83
    TIME_STEP=1
    true_pred=pd.DataFrame()
    def __init__(self,EPOCH = 10,
                LSTM_NUM = 1,
                BATCH_SIZE = 8,
                Hidden_SIZE = 128,
                LSTM_layer_NUM = 1,WINDOW_SIZE=30):
        self.LSTM_NUM=LSTM_NUM
        self.BATCH_SIZE=BATCH_SIZE
        self.Hidden_SIZE=Hidden_SIZE
        self.LSTM_layer_NUM=LSTM_layer_NUM
        self.WINDOW_SIZE=WINDOW_SIZE

    def set_sliding_window_para(self):
        for i in [4]:
            for window_size in range(15,35,5):
                for j in range (3):
                        self.train(Event_num=i,read_from_file=False, include_MANRS_data=True,WINDOW_SIZE=window_size,HIDDEN_SIZE=128)
    def set_hidden_size_para(self):
        for i in [4]:
            for hidden_size in [64, 128, 256, 512]:
                for j in range (3):
                        self.train(Event_num=i,read_from_file=False, include_MANRS_data=True,WINDOW_SIZE=30,HIDDEN_SIZE=hidden_size)
    def begin_train_all_model(self,baseline=False,baseline_Feature=False):
        '''
        description: to train all the model from our weak dataset
        :param baseline: if is the baseline model (Bool)
        :param baseline_Feature: if is the baseline feature set (Bool)
        :return: save all model under the path ../params/
        '''


        if baseline:
            for i in self.event_nums:
                self.baseline(Event_num=i, read_from_file=False, include_MANRS_data=True,baseline_Feature=baseline_Feature)
        else:


            save_epochs = [7, 1, 2, 8, 8]
            for i in self.event_nums:
                self.train(Event_num=i,read_from_file=False, include_MANRS_data=True,WINDOW_SIZE=30,HIDDEN_SIZE=128,save_epoch=save_epochs[i-1])
    def train(self,Event_num,read_from_file=True, include_MANRS_data=True,WINDOW_SIZE=30,HIDDEN_SIZE=128,save_epoch=10): #the implement of training model
        '''

        :param Event_num: the index type of the anomaly
        :param read_from_file: already have the file to read
        :param include_MANRS_data: join the legitimate data
        :param WINDOW_SIZE: the sliding window size
        :param HIDDEN_SIZE: the hidden size of LSTM model
        :return: save model under the path ../params/best_lstm_params_' + Event_name + '2.pkl
        '''


        self.WINDOW_SIZE=WINDOW_SIZE
        self.Hidden_SIZE=HIDDEN_SIZE

        global SA_LSTM_flag
        Event_name=self.Event_list[Event_num - 1]
        Path = '../params/best_lstm_params_' + Event_name + '.pkl'
        target_list = ['normal', Event_name]
        loader=Data_Loader(Event_name=Event_name,Event_num=Event_num,TIME_STEP=1,WINDOW_SIZE=self.WINDOW_SIZE)

        train_x, train_y, test_x, test_y,eval_x,eval_y = loader.loadDataSet(read_from_file=read_from_file,
                                                        include_MANRS_data=include_MANRS_data)
        self.INPUT_SIZE = loader.INPUT_SIZE
        datasets = Data.TensorDataset(train_x, train_y)
        train_loader = Data.DataLoader(dataset=datasets, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=2)

        eval_x=eval_x.cuda()
        eval_y=eval_y.numpy()
        test_x = test_x.cuda()
        test_y = test_y.numpy()
        if SA_LSTM_flag:
            lstm = SA_LSTM(WINDOW_SIZE=self.WINDOW_SIZE, INPUT_SIZE=self.INPUT_SIZE, Hidden_SIZE=self.Hidden_SIZE,
                           LSTM_layer_NUM=self.LSTM_layer_NUM)
        else:
            lstm = LSTM(WINDOW_SIZE=self.WINDOW_SIZE, INPUT_SIZE=self.INPUT_SIZE, Hidden_SIZE=self.Hidden_SIZE,
                       LSTM_layer_NUM=self.LSTM_layer_NUM)

        lstm = lstm.cuda()
        optimizer = torch.optim.Adam(lstm.parameters(), lr=self.LR)

        loss_func = nn.CrossEntropyLoss()
        h_state = None

        from sklearn.metrics import f1_score
        best_f1_score = 0.0
        best_epoch = 0

        #train_length = len(train_loader)

        for epoch in range(self.EPOCH):
            for step, (x, y) in tqdm(enumerate(train_loader)):
                x = x.cuda()
                y = y.cuda()
                if SA_LSTM_flag:
                    output , attn_weights= lstm(x)
                else:
                    output = lstm(x)

                loss = loss_func(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 10000 == 0:
                    if SA_LSTM_flag:
                        eval_output, attn_weights = lstm(test_x)
                    else:
                        eval_output = lstm(test_x)
                    pred_y = torch.max(eval_output, 1)[1].cpu().data.numpy()

                    #print(pred_y)
                    #print(eval_y)
                    accuracy = float(np.sum(pred_y == test_y)) / float(test_y.size)
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),
                          '| test accuracy: %.2f' % accuracy)
                    from sklearn.metrics import classification_report

                    temp_str = classification_report(y_true=test_y, y_pred=pred_y,
                                                     target_names=target_list)
                    temp_f1 = f1_score(y_pred=pred_y, y_true=test_y, average='macro')
                    print('temp_f1', temp_f1)
                    # temp_sum=temp_f1+temp_route_f1
                    #if (best_f1_score < temp_f1):
                    if(epoch==save_epoch):
                        print(temp_str + '\n' + str(temp_f1))
                        with open('../result_doc/test_best_f1' + Event_name + '.txt', 'a') as f:
                            message = 'epoch:' + str(epoch) + ' f1_score:' + str(temp_f1) + '\n'
                            f.write(message)
                        best_f1_score = temp_f1
                        best_epoch = epoch
                        torch.save(lstm.state_dict(), Path)


        lstm.load_state_dict(torch.load(Path))
        if SA_LSTM_flag:
            test_output, attn_weights= lstm(test_x)
        else:
            test_output = lstm(test_x)
        pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()

        from sklearn.metrics import classification_report

        test_report = classification_report(y_true=test_y, y_pred=pred_y,
                                         target_names=target_list)
        test_parameter_path = '../result_doc/test_parameter' + '_' + Event_name + '.txt'
        with open(test_parameter_path, 'a') as f:
            message = "TimeStep:" + str(self.TIME_STEP) + '\tWINDOW_SIZE:' + str(
                self.WINDOW_SIZE) + "\tLSTM_NUM: " + str(
                self.LSTM_NUM) + '\tLayer num: ' + str(self.LSTM_layer_NUM) + '\tLR:' + str(
                self.LR) + '\tBatch_size: ' + str(
                self.BATCH_SIZE) + '\tHidden_size: ' + str(
                self.Hidden_SIZE) + '\tNormalizer：MinMaxScaler' + '\t epoch:' + str(
                best_epoch) + '\tf1_score:' + str(best_f1_score) + '\n' + 'include_MANRS_data:' + str(
                include_MANRS_data) + '\t time_bins:60s'+'\n' + test_report + '\n\n'
            print(message)

            f.write(message)
        self.models.append(lstm)
        #attn_weights_df=pd.DataFrame(best_attn_weights)
        #print(attn_weights_df)
        #attn_weights_df.to_csv('../result_doc/atten_weights' + '_' + Event_name + '.csv')
        torch.save(lstm, '../params/lstm' + Event_name + '.pkl')

    def test_route_leak(self):
        '''
        description: directly use the after-trained model (from weak dataset) to pred the well-known dataset
        :return: csv file
        '''
        global SA_LSTM_flag
        loader = Data_Loader(Event_name='route_leak', Event_num=2)
        self.INPUT_SIZE = loader.INPUT_SIZE
        x, y0 = loader.loadroute_leak()
        INPUT_SIZE = x.shape[1]
        true_pred = pd.DataFrame()
        Event_list = ['prefix_hijack', 'route_leak', 'breakout', 'edge', 'defcon']
        print(INPUT_SIZE)
        import pickle
        for Event_name in Event_list:
            scaler = pickle.load(open('../params/' + Event_name + '_scaler.pkl', 'rb'))
            x0 = scaler.transform(x.values)
            test_x, test_y = loader.to_timestep(x=x0, y=y0.values, event_len=1440)

            test_x = torch.tensor(test_x, dtype=torch.float32)
            test_y = torch.tensor(np.array(test_y))
            Path = '../params/best_lstm_params_' + Event_name + '.pkl'
            if SA_LSTM_flag:
                model = SA_LSTM(WINDOW_SIZE=self.WINDOW_SIZE, INPUT_SIZE=self.INPUT_SIZE, Hidden_SIZE=128,
                                LSTM_layer_NUM=1)
                model.load_state_dict(torch.load(Path))
                test_output, attn = model(test_x)
            else:
                model = LSTM(WINDOW_SIZE=self.WINDOW_SIZE, INPUT_SIZE=self.INPUT_SIZE, Hidden_SIZE=128, LSTM_layer_NUM=1)
                model.load_state_dict(torch.load(Path))
                test_output = model(test_x)

            pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
            from sklearn.metrics import classification_report
            print(classification_report(y_true=test_y, y_pred=pred_y,
                                        target_names=['Normal', 'Abnomarl ']))
            true_pred['pred_' + Event_name] = pred_y
        true_pred['true'] = test_y
        true_pred.to_csv('../result_doc/pred_true_route_leak_train.csv')

    def transfer_fine_tune(self, Event_name,Scheme='A',save_epoch=0,labelsmoothing=False,confidence=False,base_lr=1e-6,out_lr=1e-4): #the implement of transfer learning
        loader = Data_Loader(Event_name='route_leak', Event_num=2)
        train_x, train_y0,test_x,test_y = loader.loadroute_leak_train_test(scheme=Scheme)
        INPUT_SIZE = train_x.shape[1]
        true_pred = pd.DataFrame()
        #Event_name = 'route_leak'
        self.INPUT_SIZE = loader.INPUT_SIZE
        print(INPUT_SIZE)
        import pickle

        scaler = pickle.load(open('../params/' + Event_name + '_scaler.pkl', 'rb'))
        train_x = scaler.transform(train_x.values)
        train_x, train_y = loader.to_timestep(x=train_x, y=train_y0.values, event_len=1440)
        train_x,eval_x,train_y,eval_y=train_test_split(train_x,train_y,test_size=0.2,random_state=42)

        train_x = torch.tensor(train_x, dtype=torch.float32)
        train_y = torch.tensor(train_y,dtype=torch.long)
        datasets = Data.TensorDataset(train_x, train_y)
        train_loader = Data.DataLoader(dataset=datasets, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=2)

        test_x = scaler.transform(test_x.values)
        test_x, test_y = loader.to_timestep(x=test_x, y=test_y.values, event_len=1440)

        test_x = torch.tensor(test_x, dtype=torch.float32)
        test_y = torch.tensor(np.array(test_y))

        eval_x=torch.tensor(eval_x, dtype=torch.float32)
        eval_y = torch.tensor(np.array(eval_y))
        eval_x = eval_x.cuda()
        eval_y = eval_y.numpy()
        test_x = test_x.cuda()
        test_y = test_y.numpy()

        Path = '../params/best_lstm_params_' + Event_name + '.pkl'

        model = SA_LSTM(WINDOW_SIZE=self.WINDOW_SIZE, INPUT_SIZE=self.INPUT_SIZE, Hidden_SIZE=128,
                        LSTM_layer_NUM=1)
        model.load_state_dict(torch.load(Path))

        model = model.cuda()
        ignored_params = list(map(id, model.out.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer = torch.optim.Adam([{'params': base_params, 'lr': 1e-5},
                                      {'params': model.out.parameters(),'lr':1e-3}])


        if labelsmoothing:
            print("label smoothing")
            loss_func = LabelSmoothingLoss(classes=2, smoothing=0.5) #Special variant of CrossEntropyLoss，to prevent overfitting.
        else:
            loss_func = nn.CrossEntropyLoss()

        from sklearn.metrics import f1_score
        best_f1_score = 0.0

        for epoch in range(self.EPOCH):
            for step, (x, y) in tqdm(enumerate(train_loader)):
                x = x.cuda()
                y = y.cuda()
                output, attn_weights = model(x)
                #print(y)
                loss = loss_func(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 1400 == 0:

                    eval_output, attn_weights = model(eval_x)
                    pred_y = torch.max(eval_output, 1)[1].cpu().data.numpy()
                    accuracy = float(np.sum(pred_y == eval_y)) / float(eval_y.size)
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),
                          '| eval accuracy: %.2f' % accuracy)
                    from sklearn.metrics import classification_report

                    temp_str = classification_report(y_true=eval_y, y_pred=pred_y,
                                                     target_names=['nomral', 'abnormal'])
                    a = classification_report(y_true=eval_y, y_pred=pred_y,
                                                     target_names=['nomral', 'abnormal'],output_dict=True)
                    #temp_f1=a['abnormal']['f1-score']
                    #temp_f1 = f1_score(y_pred=pred_y, y_true=eval_y, average='macro')
                    #print('temp_f1', temp_f1)
                    # temp_sum=temp_f1+temp_route_f1
                    #if (best_f1_score < temp_f1):
                    if (epoch==save_epoch):
                        print(temp_str + '\n' + str(temp_f1))
                        with open('../result_doc/retrain' + Event_name + '.txt', 'a') as f:
                            message = 'epoch:' + str(epoch) + ' f1_score:' + str(temp_f1) + '\n'
                            f.write(message)
                        best_f1_score = temp_f1
                        torch.save(model.state_dict(), '../params/retrain' + Event_name + Scheme+'.pkl')

        path = '../params/retrain' + Event_name + Scheme+'.pkl'
        model.load_state_dict(torch.load(path))
        test_output, attn_weights = model(test_x)

        if confidence:
            output = test_output.cpu().data.numpy()
            pred_y = output[:, 1] # anomaly confidence
        else:
            pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
        #test_report = classification_report(y_true=test_y, y_pred=pred_y,
        #                                    target_names=['Normal', 'Abnormal'])
        #print(test_report)
        self.true_pred['pred_' + Event_name] = pred_y
        self.true_pred['true'] = test_y

    def plot_test_route_leak(self):
        import matplotlib.pyplot as plt

        import pandas as pd
        pred = pd.read_csv("../result_doc/pred_true_route_leak_train.csv")

        name_list = ['pred_route_leak', 'pred_prefix_hijack', 'pred_edge', 'pred_defcon', 'pred_breakout']
        for col in name_list:
            for i in range(len(pred)):
                pred[col].iloc[i] *= -1

        plt.figure()
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.yticks([-1, 0, 1], ['Warning', 'Normal', 'Anomaly'])
        x = [0, 1440, 2880, 4320]
        tick = ['AS200759', 'AS4761', 'AS4788', 'AS9121']
        pred['true'].plot(c='red', label='Ground Truth')
        pred['pred_route_leak'].plot(c='blue', alpha=0.5, label='Route Leak')
        pred['pred_prefix_hijack'].plot(alpha=0.5, label='Prefix Hijack')
        pred['pred_edge'].plot(c='green', alpha=0.5, label='Fake Route')
        pred['pred_defcon'].plot(c='grey', alpha=0.5, label='Defcon')
        pred['pred_breakout'].plot(c='black', alpha=0.5, label='Breakout')
        plt.xticks(x, tick)
        plt.legend()
        plt.show()

    def transfer_learning(self,vote_thr=2,Scheme='A'): #transfer to the well-known anomalies
        '''

        :param vote_thr: the voting threshold for ensemble vote
        :param Scheme: train test spilt scheme
        :return:
        '''
        labelsmoothing=False
        save_epochs=[]
        base_lr=1e-6
        out_lr=1e-4
        output_confidence=False #output confidence or hard label.
        if Scheme=='A':
            save_epochs=[0,0,9,0,0]
            w = np.array([1, 1, 1, 1, 1])
            vote_thr=2
        elif Scheme=='B':# need to retrain the model with weak data,
            self.EPOCH=10
            base_lr = 1e-5
            out_lr = 1e-3
            labelsmoothing = True
            save_epochs = [7, 0, 0, 0, 0]
            w = np.array([1, 0, 0, 0, 0])
            vote_thr=1
        elif Scheme=='C':
            self.EPOCH = 10
            output_confidence = True
            save_epochs = [9, 9, 1, 1, 6]
            w = np.array([0, 1, 0, 0, 0.5])
            vote_thr = 0.54
        elif Scheme=='D':
            self.EPOCH=30
            output_confidence=True
            save_epochs = [8, 0, 17, 9, 1]
            w = np.array([0, 0, 1, 0, 0])
            vote_thr = 0.1

        for i,event_name in enumerate(self.Event_list): #to retrain all the model
            self.transfer_fine_tune(event_name,Scheme=Scheme,save_epoch=save_epochs[i],confidence=output_confidence,labelsmoothing=labelsmoothing,base_lr=base_lr,out_lr=out_lr)

        self.true_pred['new'] = 0
        y_em = self.true_pred.drop(columns=['true'])

        for i in range(len(self.true_pred)):
            if ((w*y_em.iloc[i,[0,1,2,3,4]]).sum() >= vote_thr):
                self.true_pred['new'].iloc[i] = 1
        print("the ensemble pred result:")
        print(classification_report(y_pred=self.true_pred['new'], y_true=self.true_pred['true'], target_names=['normal', 'abnormal']))

        self.true_pred.to_csv('../result_doc/retrain_'+Scheme+'.csv')

    def test(self,data_path, event_name, window): #API for testing on the other datasets
        x, y0 = Data_Loader.load(data_path, window)
        INPUT_SIZE = x.shape[1]
        true_pred = pd.DataFrame()
        Event_list = ['prefix_hijack', 'route_leak', 'breakout', 'edge', 'defcon']
        print(INPUT_SIZE)
        import pickle
        for Event_name in Event_list:
            scaler = pickle.load(open('../params/' + Event_name + '_scaler.pkl', 'rb'))
            x0 = scaler.transform(x.values)
            test_x, test_y = Data_Loader.to_timestep(x0, y0.values, window)

            test_x = torch.tensor(test_x, dtype=torch.float32)
            test_y = torch.tensor(np.array(test_y))
            Path = '../params/best_lstm_params_' + Event_name + '.pkl'
            model = LSTM()
            model.load_state_dict(torch.load(Path))
            test_output = model(test_x)
            output = test_output.detach().numpy()
            output_event_conf = output[:, 1]
            # print(len(output_event_conf))
            # print(output_event_conf[output_event_conf>0.8])
            condition_1 = np.array(output_event_conf > 0.9, dtype=int)
            condition_2 = torch.max(test_output, 1)[1].cpu().data.numpy()
            pred_y = condition_1 & condition_2
            from sklearn.metrics import classification_report
            target_l = ['normal']
            target_l.append(event_name)
            print(classification_report(y_true=test_y, y_pred=pred_y,
                                        target_names=target_l))
            true_pred['pred_' + Event_name] = pred_y
        true_pred['true'] = test_y
        true_pred.to_csv('../result_doc/pred_true_' + event_name + '.csv')

    def baseline(self, Event_num, read_from_file=True, include_MANRS_data=True,baseline_Feature=False): #baseline method SVM,RF

        Event_name = self.Event_list[Event_num - 1]
        target_list = ['normal', Event_name]
        loader = Data_Loader(Event_name=Event_name, Event_num=Event_num,TIME_STEP=1,WINDOW_SIZE=self.WINDOW_SIZE)

        train_x, train_y, test_x, test_y,eval_x,eval_y= loader.loadDataSet(read_from_file=read_from_file,
                                                              include_MANRS_data=include_MANRS_data,baseline=baseline_Feature)
        self.INPUT_SIZE = loader.INPUT_SIZE
        train_x=train_x.reshape([-1,self.INPUT_SIZE * self.WINDOW_SIZE])
        test_x = test_x.reshape([-1, self.INPUT_SIZE * self.WINDOW_SIZE])
        from sklearn import tree
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        tuned_param_SVC = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        scores = ['precision', 'f1']
        tuned_param_RT=[{
            'n_estimators': range(20, 500, 30),
            }]
        for score in scores:
            svc=GridSearchCV(SVC(), tuned_param_SVC, scoring='%s_macro' % score,n_jobs=8)
            rf = GridSearchCV(RandomForestClassifier(bootstrap=True,oob_score=True), tuned_param_RT, scoring='%s_macro' % score,n_jobs=8)
            # tree=tree.DecisionTree()
            models=[svc,rf]
            for clf in models:
                clf.fit(train_x, train_y)
                pred_y = clf.predict(test_x)
                message='Tuning hyper-parameters for %s'%score
                message="Best parameters set found on development set:\n\n"
                message+=str(clf.best_params_)
                message+='\n\n'
                message+="Grid scores on development set:\n\n"

                print("Best parameters set found on development set:")
                print()
                print(clf.best_params_)
                print()
                print("Grid scores on development set:")
                print()
                means = clf.cv_results_['mean_test_score']
                stds = clf.cv_results_['std_test_score']
                for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                    print("%0.3f (+/-%0.03f) for %r"
                          % (mean, std * 2, params))
                    message+="%0.3f (+/-%0.03f) for %r \n"% (mean, std * 2, params)
                message+="\nDetailed classification report:\n\nThe model is trained on the full development set.\nThe scores are computed on the full evaluation set.\n\n"
                print()

                print("Detailed classification report:")
                print()
                print("The model is trained on the full development set.")
                print("The scores are computed on the full evaluation set.")
                print()
                from sklearn.metrics import classification_report
                message+=str(classification_report(y_true=test_y, y_pred=pred_y,target_names=target_list))
                print(classification_report(y_true=test_y, y_pred=pred_y,target_names=target_list))
                baseline_txt_path = '../result_doc/baseline.txt'
                with open(baseline_txt_path, 'a') as f:
                    f.write(message)



SA_LSTM_flag=True
detector=Detector()
detector.begin_train_all_model(baseline=False,baseline_Feature=False)
detector.test_route_leak()
detector.transfer_learning(Scheme='A')




