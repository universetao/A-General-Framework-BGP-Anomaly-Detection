
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from LSTM_model import LSTM
from Self_Attention_LSTM import SA_LSTM
from Data_Loader import Data_Loader
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
class Detector(object):
    Event_list = ['prefix_hijack', 'route_leak', 'breakout', 'edge', 'defcon']
    event_nums = [1, 2, 3, 4, 5]
    models=[]
    EPOCH = 100
    LSTM_NUM = 1
    BATCH_SIZE = 8
    Hidden_SIZE = 128
    LSTM_layer_NUM = 1
    LR=0.001
    WINDOW_SIZE = 30
    INPUT_SIZE = 83
    TIME_STEP=1
    def __init__(self,EPOCH = 100,
                LSTM_NUM = 1,
                BATCH_SIZE = 8,
                Hidden_SIZE = 128,
                LSTM_layer_NUM = 1,WINDOW_SIZE=30):
        self.LSTM_NUM=LSTM_NUM
        self.BATCH_SIZE=BATCH_SIZE
        self.Hidden_SIZE=Hidden_SIZE
        self.LSTM_layer_NUM=LSTM_layer_NUM
        self.WINDOW_SIZE=WINDOW_SIZE
    def plot_confidence_tree(self):
        Event_name = 'all'
        event_List = ['prefix_hijack', 'route_leak', 'edge', 'defcon']
        # event_List = [ 'route_leak', 'breakout', 'edge', 'defcon']
        loader = Data_Loader(Event_name=Event_name, Event_num=0,WINDOW_SIZE=self.WINDOW_SIZE,Hijack_Window=39)
        self.INPUT_SIZE = loader.INPUT_SIZE
        train_x, train_y, test_x, test_y = loader.loadDataSet(read_from_file=False,
                                                              include_MANRS_data=True)
        confidence_df = pd.DataFrame()
        confidence_test_df = pd.DataFrame()

        from collections import defaultdict

        for event_name in event_List:
            print(event_name + '\n')
            # / home / dongyutao / BGP_LSTM / best_lstm_route_leak.pkl
            Path = '../params/best_lstm_params_' + event_name + '.pkl'
            model = LSTM()
            # model = model.cuda()
            model.load_state_dict(torch.load(Path))
            # model = torch.load(Path)
            print('load_finish')

            print('begin_pred')
            print(train_x.shape)
            pred_output = model(train_x)
            print(pred_output.shape)
            # pred_y = torch.max(pred_output, 1)[1].detach().numpy()
            pred_output = pred_output.detach().numpy()
            # confidence_df[event_name + '_normal'] = pred_output[:,0]
            confidence_df[event_name] = pred_output[:, 1]
            # confidence_df[event_name] = pred_y
            pred_output_test = model(test_x)
            pred_output_test = pred_output_test.detach().numpy()
            # pred_y_test = torch.max(pred_output_test, 1)[1].detach().numpy()
            # confidence_test_df[event_name + '_normal'] = pred_output_test[:,0]
            confidence_test_df[event_name] = pred_output_test[:, 1]
            # confidence_test_df[event_name] =pred_output_test

        # / home / dongyutao / BGP_LSTM / best_lstm_route_leak.pkl
        Path = '../params/best_lstm_params_' + 'breakout2' + '.pkl'
        model = LSTM()
        model.load_state_dict(torch.load(Path))
        print('load_finish')

        print('begin_pred')
        print(train_x.shape)
        pred_output = model(train_x)
        print(pred_output.shape)
        pred_y = torch.max(pred_output, 1)[1].detach().numpy()
        confidence_df['breakout'] = pred_y
        pred_output_test = model(test_x)
        pred_y_test = torch.max(pred_output_test, 1)[1].detach().numpy()

        confidence_test_df['breakout'] = pred_y_test

        print('confidence_df shape:', confidence_df.shape)
        X, Y = confidence_df.drop(columns=['breakout'], axis=1), confidence_df['breakout']
        t_x, t_y = confidence_test_df.drop(columns=['breakout'], axis=1), confidence_test_df['breakout']

        from sklearn.tree import DecisionTreeClassifier
        from sklearn import tree
        clf = DecisionTreeClassifier(max_depth=4)
        clf = clf.fit(X, Y)
        pred_y = clf.predict(t_x)
        from sklearn.metrics import classification_report
        # print(classification_report(y_true=test_y, y_pred=pred_y,
        #                             target_names=['正常', '前缀劫持', '路由泄露', '中断', '虚假路径', 'defcon']))
        print(classification_report(y_true=t_y, y_pred=pred_y, target_names=['未发生中断', '发生中断']))
        print(X.columns)
        import matplotlib.pyplot as plt
        # fn = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        # cn = ['normal', 'prefix_hijack', 'route_leak', 'breakout', 'edge', 'defcon']
        cn = ['no_breakout', 'breakout']
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 3), dpi=600)
        tree.plot_tree(clf,
                       feature_names=X.columns,
                       class_names=cn,
                       filled=True)
        plt.show()
        fig.savefig('./result_pic/confidence_tree.png', dpi=600)
    def set_sliding_window_para(self):
        for i in self.event_nums:
            for window_size in range(15,35,5):
                for j in range (3):
                        self.train(Event_num=i,read_from_file=False, include_MANRS_data=True,WINDOW_SIZE=window_size,HIDDEN_SIZE=128)
    def set_hidden_size_para(self):
        for i in self.event_nums:
            for hidden_size in [64, 128, 256, 512]:
                for j in range (3):
                        self.train(Event_num=i,read_from_file=False, include_MANRS_data=True,WINDOW_SIZE=30,HIDDEN_SIZE=hidden_size)
    def begin_train_all_model(self,baseline=False,baseline_Feature=False):


        if baseline:
            for i in self.event_nums:
                self.baseline(Event_num=i, read_from_file=False, include_MANRS_data=True,baseline_Feature=baseline_Feature)
        else:
            for i in self.event_nums:
                for window_size in range(15,35,5):
                    for hidden_size in [64,128,256,512]:
                        self.train(Event_num=i,read_from_file=False, include_MANRS_data=True,WINDOW_SIZE=window_size,HIDDEN_SIZE=hidden_size)
    def train(self,Event_num,read_from_file=True, include_MANRS_data=True,WINDOW_SIZE=30,HIDDEN_SIZE=128):
        self.WINDOW_SIZE=WINDOW_SIZE
        self.Hidden_SIZE=HIDDEN_SIZE

        Event_name=self.Event_list[Event_num - 1]
        target_list = ['正常', Event_name]
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

        lstm = SA_LSTM(WINDOW_SIZE=self.WINDOW_SIZE,INPUT_SIZE=self.INPUT_SIZE,Hidden_SIZE=self.Hidden_SIZE,LSTM_layer_NUM=self.LSTM_layer_NUM)
        print(lstm)
        lstm = lstm.cuda()
        optimizer = torch.optim.Adam(lstm.parameters(), lr=self.LR)

        loss_func = nn.CrossEntropyLoss()
        h_state = None

        from sklearn.metrics import f1_score
        best_f1_score = 0.0
        best_epoch = 0

        train_length = len(train_loader)

        for epoch in range(self.EPOCH):
            for step, (x, y) in tqdm(enumerate(train_loader)):
                x = x.cuda()
                y = y.cuda()

                output , attn_weights= lstm(x)

                loss = loss_func(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 10000 == 0:
                    eval_output,attn_weights = lstm(eval_x)
                    pred_y = torch.max(eval_output, 1)[1].cpu().data.numpy()

                    print(pred_y)
                    print(eval_y)
                    accuracy = float(np.sum(pred_y == eval_y)) / float(eval_y.size)
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(),
                          '| test accuracy: %.2f' % accuracy)
                    from sklearn.metrics import classification_report

                    temp_str = classification_report(y_true=eval_y, y_pred=pred_y,
                                                     target_names=target_list)
                    temp_f1 = f1_score(y_pred=pred_y, y_true=eval_y, average='macro')
                    print('temp_f1', temp_f1)
                    # temp_sum=temp_f1+temp_route_f1
                    if (best_f1_score < temp_f1):
                        print(temp_str + '\n' + str(temp_f1))
                        with open('../result_doc/test_best_f1' + Event_name + '3.txt', 'a') as f:
                            message = 'epoch:' + str(epoch) + ' f1_score:' + str(temp_f1) + '\n'
                            f.write(message)
                        best_f1_score = temp_f1
                        best_epoch = epoch
                        best_attn_weights=attn_weights.cpu().detach().numpy()
                        print(best_attn_weights)
                        attn_weights_df = pd.DataFrame(best_attn_weights)
                        print(attn_weights_df)
                        attn_weights_df.to_csv('../result_doc/atten_weights' + '_' + Event_name + '.csv')
                        torch.save(lstm.state_dict(), '../params/best_lstm_params_' + Event_name + '3.pkl')



        Path = '../params/best_lstm_params_' + Event_name + '3.pkl'
        lstm.load_state_dict(torch.load(Path))
        test_output, attn_weights= lstm(test_x)
        pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()

        from sklearn.metrics import classification_report

        test_report = classification_report(y_true=test_y, y_pred=pred_y,
                                         target_names=target_list)
        test_parameter_path = '../result_doc/test_parameter' + '_' + Event_name + '3.txt'
        with open(test_parameter_path, 'a') as f:
            message = "TimeStep:" + str(self.TIME_STEP) + '\tWINDOW_SIZE:' + str(
                self.WINDOW_SIZE) + "\tLSTM_NUM: " + str(
                self.LSTM_NUM) + '\tLayer num: ' + str(self.LSTM_layer_NUM) + '\tLR:' + str(
                self.LR) + '\tBatch_size: ' + str(
                self.BATCH_SIZE) + '\tHidden_size: ' + str(
                self.Hidden_SIZE) + '\tNormalizer：MinMaxScaler' + '\t epoch:' + str(
                best_epoch) + '\tf1_score:' + str(best_f1_score) + '\n' + 'include_MANRS_data:' + str(
                include_MANRS_data) + '\t time_bins:30s'+'\n' + test_report + '\n\n'
            print(message)

            f.write(message)
        self.models.append(lstm)
        attn_weights_df=pd.DataFrame(best_attn_weights)
        print(attn_weights_df)
        attn_weights_df.to_csv('../result_doc/atten_weights' + '_' + Event_name + '.csv')
        torch.save(lstm, '../params/lstm' + Event_name + '.pkl')

    def test_route_leak(self):
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
            model = SA_LSTM(WINDOW_SIZE=self.WINDOW_SIZE,INPUT_SIZE=self.INPUT_SIZE,Hidden_SIZE=128,LSTM_layer_NUM=1)
            model.load_state_dict(torch.load(Path))
            test_output,attn = model(test_x)
            pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
            from sklearn.metrics import classification_report
            print(classification_report(y_true=test_y, y_pred=pred_y,
                                        target_names=['正常', '大型路由劫持']))
            true_pred['pred_' + Event_name] = pred_y
        true_pred['true'] = test_y
        true_pred.to_csv('../result_doc/pred_true_route_leak_train.csv')

    def test(self,data_path, event_name, window):
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
            target_l = ['正常']
            target_l.append(event_name)
            print(classification_report(y_true=test_y, y_pred=pred_y,
                                        target_names=target_l))
            true_pred['pred_' + Event_name] = pred_y
        true_pred['true'] = test_y
        true_pred.to_csv('../result_doc/pred_true_' + event_name + '.csv')

    def real_time_detect(self):
        pass
    def baseline(self, Event_num, read_from_file=True, include_MANRS_data=True,baseline_Feature=False):

        Event_name = self.Event_list[Event_num - 1]
        target_list = ['正常', Event_name]
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




detector=Detector()
# detector.begin_train_all_model(baseline=False,baseline_Feature=False)
# detector.set_hidden_size_para()
# detector.train(Event_num=1,read_from_file=False, include_MANRS_data=True,WINDOW_SIZE=30,HIDDEN_SIZE=128)
detector.test_route_leak()