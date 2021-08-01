#!/usr/bin/env python
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import pybgpstream


def dd():
    return defaultdict(int)  
def ds():
    return defaultdict(set)
def dl():
    return defaultdict(list)


class Feature_Extractor(object):
    def __init__(self):
        pass

    def edit_distance(self,l1, l2):  # 
        """

        :param l1: list 1
        :param l2: list 2
        :return: edit distance between l1 and l2
        """
        rows = len(l1) + 1
        cols = len(l2) + 1

        dist = [[0 for x in range(cols)] for x in range(rows)]

        for i in range(1, rows):
            dist[i][0] = i
        for i in range(1, cols):
            dist[0][i] = i

        for col in range(1, cols):
            for row in range(1, rows):
                if l1[row - 1] == l2[col - 1]:
                    cost = 0
                else:
                    cost = 1
                dist[row][col] = min(dist[row - 1][col] + 1,  # deletion
                                     dist[row][col - 1] + 1,  # insertion
                                     dist[row - 1][col - 1] + cost)  # substitution
        return dist[row][col], rows - cols

    def is_sub_pfx(self,father_prefix, sub_prefix):
        father_pfx, f_mask = father_prefix.split('/')
        sub_pfx, sub_mask = sub_prefix.split('/')
        f_mask = int(f_mask)
        sub_mask = int(sub_mask)
        if (f_mask > sub_mask):
            return False
        elif (f_mask % 8 == 0):  # 
            block = int(f_mask / 8)
            list1 = father_pfx.split('.')
            list2 = sub_pfx.split('.')
            return (list1[0:block] == list2[0:block])
        else:  # 
            father_IP_bin = self.to_bin(father_pfx)
            sub_IP_bin = self.to_bin(sub_pfx)
            father_IP_bin = father_IP_bin[0:f_mask]
            sub_IP_bin = sub_IP_bin[0:f_mask]
            if father_IP_bin == sub_IP_bin:
                return True
            else:
                return False

    def to_bin(self,IP):
        list1 = IP.split('.')
        list2 = []
        for item in list1:
            try:
                item = bin(int(item))  # ---0b11000000 0b10101000 0b1100000 0b1011110 ----

                # cut the first 2 bin :0b.
                item = item[2:]
            except:
                print(IP)

            list2.append(item.zfill(8))  # --['11000000', '10101000', '01100000', '01011110']--
        v2 = ''.join(list2)  # ----11000000101010000110000001011110----
        # print(v2)
        return v2

    def real_time_extract(self):
        pass
    def get_his_info(self):
        events = pd.read_csv('../events_data/edge_events_from_bgpObservatory1.csv')
        begin_minute = 120
        until_minute = 60
        label_range = [1, 2, 3, 0]  # 0 means all
        Window_size = 2  # min
        total_num = events.shape[0]  # event_total_num in this list
        duration = 0
        all_features = pd.DataFrame()
        for i in range(total_num):
            event_after = False
            VICTIM_AS = str(events['victim_AS'].iloc[i])
            begin_timestamp = events['timestamp'].iloc[i] - begin_minute * 60
            until_timestamp = events['timestamp'].iloc[i] + until_minute * 60
            begin_time = str(pd.to_datetime(begin_timestamp, unit='s'))
            until_time = str(pd.to_datetime(until_timestamp, unit='s'))
            start_time = events['timestamp'].iloc[i]
            label = events['label'].iloc[i]
            # duration=events['Duration'].iloc[i]
            count_2 = 2
            count_3 = 3
            step = 0

            stream = pybgpstream.BGPStream(
                # Consider this time interval:
                # Sat, 01 Aug 2015 7:50:00 GMT -  08:10:00 GMT
                from_time=begin_time, until_time=until_time,
                collectors=["rrc00", 'rrc01', 'rrc02', 'rrc03', 'rrc04', 'rrc05'
                    , 'rrc06', 'rrc07', 'rrc08', 'rrc09', 'rrc10', 'rrc11'
                    , 'rrc12', 'rrc13', 'rrc14', 'rrc15', 'rrc16', 'rrc18'
                    , 'rrc19', 'rrc20', 'rrc21', 'rrc22', 'rrc23'],
                record_type="updates",  # announcement、rib field中才有as-path，
                # withdrawal中只有prefix
                filter='ipversion 4'
                # filter='ipversion 4 and path "_{:}$"'.format(VICTIM_AS),
            )

            # <prefix, origin-ASns-set > dictionary
            path = set()  
            victim_prefix = set()  
            MOAS = defaultdict(set)  
            old_time = 0
            first = True
            peer = set()  # victime's peer
            peer_num = 0  

            peer_increase = defaultdict(int)  
            features = pd.DataFrame()  # [timestamp][feature][values]

            # prefix_origin = defaultdict(set)   #prefix origin pairs
            temp = list

            # feature
            MPL = defaultdict(int)  # MAX path length in time t
            PL = defaultdict(dd)  # each number of path len in time t.
            MOAS_AS = defaultdict(ds)  # Number of ASes that conflict with victime AS
            old_AS_Path = defaultdict(list)  
            new_AS_Path = defaultdict(list)  
            diff_AS_Path = defaultdict(dd)  
            diff_AS_Path_num = defaultdict(dd)  # AS-PATH edit distance set
            withdraw_num = defaultdict(int)
            new_sub_prefix_num = defaultdict(int)  # Number of new sub-prefixes belongs to Victim AS
            new_sub_prefix = defaultdict(set)  
            own_ann_num = defaultdict(int)  # Number of Announcements from victim AS
            MOAS_ann_num = defaultdict(int)  # Number of announcements from origin conflict AS
            ann_shorter = defaultdict(dd)  # AS-PATH length decrease set
            ann_longer = defaultdict(dd)  # AS-PATH length increase set

            Diff_Ann = defaultdict(int)
            duplicate_ann = defaultdict(int)
            withdraw_unique_prefix = defaultdict(set)
            # IGP_num=defaultdict(int)
            # EGP_num=defaultdict(int)
            # incomplete_packets=defaultdict(int)
            new_MOAS = defaultdict(int)

            avg_edit_distance = 0

            avg_PL = 0

            # 标签
            labels = defaultdict(dd)
            for rec in tqdm(stream.records()):
                for elem in rec:  
                    # Get the prefix
                    if (first == True):
                        old_time = elem.time
                        first = False
                    pfx = elem.fields["prefix"]
                    # Get the list of ASes in the AS path
                    # print(elem)
                    if (elem.type == 'A'):
                        ases = elem.fields["as-path"].split(" ")

                        len_path = len(ases)  # AS-PATH len

                        if len_path > 0:
                            # Get the origin ASn (rightmost)
                            origin = ases[-1]

                            if (origin == VICTIM_AS):  
                                own_ann_num[old_time] += 1
                                if pfx not in victim_prefix:
                                    for father_pfx in victim_prefix:  # if it's the new_subprefix
                                        if self.is_sub_pfx(father_pfx, pfx):
                                            new_sub_prefix_num[old_time] += 1
                                            new_sub_prefix[old_time].add(pfx)
                                            break
                                    victim_prefix.add(pfx)
                                peer = ases[0]

                                if peer not in new_AS_Path.keys():
                                    peer_num += 1
                                    peer_increase[old_time] += 1
                                    new_AS_Path[peer] = ases
                                    path_str = 'len_path' + str(len_path)
                                    PL[old_time][path_str] += 1
                                    if (len_path > MPL[old_time]):
                                        MPL[old_time] = len_path
                                else:
                                    if ases != new_AS_Path[peer]:  # if path change, calculate it's edit distance
                                        Diff_Ann[old_time] += 1
                                        old_AS_Path[peer] = new_AS_Path[peer]
                                        new_AS_Path[peer] = ases
                                        num, len_cut = self.edit_distance(old_AS_Path[peer], new_AS_Path[peer])
                                        if (len_cut > 0):
                                            ann_shorter_str = 'ann_shorter_' + str(len_cut)
                                            ann_shorter[old_time][ann_shorter_str] += 1
                                        else:
                                            ann_longer_str = 'ann_longer_' + str(-len_cut)
                                            ann_longer[old_time][ann_longer_str] += 1
                                        diff_num = 'diff_' + str(num)
                                        # diff_peer = 'diff_peer_' + str(peer)
                                        diff_AS_Path_num[old_time][diff_num] += 1
                                        # diff_AS_Path[old_time][diff_peer] = num
                                        path_str = 'len_path' + str(len_path)
                                        PL[old_time][path_str] += 1
                                        if (len_path > MPL[elem.time]):
                                            MPL[old_time] = len_path

                                    else:
                                        duplicate_ann[old_time] += 1
                                # print(elem.fields["as-path"])
                                # print(pfx)
                            # Insert the origin ASn in the set of
                            # origins for the prefix
                            else:
                                if pfx in victim_prefix:
                                    MOAS_ann_num[old_time] += 1
                                    if origin not in MOAS:
                                        new_MOAS[old_time] += 1
                                        MOAS[old_time].add(origin)

                                    MOAS_AS[old_time][pfx].add(origin)

                    elif (elem.type == 'W'):
                        if pfx in victim_prefix:
                            withdraw_num[old_time] += 1
                            withdraw_unique_prefix[old_time].add(pfx)

                    if (elem.time >= (old_time + 30 * Window_size)):
                        if (abs(old_time - start_time) < 30):  # label our date
                            labels[old_time]['label_1'] = label
                            event_after = True
                        # print(abs(old_time-start_time))
                        if event_after:
                            labels[old_time]['label_0'] = label
                            if (count_2 > 0):
                                labels[old_time]['label_2'] = label
                                count_2 -= 1
                            if (count_3 > 0):
                                labels[old_time]['label_3'] = label
                                count_3 -= 1
                        else:

                            labels[old_time]['label_0'] = 0
                            labels[old_time]['label_1'] = 0
                            labels[old_time]['label_2'] = 0
                            labels[old_time]['label_3'] = 0

                        df = pd.DataFrame({'time': pd.to_datetime(old_time, unit='s'),
                                           'MPL': MPL[old_time],
                                           'MOAS_prefix_num': len(MOAS_AS[old_time]),
                                           'MOAS_AS': [MOAS_AS[old_time]],
                                           'MOAS': [MOAS[old_time]],
                                           'new_MOAS': new_MOAS[old_time],
                                           'MOAS_num': len(MOAS[old_time]),
                                           'withdraw_num': withdraw_num[old_time],
                                           'peer_increase': peer_increase[old_time],
                                           'peer_num': peer_num,
                                           'new_prefix_num': new_sub_prefix_num[old_time],
                                           'MOAS_Ann_num': MOAS_ann_num[old_time],
                                           'own_Ann_num': own_ann_num[old_time],
                                           'new_sub_prefix': [new_sub_prefix[old_time]],
                                           'Victim_AS': VICTIM_AS,
                                           'Diff_Ann': Diff_Ann[old_time],
                                           'duplicate_ann': duplicate_ann[old_time],
                                           'withdraw_unique_prefix_num': len(withdraw_unique_prefix[old_time]),
                                           'withdraw_unique_prefix': [withdraw_unique_prefix[old_time]],
                                           }, index=[old_time])
                        d1 = pd.DataFrame(diff_AS_Path_num[old_time], index=[old_time])
                        # d2=pd.DataFrame(diff_AS_Path[old_time],index=[old_time])
                        d2 = pd.DataFrame(labels[old_time], index=[old_time])
                        d3 = pd.DataFrame(PL[old_time], index=[old_time])
                        d5 = pd.DataFrame(ann_shorter[old_time], index=[old_time])
                        d6 = pd.DataFrame(ann_longer[old_time], index=[old_time])

                        # df2=pd.concat([d1,d3],axis=1)
                        d4 = pd.concat([df, d1, d2, d3, d5, d6], axis=1)
                        print(d4)

                        features = features.append(d4)
                        old_time = elem.time
            print(features)
            print(features['label_0'])
            # print(victim_prefix)
            # Print the list of MOAS prefix and their origin ASns
            for pfx in victim_prefix:
                print('victim_prefix', pfx)
            for time in MOAS_AS:
                for pfx in MOAS_AS[time]:
                    print((time, pfx, ','.join(MOAS_AS[time][pfx])))
            for time in MPL:
                print('MPL', MPL[time])
            for time in PL:
                print('path_legth_num', PL[time])
            for time in diff_AS_Path:
                print('edit_distance', diff_AS_Path[time])
            all_features = all_features.append((features))
            features.to_csv('../datasets/features' + VICTIM_AS + '_' + str(begin_time) + '.csv')
        # all_features.to_csv('../all_datasets/add_bgpstream.csv')

    def delete_file_with_MPL_0(self,path):
        import os
        datasets_path=path
        datasets_files = os.listdir(datasets_path)
        count=0
        for data_file in datasets_files:
            try:
                temp = pd.read_csv(datasets_path + data_file, index_col=0)
                if temp['MPL'].sum()<5:
                    print(datasets_path + data_file+'is removed\n')
                    os.remove(datasets_path + data_file)
                    count+=1
            except:
                print(datasets_path + data_file)
        print('remove {:} files numbers'.format(count))
#extractor=Feature_Extractor()
#extractor.get_his_info()
# extractor.delete_file_with_MPL_0('../datasets_30s/')
