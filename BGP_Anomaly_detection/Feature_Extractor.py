#!/usr/bin/env python
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import pybgpstream


def dd():
    return defaultdict(int)  # 返回默认参数类别为int，即0，当字典查找不到值时候。
def ds():
    return defaultdict(set)
def dl():
    return defaultdict(list)


class Feature_Extractor(object):
    def __init__(self):
        pass

    def edit_distance(self,l1, l2):  # 计算l1,l2的编制距离
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
        elif (f_mask % 8 == 0):  # 如果是8的整数，可以方便比较字符串是否相同
            block = int(f_mask / 8)
            list1 = father_pfx.split('.')
            list2 = sub_pfx.split('.')
            return (list1[0:block] == list2[0:block])
        else:  # 否则转换成二进制比较
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

                # 去掉每段二进制前的0b.
                item = item[2:]
            except:
                print(IP)

            # 将IP地址地址的每个字段转换成八位，不足的在每段前补0.
            list2.append(item.zfill(8))  # --['11000000', '10101000', '01100000', '01011110']--

        # 将4段8位二进制连接起来，变成32个0101的样子.
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
        Window_size = 1  # min
        total_num = events.shape[0]  # 事件总数
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
            path = set()  # 受害者path路径
            victim_prefix = set()  # 受害者拥有的所有前缀
            MOAS = defaultdict(set)  # 受害者AS拥有的所有冲突AS集合
            old_time = 0
            first = True
            peer = set()  # 对等体集合
            peer_num = 0  # 对等体的数量

            peer_increase = defaultdict(int)  # 对等体增加的数量
            features = pd.DataFrame()  # [timestamp][feature][values]

            # prefix_origin = defaultdict(set)   #跟victim声明同样前缀的集合，可以获得该AS的哪个前缀被劫持。
            temp = list

            # feature
            # MOAS = defaultdict(int)  # 在某个时刻的前缀冲突MOAS值
            MPL = defaultdict(int)  # 在某个时刻最长的path
            PL = defaultdict(dd)  # 在某个时刻每个长度path的数量
            MOAS_AS = defaultdict(ds)  # 与之冲突的AS号,包括subPrefix冲突
            old_AS_Path = defaultdict(list)  # 上一时刻，每个peer到达该对等体的AS路径
            new_AS_Path = defaultdict(list)  # 当前时刻，每个peer到达该对等体的路径
            diff_AS_Path = defaultdict(dd)  # 与上一时刻，peer到达origin路径改变的编辑距离。
            diff_AS_Path_num = defaultdict(dd)  # 与上一时刻，peer到达origin路径改变的编辑距离数量。
            withdraw_num = defaultdict(int)
            new_sub_prefix_num = defaultdict(int)  # 同个起源AS，但是是已有前缀的子前缀 的数量
            new_sub_prefix = defaultdict(set)  # 新出现的已有前缀的子前缀
            own_ann_num = defaultdict(int)  # 自己的重复宣告数量
            MOAS_ann_num = defaultdict(int)  # 其他AS重复宣告
            ann_shorter = defaultdict(dd)  # 路径变短的数量
            ann_longer = defaultdict(dd)  # 路径变短的数量

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
                for elem in rec:  # 能否按照时间来遍历。
                    # Get the prefix
                    if (first == True):
                        old_time = elem.time
                        first = False
                    pfx = elem.fields["prefix"]
                    # Get the list of ASes in the AS path
                    # print(elem)
                    if (elem.type == 'A'):
                        ases = elem.fields["as-path"].split(" ")

                        len_path = len(ases)  # 计算该长度

                        if len_path > 0:
                            # Get the origin ASn (rightmost)
                            origin = ases[-1]

                            if (origin == VICTIM_AS):  # 如果是受害者AS的前缀，则加入集合中
                                own_ann_num[old_time] += 1
                                if pfx not in victim_prefix:
                                    for father_pfx in victim_prefix:  # 判断是否是第一次出现的子前缀
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
                                    if ases != new_AS_Path[peer]:  # 如果路径更改的话，计算编辑距离
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

                                # pfx是否是subprefix攻击。
                                # else:
                                # for father_pfx in victim_prefix:
                                #     if is_sub_pfx(father_pfx, pfx):
                                #         MOAS_ann_num[old_time]+=1
                                #         # print(pfx)
                                #         # print(str(origin))
                                #         # print(ases)
                                #         if origin not in MOAS:
                                #             new_MOAS[old_time] += 1
                                #             MOAS.add(origin)
                                #         MOAS_AS[old_time][pfx].add(origin)
                                #         #MOAS[old_time] += 1
                                #         victim_prefix.add(pfx)
                                #         break
                    elif (elem.type == 'W'):
                        if pfx in victim_prefix:
                            withdraw_num[old_time] += 1
                            withdraw_unique_prefix[old_time].add(pfx)

                    if (elem.time >= (old_time + 30 * Window_size)):
                        if (abs(old_time - start_time) < 30):  # 这里6是解决时间误差，
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
            features.to_csv('../datasets_30s/features' + VICTIM_AS + '_' + str(begin_time) + '.csv')
        # all_features.to_csv('../all_datasets/add_bgpstream.csv')
    def get_his_info_new(self):
        events = pd.read_csv('../events_data/events_from_bgpstream_part1.csv')
        begin_minute = 120
        until_minute = 60
        label_range = [1, 2, 3, 0]  # 0 means all
        Window_size = 1  # min
        total_num = events.shape[0]  # 事件总数
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
            path = set()  # 受害者path路径
            victim_prefix = set()  # 受害者拥有的所有前缀
            MOAS = defaultdict(set)  # 受害者AS拥有的所有冲突AS集合
            old_time = 0
            first = True
            peer = set()  # 对等体集合
            peer_num = 0  # 对等体的数量

            peer_increase = defaultdict(int)  # 对等体增加的数量
            features = pd.DataFrame()  # [timestamp][feature][values]

            # prefix_origin = defaultdict(set)   #跟victim声明同样前缀的集合，可以获得该AS的哪个前缀被劫持。
            temp = list

            # feature
            # <prefix, origin-ASns-set > dictionary
            prefix_origin = defaultdict(set)

            MPL = defaultdict(int)  # 在某个时刻最长的path
            PL = defaultdict(dd)  # 在某个时刻每个长度path的数量
            MOAS_AS = defaultdict(ds)  # 与之冲突的AS号,包括subPrefix冲突
            old_AS_Path = defaultdict(list)  # 上一时刻，每个peer到达该对等体的AS路径
            new_AS_Path = defaultdict(list)  # 当前时刻，每个peer到达该对等体的路径
            diff_AS_Path = defaultdict(dd)  # 与上一时刻，peer到达origin路径改变的编辑距离。
            diff_AS_Path_num = defaultdict(dd)  # 与上一时刻，peer到达origin路径改变的编辑距离数量。
            withdraw_num = defaultdict(int)
            new_sub_prefix_num = defaultdict(int)  # 同个起源AS，但是是已有前缀的子前缀 的数量
            new_sub_prefix = defaultdict(set)  # 新出现的已有前缀的子前缀
            own_ann_num = defaultdict(int)  # 自己的重复宣告数量
            MOAS_ann_num = defaultdict(int)  # 其他AS重复宣告
            ann_shorter = defaultdict(dd)  # 路径变短的数量
            ann_longer = defaultdict(dd)  # 路径变短的数量

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
                for elem in rec:  # 能否按照时间来遍历。
                    # Get the prefix
                    if (first == True):
                        old_time = elem.time
                        first = False
                    pfx = elem.fields["prefix"]
                    # Get the list of ASes in the AS path
                    # print(elem)
                    if (elem.type == 'A'):
                        ases = elem.fields["as-path"].split(" ")

                        len_path = len(ases)  # 计算该长度

                        if len_path > 0:
                            # Get the origin ASn (rightmost)
                            origin = ases[-1]

                            if (origin == VICTIM_AS):  # 如果是受害者AS的前缀，则加入集合中
                                own_ann_num[old_time] += 1
                                if pfx not in prefix_origin.keys():
                                    prefix_origin[pfx].add(origin)
                                    for father_pfx in prefix_origin.keys():  # 判断是否是第一次出现的子前缀
                                        if self.is_sub_pfx(father_pfx, pfx):
                                            new_sub_prefix_num[old_time] += 1
                                            new_sub_prefix[old_time].add(pfx)
                                            break
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
                                    if ases != new_AS_Path[peer]:  # 如果路径更改的话，计算编辑距离
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
                                # pfx是否是subprefix攻击。
                                for father_pfx in prefix_origin.keys():
                                    if self.is_sub_pfx(father_pfx, pfx):
                                        MOAS_ann_num[old_time] += 1
                                        if origin not in prefix_origin[father_pfx]:

                                            new_MOAS[old_time] += 1
                                            MOAS[old_time].add(origin)
                                            MOAS_AS[old_time][pfx].add(origin)
                                            prefix_origin[father_pfx].add(origin)
                                        break

                    elif (elem.type == 'W'):
                        if pfx in prefix_origin.keys():
                            withdraw_num[old_time] += 1
                            withdraw_unique_prefix[old_time].add(pfx)

                    if (elem.time >= (old_time + 30)):
                        if (abs(old_time - start_time) < 30):  # 这里6是解决时间误差，
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
            # print(features)
            # print(features['label_0'])
            # # print(victim_prefix)
            # # Print the list of MOAS prefix and their origin ASns
            # for pfx in prefix_origin.keys():
            #     print('victim_prefix', pfx)
            # for time in MOAS_AS:
            #     for pfx in MOAS_AS[time]:
            #         print((time, pfx, ','.join(MOAS_AS[time][pfx])))
            # for time in MPL:
            #     print('MPL', MPL[time])
            # for time in PL:
            #     print('path_legth_num', PL[time])
            # for time in diff_AS_Path:
            #     print('edit_distance', diff_AS_Path[time])
            all_features = all_features.append((features))
            features.to_csv('../datasets_30s/features' + VICTIM_AS + '_' + str(begin_time) + '.csv')
            all_features.to_csv('../all_datasets/add_bgpstream.csv')
    def delete_file_with_MPL_0(self,path):
        import os
        datasets_path=path
        datasets_files = os.listdir(datasets_path)
        count=0
        for data_file in datasets_files:
            try:
                temp = pd.read_csv(datasets_path + data_file, index_col=0)
                if temp['MPL'].sum()<5:
                    print(datasets_path + data_file+'被移除\n')
                    os.remove(datasets_path + data_file)
                    count+=1
            except:
                print(datasets_path + data_file)
        print('共移除了{:}文件数量'.format(count))
extractor=Feature_Extractor()
extractor.get_his_info()
# extractor.delete_file_with_MPL_0('../datasets_30s/')