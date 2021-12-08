# ## Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def test_date(x):
    moth_dic = {"Jan" : "1", "Feb" : "2",
                "Mar" : "3", "Apr" : "4",
                "May" : "5", "Jun" : "6",
                "Jul" : "7", "Aug" : "8",
                "Sep" : "9", "Oct" : "10",
                "Nov" : "11", "Dec" : "12"}
    x = x.split(' ')[3] + "-" + moth_dic[x.split(' ')[1]] + "-" + x.split(' ')[2] + " " + x.split(' ')[4]
    return x

def data_imputation_method(left_join_df, target):
    test = left_join_df[target]
    actual_missing = np.where(np.isnan(test))[0]
    minGap = 90
    motifSize = 800
    x = test
    window_size = motifSize
    w = observationWindow = 10
    data = x
    data_interp = test.interpolate(method = 'linear', limit_direction='both')
    decompfreq = 4000
    try:
        decomposition = seasonal_decompose(data_interp, model='additive', period = decompfreq)
        data_seasonality = decomposition.seasonal
        data_trend = decomposition.trend
        npwhere = np.where(np.isnan(data_trend))[0]
        data_irregular = decomposition.resid
        data_seansonality_irregular = data_seasonality + data_irregular
        data_trend = data_trend.interpolate(method = 'linear', limit_direction='both')
        imputeddata = data_seansonality_irregular.interpolate(method = 'linear', limit_direction='both')
        acc1 = acc2 = np.array(range(1,w+1))*np.nan
        s_e =  np.array(range(1,w+1))
        temp1 = 0
        temp2 = 0
        intern = 1
        big_size = minGap
        big_win = []
        for i in range(len(actual_missing)-1):
            if actual_missing[i] + 1 == actual_missing[i+1]:
                temp1 += 1
            else:
                temp1 = temp2 = 0
            if (temp1 > big_size) & (temp2 == 0):
                big_win.append(actual_missing[i] - big_size)
                temp2 = 1
                intern += 1
        option_list = []
        for inter in range(len(big_win)):
            indexstart = indexend = iq = big_win[inter]
            acc1 = acc2 = np.array(range(1,w+1))*np.nan
            try:
                while(np.isnan(data[indexend])): # Identify actual missing gap  
                    indexend += 1
            except:
                indexend = indexend - 1
            indexgap = indexend - indexstart
#             print(big_win[inter])
            for temp_iq in range(w):
                threshold_set = 100
                temp_iq_index = int(threshold_set * temp_iq)
                try:
                    if (indexstart - temp_iq_index - (2 * window_size) - indexgap >= 1) & (indexend+window_size < len(data)):
                        a1 = imputeddata[(indexstart - window_size) : (indexstart)]
                        b1 = imputeddata[(indexstart - int(temp_iq_index) - (2*window_size) - indexgap) : (indexstart - window_size - int(temp_iq_index) - indexgap)]
                        acc1[temp_iq] = mean_absolute_error(np.array(a1), np.array(b1))
                        a2 = imputeddata[(indexend) : (indexend+window_size)] 
                        b2 = imputeddata[(indexstart-temp_iq_index - window_size) : (indexstart-temp_iq_index)]
                        acc2[temp_iq] = mean_absolute_error(np.array(a2), np.array(b2))

                        option_list.append("upper")
                        option = "both"
                    elif (indexstart + temp_iq_index + (2 * window_size) + indexgap >= 1) & (indexstart-window_size -1 >= 1) & (indexend + temp_iq_index + window_size <= len(data)):
                        a1 = imputeddata[(indexend) : (indexend + window_size)]
                        b1 = imputeddata[(indexend + int(temp_iq_index) + (window_size) + indexgap) : (indexend + 2*window_size + int(temp_iq_index) + indexgap)]
                        acc1[temp_iq] = mean_absolute_error(np.array(a1) , np.array(b1))
                        acc2 = acc1
                        option = "both"
                        option_list.append("lower")
                    else:
                        option = "Linear"
                except:
                    print("Check Initial NA point has past window")

            if option == "head":
                acc = acc1
            elif option == "tail":
                acc = acc2
            elif option == "both":
                acc = acc1 + acc2

            if option != "Linear":
                bestindx = s_e[np.where(acc == np.sort(acc)[0])]-1
                if option_list[bestindx[0]] == "upper":
                    try:
                        imputeddata[iq : (iq + indexgap)] = imputeddata[(iq - bestindx[0] * int(threshold_set) - indexgap - window_size) : (iq - bestindx[0] * int(threshold_set) - window_size)]
                    except:
                        print('linear')

                elif option_list[bestindx[0]] == "lower":
                    try:
                        imputeddata[iq : (iq + indexgap)] = imputeddata[(iq + indexgap + bestindx[0] * int(threshold_set) + window_size) : (iq + indexgap + bestindx[0] * int(threshold_set) + window_size + indexgap)]
                    except:
                        print('linear')

            data[iq : (iq + indexgap)] = imputeddata[iq : (iq + indexgap)] + data_trend[iq : (iq + indexgap)]        
            data[npwhere] = test[npwhere]
    except:
        print("imputation")
    data = data.interpolate(method = 'linear', limit_direction='both')        
    return data, test

def find_missing_gap(folder_open, file_name_to_imputation):
    target_folder_dir =os.path.abspath(folder_open)
    file_error_list = []  
    df = pd.read_csv(target_folder_dir +'\\' + file_name_to_imputation)
    
    ## Create directiory and save data
    current_time = datetime.date.today().strftime("%Y-%m-%d")
    who_create = '[completed]'
    folder_save_name = folder_open + who_create + current_time

    if not os.path.exists(folder_save_name):
        os.mkdir(folder_save_name)
        print("successfully", folder_save_name, "CREATED")
        
    try:
        print("In the midddle of interpolation")
        df["logtime"] = df['logtime'].apply(lambda x:test_date(x))
        df["logtime"] = pd.to_datetime(df['logtime']) 

    # device 내에서 key 뽑는거임
        key_list = []
        grouped_df = df.groupby('device')
        for key, group in grouped_df:
            key_list.append(key)

    # device 내에서 같은 key 끼리 묶어놓은거임
        grouped_list = []
        for i in key_list:
            n_group = grouped_df.get_group(i).sort_values(by=['logtime'])
            grouped_list.append(n_group)     

    #Create 10-second Difference Datapoint
        individual_missing_create = [] 
        for i in grouped_list:
            t_initial = i['logtime'].iloc[0]
            t_final = i['logtime'].iloc[-1]
            t_exp_initial = t_initial
            t_delta = pd.Timedelta(pd.offsets.Second(10))
            # create 10-second difference dataset
            new_time_set =[t_exp_initial]
            while t_exp_initial <= t_final:
                t_exp_initial += t_delta
                new_time_set.append(t_exp_initial)
            df_10sec_diff= pd.DataFrame(new_time_set, columns=['logtime']) 

    # Eliminate the duplicated rows in Original Data
            df_original = i
            a= df_original.shape[0]
            df_original = df_original.drop_duplicates('logtime')
            n_of_duplicated_in_original = a- df_original.shape[0]

    # Method for synching the original data with 10-second data
            def original_time_change(x, df_original):
                df_original_copy = df_original.copy()
                df_original_copy['logtime']= df_original_copy['logtime'] + pd.Timedelta(pd.offsets.Second(x))
                return df_original_copy

    # first, in one dataframe, concatenate all the forward-filled, backward-filled data
            list_for_loop = [0,-1,-2,-3,-4,1,2,3,5]
            data_table_added = pd.DataFrame([]) # Data accumulated in here
            n_in_data_new = [] # Number of data in each list_for_loop
            n_in_data_new_cumulative = [] # cumulatived number of each list_for_loop
            percent_recovered = [] # percent recoverd 
            n_of_duplicated = [] # number of duplicated data
            n_of_duplicated_cum = [] # number of duplicated data (cumulatived)

            for j in list_for_loop:
                data_aft =(pd.merge(df_10sec_diff, original_time_change(j, df_original), on='logtime', how='inner')) # inner join
                data_table_added = pd.concat([data_table_added, data_aft])
                n_in_data_new.append(data_aft.shape[0])
                n_in_data_new_cumulative.append(sum(n_in_data_new))
                percent_recovered.append(n_in_data_new_cumulative[-1] / df_10sec_diff.shape[0])
                n_of_duplicated.append(data_aft.shape[0]- data_aft.drop_duplicates('logtime').shape[0])
                n_of_duplicated_cum.append(data_table_added.shape[0]- data_table_added.drop_duplicates('logtime').shape[0])

        # Eliminate the duplicated rows from data-cumulated in one dataframe and execute left-joined method
            data_table_added = data_table_added.drop_duplicates('logtime', keep='last')
            left_join_df= pd.merge(df_10sec_diff, data_table_added, on='logtime', how='left')
            left_join_df = left_join_df.reset_index(drop=True)

            last_column_dataset = ["Power", "IR", "소음", "조도", "진동", "전력", "전류", "전압"]
            data_set_eva = False
            on_off = False
            
            for lol in last_column_dataset:
                if lol in file_name_to_imputation:
                    data_set_eva = True
                    
            if "On-Off" in file_name_to_imputation:
                on_off = True

            if (on_off == False) and (data_set_eva == True) and ((left_join_df.iloc[: , -1:]).isnull().sum() * 100 / len(left_join_df) <= 70).bool() :
                target = [left_join_df.columns[-1]]
                for col in left_join_df.columns[:-1].tolist():
                    left_join_df[col] = left_join_df[col].fillna(method='pad')
                left_join_df[target[0]][left_join_df[target[0]] < 0] = 0
                data, test = data_imputation_method(left_join_df, target[0])
                left_join_df[target[0]] = data
                left_join_df[target[0]][left_join_df[target[0]] < 0] = 0
                individual_missing_create.append(left_join_df)
            elif (on_off == False):
                if (((left_join_df.iloc[: , -1:]).isnull().sum() * 100 / len(left_join_df)) <= 70).bool():
                    target = left_join_df.columns[-5:].tolist()
                    for col in left_join_df.columns[:-5].tolist():
                        left_join_df[col] = left_join_df[col].fillna(method='pad')
                    for k in target:
                        left_join_df[k][left_join_df[k] < 0] = 0
                        data, test = data_imputation_method(left_join_df, k)
                        left_join_df[k] = data
                        left_join_df[k][left_join_df[k] < 0] = 0
                    individual_missing_create.append(left_join_df)
            else:
                for col in left_join_df.columns[:-1].tolist():
                    left_join_df[col] = left_join_df[col].fillna(method='pad')
                individual_missing_create.append(left_join_df)

        concating = individual_missing_create[0]
        for j in individual_missing_create[1:]:
            concating = pd.concat([concating, j])

        sorted_by_logtime = concating.sort_values(by=['logtime'])         
        sorted_by_logtime = sorted_by_logtime.reset_index(drop=True)

        file_name_without_dot = file_name_to_imputation.split(".", 1)
        file_name_without_dot = file_name_without_dot[0] + "_[completed]"
        new_dir_left_join_df = os.path.join(os.path.abspath(folder_save_name), file_name_without_dot)
        for_csv =new_dir_left_join_df + ".csv"
        sorted_by_logtime.to_csv(for_csv, index=False)
        second_file_name = new_dir_left_join_df + "-result" + ".xlsx"
        print("Done interpolation")
    except:
        file_error_list.append(file_name_to_imputation)



if __name__ == "__main__":
    
    # 복원이 필요한 데이터가 저장되어있는 폴더 지정
    folder_open = 'Raw_Data'
    
    # 위 폴더 내에서 복원할 csv 파일들을 list 안에 넣기
    file_list = ['실내_IR_센서_데이터_11월.csv', '실내_조도_데이터_11월.csv']
    for file_name_to_imputation in file_list:
        find_missing_gap(folder_open, file_name_to_imputation)
