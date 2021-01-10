# coding = 'utf-8'
import numpy as np
import pandas as pd
from collections import defaultdict


def target_mean_v1(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        groupby_result = data[data.index != i].groupby([x_name], as_index=False).agg(['mean', 'count'])
        result[i] = groupby_result.loc[groupby_result.index == data.loc[i, x_name], (y_name, 'mean')]
    return result


def target_mean_v2(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result


def target_mean_v3(y_np, x_np):
    length = y_np.shape[0]
    result = np.zeros(length)
    value_dict = defaultdict(int)
    count_dict = defaultdict(int)
    for i in range(length):
        value_dict[x_np[i]] += y_np[i]
        count_dict[x_np[i]] += 1
    for i in range(length):
        result[i] = (value_dict[x_np[i]] - y_np[i]) / (count_dict[x_np[i]] - 1)
    return result


if __name__ == '__main__':
    y = np.random.randint(2, size=(500, 1))
    x = np.random.randint(10, size=(500, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
    from line_profiler import LineProfiler

    lp1 = LineProfiler()
    lp1_wrapper = lp1(target_mean_v1)
    lp1_wrapper(data, 'y', 'x')
    lp1.print_stats()

    lp2 = LineProfiler()
    lp2_wrapper = lp2(target_mean_v2)
    lp2_wrapper(data, 'y', 'x')
    lp2.print_stats()

    lp3 = LineProfiler()
    lp3_wrapper = lp3(target_mean_v3)
    lp3_wrapper(data['y'].values, data['x'].values)
    lp3.print_stats()

    result_1 = target_mean_v1(data, 'y', 'x')
    result_2 = target_mean_v2(data, 'y', 'x')
    result_3 = target_mean_v3(data['y'].values, data['x'].values)

    diff = np.linalg.norm(result_1 - result_3)
    print(diff)
