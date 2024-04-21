"""
用于处理电池数据的一份代码：
chatgpt：
这段代码执行了一些电压-容量曲线数据处理的操作。让我一步步来解释：

1. `generate_composition_vector(std_data)`: 这个函数用于生成化学组成向量。它接受一个包含标准化数据的字典作为输入，并返回一个长度为16的向量，表示该化学组成中每种元素的数量。

2. `regenerate_capacity(dQdV, dV, c_dc)`: 这个函数用于根据电导率曲线和电压变化来重新生成电容曲线。它接受电导率曲线（dQ/dV）、电压变化（dV）和充电/放电（charge/discharge）标识作为输入，并返回重新生成的电容曲线。

3. `readStdJSON`类：这个类用于读取标准化的JSON数据文件，并提供一些方法来获取充电和放电信息以及生成循环系列数据。

4. `voltageProfileProcessor`类：这个类用于处理电压-容量曲线数据。它接受电压和容量曲线数据以及充电/放电标识作为输入，并提供方法来生成电导率曲线和重新生成电容曲线。

这些函数和类一起构成了一个电池数据处理流水线，用于处理标准化的JSON数据文件并从中提取电压-容量曲线数据。
"""


import json
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import PchipInterpolator
from scipy import signal

compositions = ['Li', 'Mg', 'Al',
                'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
                'Zr', 'Nb', 'Mo', 'W',
                'O', 'F']


def generate_composition_vector(std_data):
    comp_vec = np.zeros(len(compositions))
    for key in std_data['composition'].keys():
        formula_unit = std_data['composition'][key]
        specie_index = compositions.index(key)
        #         print(specie_index)

        comp_vec[specie_index] = formula_unit / 2

    return comp_vec


def regenerate_capacity(dQdV, dV, c_dc):
    if c_dc == 'discharge':
        redQ = -np.flip(dQdV * dV)
        print(np.average(dV))
        print(sum(redQ))
        reQ = np.zeros(len(redQ))
        for i in range(len(redQ)):
            reQ[i] = np.sum(redQ[:i])

        reQ = np.flip(reQ)
        return reQ

    elif c_dc == 'charge':
        redQ = dQdV * dV
        reQ = np.zeros(len(redQ))
        for i in range(len(redQ)):
            reQ[i] = np.sum(redQ[:i])
        return reQ
    else:
        raise Exception("Error: Please identify the charge/discharge information")


class readStdJSON:
    def __init__(self, file_name):
        with open(file_name, 'r') as readfile:
            self.std_data = json.load(readfile)

        self.discharge_info = self.std_data['discharge_info']
        self.charge_info = self.std_data['charge_info']

    def discharge_info(self):
        return self.discharge_info

    def charge_info(self):
        return self.charge_info

    def generate_cycling_series(self, cycle_index, c_dc):
        """
        :param cycle_index: index of cyling, not always equal to cycling number
        :return:
        """
        if c_dc == 'discharge':
            cycle_number = self.discharge_info[cycle_index][0]
            voltage_series = self.discharge_info[cycle_index][1]['Voltage(V)']
            capacity_series = self.discharge_info[cycle_index][1]['Capacity(mAh/g)']

        elif c_dc == 'charge':
            cycle_number = self.charge_info[cycle_index][0]
            voltage_series = self.charge_info[cycle_index][1]['Voltage(V)']
            capacity_series = self.charge_info[cycle_index][1]['Capacity(mAh/g)']

        else:
            raise Exception("Error: Please identify the charge/discharge information")

        return cycle_number, voltage_series, capacity_series

    def update(self, update_dict):
        self.std_data.update(update_dict)


class voltageProfileProcessor:
    def __init__(self, voltage_series, capacity_series, c_dc, cycle_number, num_points=None):
        self.c_dc = c_dc
        self.cycle_number = cycle_number
        indices = np.argsort(voltage_series)
        exp_indices = indices[3:-3]
        # sampling_step = max(1, len(voltage_series) // 100)



        # N = max(1, len(voltage_series)//20) # the average window of convolution
        # voltage_conv = np.convolve(voltage_series, np.ones(N)/N, mode= 'full')[(N+3):-(N+3):1]
        # capacity_conv = np.convolve(capacity_series, np.ones(N)/N, mode= 'full')[(N+3):-(N+3):1]


        self.V_exp = np.array(voltage_series)[exp_indices]
        self.Q_exp = np.array(capacity_series)[exp_indices]

        self.voltage_series = np.array(voltage_series)[exp_indices]#  voltage_conv  # voltage_conv #[V_indices]

        self.capacity_series = np.array(capacity_series)[exp_indices] # capacity_conv # [Q_indices]



        if num_points is None:
            self.num_points = len(voltage_series)
        else:
            self.num_points = num_points

    def generate_dQdV_unispl(self, num_points):

        try:
            num_points = num_points # max(50,  int(len(self.voltage_series)* np.exp(-(self.cycle_number-1) / 30)))

            xs = np.linspace(np.min(self.voltage_series), np.max(self.voltage_series), num_points)

            unispl = UnivariateSpline(self.voltage_series, self.capacity_series, s = 10)
            unispl_derivative = unispl.derivative()

            self.V_std = xs
            self.Q_std = unispl(xs)
            self.dQdV_std = unispl_derivative(xs)

            self.Q_regenerate = np.zeros(len(self.Q_std))
        except:
            # print(xs)
            # print(self.capacity_series)
            self.V_std = np.nan
            self.Q_std = np.nan


    def generate_dQdV_unispl_withV(self, V_std):

        try:
            unispl = UnivariateSpline(self.voltage_series, self.capacity_series, s = 10)
            unispl_derivative = unispl.derivative()

            self.V_std = V_std
            self.Q_std = unispl(self.V_std)
            self.dQdV_std = unispl_derivative(self.V_std)

            self.Q_regenerate = np.zeros(len(self.Q_std))
        except:
            # print(xs)
            # print(self.capacity_series)
            self.V_std = np.nan
            self.Q_std = np.nan
