import pandas as pd
pd.options.mode.chained_assignment = 'warn'
#Can be set to None, "warn", or "raise". "warn" is the default.
import copy
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import uniform_filter1d


# self.conditions[] - useful for accessing 
R = 8.314 #J/mol/K (universal gas constant)
P_atm = 1.01325E5 #Atmopsheric pressure [Pa]
MFCconversion = 1.66667e-8*P_atm/(R*273.15) # used to convert MFC max values from mL/min to mol/s as 1mL/min = 1.6667E-8 m3/s then ideal gas law for mol/s
ExperimentalSetup = {'MS_header_row':26,
'stop_pickup_percentage' : 0.04, # used in the RS joining section of the code
'R': 8.314,
'P_atm' : 1.01325E5,
'P_exp' : 1.01325E5,
'T_exp' : 313.15,
'bed_mass' : 0.25E-3, # bed mass is in kg
'H2O_ratio' : 0,
'P_sat_H2O_bath' : 0,
'filter_window' : 4,
'initial_sweep' : 0,
'smoothing_start' : {'CO2':0, 'H2O':0, 'N2':0, 'He':0}, 
'water_outliers' : False,
'water_outlier_value' : 2.5E-5, 
'spike_reduction' :False,
'spike_reduction_parameter' : 1.1, 
'extra_normalisation' : False, 
'integration_end' : 1100,
'MS_start':0,
'Coriolis_start':14,
'LowConcCo2' : True,
'breakthrough_start': 3600*4 + 60*42 + 5,
'breakthrough_end': 3600*5 + 60*7 + 30,
'MaxInletFlows' : {'N2':100*MFCconversion, 'He':142*MFCconversion, 'CO2':{True:MFCconversion*2, False: MFCconversion*100}, 'O2':1, 'H2O':1.02},
'RS' : {'N2':1, 'He':1.68305, 'CO2':0.7126, 'O2':1, 'H2O':1.02},
'Constants':{'P_atm':1.01325E5,
            'max_outlet' : 100 * 1.66667e-8,
            'T_exp':313,
            'Background MS':{'N2':1.54E-11, 'He':1.40E-8, 'CO2':3.74E-13, 'O2':1.86E-10, 'H2O':3.50E-11},
            'Mw':{'N2':28E-3, 'He':4E-3, 'CO2':44E-3, 'O2':32E-3, 'H2O':18E-3} }}



# 2_1 differs from 1 as 2_1 can find RS for each run by itself


class experiment_analysis():

    """
    Aims to simplify the analysis of experimental data.
    This will be done through a jupyter notebook in which this file will act as a black box.
    People wanting data will simply have to run the necessary cells on the notebook and results should be outputted.
    """

    def __init__(
        self,
        coriolis_file_name,
        MS_file_name,
        conditions
    ):

        """ Initialise the class
        Args:
            coriolis_file_name: path and filename of coriolis data
            breakthrough_file_name: path and filename of breakthrough data
            conditions: the setup of the experiment aswell as timings 
            RSU_used: used to test the differences between old relative MS sensitivities versus new (RSU_Uused = True means updated RS used)
            Orig: this is to test simple bug fixes (Orig = True means bug fix not used)
        """
        self.coriolis_file_name = coriolis_file_name # self.variable_name is a variable that is persistent everywhere within this class (i.e. it can be accessed within the methods)
        self.MS_file_name = MS_file_name
        self.conditions = conditions
        self.sorted_data = self.sort_data() # This is an example. When the class is initiated, the sort_data function is called, and the results are stored in self.sorted_data
        self.loading_data = self.calculate_loading(integration_end=self.conditions['integration_end'])

    def discontinuity_search(self, series_breakthrough_time, series_corrected_pressure, percentage_diff = 0.35, avg_range = 3):
        '''
        This is used to find where the MS has changed sensitivities
        This is identified by a plateau after a jump
        numpy is then used to guess where the next point should be after the jump
        The corrected pressures is then ammended by dividing by the new RS from the jump
        '''
        Found = False
        position ='None'
        i = len(series_corrected_pressure) -1 
        end_pos = 0
        while sum([round(series_corrected_pressure.loc[i] / (2*  self.conditions['stop_pickup_percentage'] * series_corrected_pressure.loc[len(series_corrected_pressure.index)-(i + 1)])) for i in range(end_pos)]) <1:
            end_pos +=1
        end_pos -= 1
        while Found == False:
            if i == end_pos + 2*avg_range + 2:
                Found = True
            if sum(series_corrected_pressure.loc[i - (2*avg_range + 2):i - (avg_range + 2)])/(avg_range * series_corrected_pressure.loc[i - (avg_range + 2)]) > 0.9:
                if abs((sum(series_corrected_pressure.loc[(i-avg_range):i])/avg_range - series_corrected_pressure.loc[i - (avg_range + 2)])/(sum(series_corrected_pressure.loc[(i-avg_range):i])/avg_range)) > percentage_diff:
                    #print(sum(series_corrected_pressure.loc[(i-avg_range):i])/avg_range - series_corrected_pressure.loc[i - (avg_range + 2)])
                    position = i - (avg_range + 1)
                    Found = True
            i -= 1
        if position != 'None':
            # using numpy to get a linear best fit for where point should go
            best_fit = np.polyfit(series_breakthrough_time.loc[position: position + avg_range] , series_corrected_pressure.loc[position: position + avg_range] , 1)
            desired_pressure = series_breakthrough_time.loc[position - 1] * best_fit[0] + best_fit[1]
            RS_new = series_corrected_pressure.loc[position - 1] / desired_pressure
        else:
            RS_new = 1
        return {'RS_new' : RS_new, 'position' : position}

    def sort_data(self):
        '''
        this takes in the raw MS and Coriolis files and joins them into a dataframe. 
        Interpolates flow readings to gain values at MS reading times
        Then chops off what you dont want ie mixing, purging etc
        Converts from partial pressures and flow to mass and molar flow.
        produces normalised and smoothed results about flow info
        final columns produced are the standard result (IAS) y(t)Q(t)/yinQin for CO2 and N2
        '''
        df_MS = pd.read_csv(self.MS_file_name, header=(self.conditions['MS_header_row']-1)).drop(['Time', 'Unnamed: 7'], axis=1) #reading the MS csv as a dataframe
        df_MS.loc[:,'Time [s]'] = pd.Series([i/1000 + self.conditions['MS_start'] for i in df_MS['ms']] , index=df_MS.index) #Adding the MS start time to the times in our dataframe
        df_FM = pd.read_csv(self.coriolis_file_name, sep=';', names = ['Time', 'CO2', 'Time 2', 'He', 'Time 3', 'N2', 'Time 4', 'Outlet']).drop(['Time 2', 'Time 3', 'Time 4'], axis=1) #reading the flow meter csv as a dataframe
        df_FM.loc[:,'Time [s]'] = pd.Series([i + self.conditions['Coriolis_start'] for i in df_FM['Time']] , index=df_FM.index) #Adding the flow meter start time to the time in this data
        #This line below is the all important step. We merge the two dataframes based on the times defined above, then we make sure its ordered in time order, and also renaming columns
        df_all = pd.merge(df_MS, df_FM, on='Time [s]',how='outer', sort=True).drop(['Time', 'ms'], axis=1).rename(columns={"Nitrogen": "N2 pressure [torr]", "Water": "H2O pressure [torr]", "Carbon dioxide": "CO2 pressure [torr]", "Oxygen": "O2 pressure [torr]", "Helium": "He pressure [torr]", "CO2": "CO2 flow [%]", "He": "He flow [%]", "N2": "N2 flow [%]", "Outlet": "Outlet flow [%]"}) 
        df_all.reset_index(drop=True, inplace=True)
        #In this for loop we are interpolating between each time point that exists for the MS data, to get the flows from the coriolis at that time
        # below we are getting from where to start interpolating - the first positions outside the breakthrough time where there is a non-NaN pp reading 
        BBTMSR = Before_BT_Real_MS_index = df_all.loc[(np.isnan(df_all['CO2 pressure [torr]']) == False) & (df_all['Time [s]'] < self.conditions['breakthrough_start']), ['Time [s]']].index[-1]
        ABTMSR = After_BT_Real_MS_index = df_all.loc[(np.isnan(df_all['CO2 pressure [torr]']) == False) & (df_all['Time [s]'] > self.conditions['breakthrough_end']), ['Time [s]']].index[0]
        #interpolating flow
        df_all.loc[np.isnan(df_all['CO2 pressure [torr]']) == False].loc[BBTMSR:ABTMSR, 'Time [s]']
        for comp in ['CO2', 'N2', 'He', 'Outlet']:
            label = comp + ' flow [%]'
            Index_of_real_flow = df_all[(np.isnan(df_all.loc[:,label]) == False)].loc[BBTMSR -10:ABTMSR +10].index
            interpolated_flow_list = list(np.interp(x = list(df_all[np.isnan(df_all['CO2 pressure [torr]']) == False].loc[BBTMSR-1:ABTMSR+1, 'Time [s]']), xp = list(df_all.loc[Index_of_real_flow, 'Time [s]']), fp = list(df_all.loc[Index_of_real_flow, label])))
            count = 0
            original_flow = copy.deepcopy(df_all.loc[:,label])
            for i in df_all[(np.isnan(df_all.loc[:,label]))].loc[BBTMSR -10:ABTMSR +10].index:
                original_flow.loc[i] = interpolated_flow_list[count]
                count += 1
            df_all.loc[:,'Interpolated ' + label] = pd.Series(original_flow, index=df_all.index)
        #Now converting the data time to the time of the breakthrough step
        df_all.loc[:,'Breakthrough time [s]'] = (df_all['Time [s]'] - self.conditions['breakthrough_start'])
        #Now deleting rows without MS values
        df_breakthrough_start = df_all.loc[(abs(df_all['CO2 pressure [torr]']) >= 0)]
        #Now we are only taking the part of the dataframe that we are interested in (ignoring drying, cooling, purging etc.)
        df_not_breakthrough = df_breakthrough_start.loc[((df_breakthrough_start['Time [s]'] < self.conditions['breakthrough_start']))]
        df_breakthrough = df_breakthrough_start.loc[((df_breakthrough_start['Time [s]'] > self.conditions['breakthrough_start']) & (df_breakthrough_start['Time [s]'] < self.conditions['breakthrough_end']))]
        df_breakthrough.reset_index(drop=True, inplace=True)
        # inserting the last value from df_not_breakthrough as time 0 into df_breakthrough
        mask = copy.deepcopy(df_not_breakthrough.loc[max(df_not_breakthrough.index)])
        mask.loc['Time [s]'], mask.loc['Breakthrough time [s]'] = self.conditions['breakthrough_start'], 0
        df_breakthrough = pd.concat([(pd.DataFrame(mask.to_dict(), index=[0])),df_breakthrough.loc[:]]).reset_index(drop=True)
        df_breakthrough.sort_values('Breakthrough time [s]', inplace=True)
        df_breakthrough.reset_index(drop=True, inplace=True)
        # drops non interpolated flow - we dont need
        df_breakthrough.drop(['CO2 flow [%]', 'He flow [%]', 'N2 flow [%]', 'Outlet flow [%]'], axis=1, inplace=True)
        AvgRange = 25
        PPs = {'PPdashCO2':0, 'PPdashHe':0, 'PPdashN2':0}
        for comp in ['CO2', 'He', 'N2']:
            Total = 0
            for i in range(len(df_breakthrough.index) - (AvgRange + 1), len(df_breakthrough.index) - 1):
                Total += df_breakthrough[comp + ' pressure [torr]'][i] - self.conditions['Constants']['Background MS'][comp]
            PPs['PPdash' + comp] = Total/AvgRange
        FlowAverages = {'Outlet_flow':0, 'CO2_flow':0, 'N2_flow':0, 'He_flow':0}
        for comp in ['Outlet', 'CO2', 'N2', 'He']:
            FlowAverages[comp + '_flow'] = sum(df_breakthrough['Interpolated '+ comp + ' flow [%]'][i] for i in range(len(df_breakthrough.index) - (AvgRange + 1), len(df_breakthrough.index) - 1))/AvgRange
        if (self.conditions['LowConcCo2'] == False) and (sum(df_breakthrough.loc[:,'Interpolated CO2 flow [%]'])/len(df_breakthrough.index) > 4):
            RSN2 =0.78
        if (self.conditions['LowConcCo2'] == False) and (sum(df_breakthrough.loc[:,'Interpolated CO2 flow [%]'])/len(df_breakthrough.index) <= 4):
            RSN2 =0.96
        else:
            RSN2 = 1
        RSCO2 = (FlowAverages['N2_flow'] * self.conditions['MaxInletFlows']['N2'] * PPs['PPdashCO2'])/(RSN2 *FlowAverages['CO2_flow'] * self.conditions['MaxInletFlows']['CO2'][self.conditions['LowConcCo2']] * PPs['PPdashN2'])
        RSHe = (FlowAverages['N2_flow'] * self.conditions['MaxInletFlows']['N2'] * PPs['PPdashHe'])/(RSN2 * FlowAverages['He_flow'] * self.conditions['MaxInletFlows']['He'] * PPs['PPdashN2'])
        print('updated code used, RS are:', RSCO2, RSN2, RSHe)
        #correcting MS pressures
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = comp + ' pressure [torr]'
            new_label = 'Corrected ' + label
            if comp == 'N2':
                corrected_list = []
                for i in range(len(df_breakthrough['Breakthrough time [s]'])): 
                    if (df_breakthrough[label][i] - self.conditions['Constants']['Background MS'][comp] - 0.114*df_breakthrough['Corrected CO2 pressure [torr]'][i]) >= 0:
                        corrected_list.append((copy.deepcopy((df_breakthrough[label][i] - self.conditions['Constants']['Background MS'][comp] - 0.114*(df_breakthrough['Corrected CO2 pressure [torr]'][i]))))/RSN2)
                    else:
                        corrected_list.append(0)
                df_breakthrough.loc[:,new_label] = pd.Series(corrected_list, index=df_breakthrough.index)
            elif comp == 'CO2':
                df_breakthrough.loc[:,new_label] = pd.Series(copy.deepcopy([(i - self.conditions['Constants']['Background MS'][comp])/RSCO2 if (i - self.conditions['Constants']['Background MS'][comp])/RSCO2 > 0 else 0 for i in df_breakthrough[label]]), index=df_breakthrough.index)
            elif comp == 'He':
                df_breakthrough.loc[:,new_label] = pd.Series(copy.deepcopy([(i - self.conditions['Constants']['Background MS'][comp])/RSHe if (i - self.conditions['Constants']['Background MS'][comp])/RSHe > 0 else 0 for i in df_breakthrough[label]]), index=df_breakthrough.index)
            else:
                df_breakthrough.loc[:,new_label] = pd.Series(copy.deepcopy([(i - self.conditions['Constants']['Background MS'][comp])/self.conditions['RS'][comp] if (i - self.conditions['Constants']['Background MS'][comp])/self.conditions['RS'][comp] > 0 else 0 for i in df_breakthrough[label]]), index=df_breakthrough.index)
        #This connects discontinuities present due to current Mass spec 
        if self.conditions['LowConcCo2'] == False:
            for i in range(2):
                for comp in ['CO2', 'N2']:
                    RS_and_pos = self.discontinuity_search(series_breakthrough_time = df_breakthrough.loc[:,('Breakthrough time [s]')] , series_corrected_pressure = df_breakthrough.loc[:,('Corrected ' +  comp + ' pressure [torr]')])
                    if (RS_and_pos['position'] != 'None') and (RS_and_pos['position'] != 0):
                        df_breakthrough.loc[:RS_and_pos['position'] - 1,'Corrected ' + comp + ' pressure [torr]'] /= RS_and_pos['RS_new'] 
                        print(comp + ' discontinuity at time ', df_breakthrough.loc[RS_and_pos['position'], 'Time [s]'], ' where an intermediate RS of ', RS_and_pos['RS_new'], ' divided all previous points')   
        #Calculating mole fractions in the mass spectrometer in the below for loop
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = 'Corrected ' +comp + ' pressure [torr]'
            new_label = 'True MS ' +comp + ' mole fraction [-]'
            molar_list = []
            for i in range(len(df_breakthrough['Breakthrough time [s]'])):
                molar_list.append(df_breakthrough[label][i]/sum(df_breakthrough['Corrected '+ j +' pressure [torr]'][i] for j in ['CO2', 'N2', 'He', 'H2O', 'O2']))
            df_breakthrough.loc[:,new_label] = pd.Series(molar_list, index=df_breakthrough.index)
        df_breakthrough.loc[:,'Fake Outlet average molecular weight [kg/mol]'] = pd.Series([sum(df_breakthrough['True MS ' + j + ' mole fraction [-]'][i]*self.conditions['Constants']['Mw'][j] for j in ['CO2', 'N2', 'He', 'H2O', 'O2']) for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            # mass flows for each component is calculated
            label = 'True MS ' +comp + ' mole fraction [-]'
            new_label = 'True ' +comp + ' mass flow [kg/s]'
            df_breakthrough.loc[:,new_label] = pd.Series([(self.conditions['Constants']['max_outlet'] * df_breakthrough['Interpolated Outlet flow [%]'][i]/100) * df_breakthrough[label][i] * self.conditions['Constants']['Mw'][comp]/df_breakthrough['Fake Outlet average molecular weight [kg/mol]'][i] for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)  
        #Now converting the mass flow to molar flow values
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = 'True '+comp + ' mass flow [kg/s]'
            new_label = 'True ' + comp + ' molar flow [mol/s]'
            df_breakthrough.loc[:,new_label] = pd.Series(df_breakthrough[label] / self.conditions['Constants']['Mw'][comp], index=df_breakthrough.index)
        #Now calculating the molar flow rate of the helium through the bypass from the helium mass flow meter
        df_breakthrough.loc[:,'He bypass flow [mol/s]'] = pd.Series([df_breakthrough['Interpolated He flow [%]'][i] * self.conditions['MaxInletFlows']['He']/100 for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
        #Now calculating the molar flow in the outlet below. We calculate the helium flow (there is helium initially in the reactor left over from the drying step), from the helium flow in the MS minus the helium flow from the bypass
        for comp in ['He', 'CO2', 'N2', 'H2O', 'O2']:
            label = 'True '+comp + ' molar flow [mol/s]'
            new_label =  comp + ' molar flow [mol/s]'
            if comp != 'He':
                if self.conditions['water_outliers']==True and comp == 'H2O':
                    df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[label][i] if df_breakthrough[label][i] < self.conditions['water_outlier_value'] else df_breakthrough[label][i-1] for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)  
                else:
                    df_breakthrough.loc[:,new_label] = pd.Series(df_breakthrough.loc[:,label], index=df_breakthrough.index) 
            else:
                df_breakthrough.loc[:,new_label] = pd.Series(df_breakthrough.loc[:,label], index=df_breakthrough.index)   
        # #Summing these molar flows to a total molar flow
        df_breakthrough.loc[:,'Total molar flow [mol/s]'] = pd.Series([sum(df_breakthrough[j + ' molar flow [mol/s]'][i] for j in ['He', 'N2', 'CO2', 'H2O', 'O2']) for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)     
        #below removes the possibilty of /0 error as it deletes rows where total molar flow is 0 
        df_breakthrough = df_breakthrough.drop(index = list(df_breakthrough.loc[df_breakthrough['Total molar flow [mol/s]'] <= 0].index))
        df_breakthrough = df_breakthrough.reset_index(drop=True)
        # setting the first however many desired values to 0 
        if self.conditions['initial_sweep'] > 0:
            df_breakthrough.loc[0:self.conditions['initial_sweep'],'CO2 molar flow [mol/s]'] = 0
        #Calculating mole fractions from these molar flows
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = comp + ' molar flow [mol/s]'
            new_label =  comp + ' mole fraction [-]'
            df_breakthrough.loc[:,new_label] = pd.Series(df_breakthrough[label] / df_breakthrough['Total molar flow [mol/s]'], index=df_breakthrough.index)    
        #Calculating concentrations from these mole fractions 
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = comp + ' mole fraction [-]'
            new_label = comp + ' concentration [mol/m3]'
            df_breakthrough.loc[:,new_label] = pd.Series(df_breakthrough[label] * self.conditions['P_exp']/(self.conditions['R']*self.conditions['T_exp']), index=df_breakthrough.index)
        #Note that He inlet flow is 0 here
        #Calculating the inlet flow rates for each component:
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = 'Interpolated '+ comp + ' flow [%]'
            new_label = comp + ' inlet flow [mol/s]'
            if comp != 'CO2' and comp != 'N2' and comp!= 'H2O':
                df_breakthrough.loc[:,new_label] = pd.Series([0]*len(df_breakthrough['Breakthrough time [s]']), index=df_breakthrough.index)
            elif comp == 'CO2':
                df_breakthrough.loc[:,new_label] = pd.Series([self.conditions['MaxInletFlows']['CO2'][self.conditions['LowConcCo2']]*df_breakthrough[label][i]/100 for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
            elif comp == 'N2':
                df_breakthrough.loc[:,new_label] = pd.Series([self.conditions['MaxInletFlows']['N2']*df_breakthrough[label][i]/100 for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
            elif comp == 'H2O':
                df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough['N2 inlet flow [mol/s]'][i]*self.conditions['H2O_ratio']/(1-self.conditions['H2O_ratio']) for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
        #Now calculating inlet mole fractions
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = comp + ' inlet flow [mol/s]'
            new_label = comp + ' inlet mole fraction [-]'
            mole_frac_list = [df_breakthrough[label][i]/sum(df_breakthrough[j + ' inlet flow [mol/s]'][i] for j in ['CO2', 'N2', 'He', 'H2O', 'O2']) if sum(df_breakthrough[j + ' inlet flow [mol/s]'][i] for j in ['CO2', 'N2', 'He', 'H2O', 'O2']) != 0 else np.NAN for i in range(len(df_breakthrough['Breakthrough time [s]']))]
            df_breakthrough.loc[:,new_label] = pd.Series(mole_frac_list, index=df_breakthrough.index)
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            new_label = comp + ' inlet mole fraction [-]'
            df_breakthrough = df_breakthrough.drop(index = df_breakthrough[np.isnan(df_breakthrough.loc[:,new_label])].index)
            df_breakthrough = df_breakthrough.reset_index(drop=True)
        #And calculating inlet concentrations
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = comp + ' inlet mole fraction [-]'
            new_label = comp + ' inlet concentration [mol/m3]'
            df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[label][i] * self.conditions['P_exp']/(self.conditions['R']*self.conditions['T_exp']) for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
        # This code normalises the different measurements (listed in first for statement)
        for measurement in ['mole fraction [-]', 'concentration [mol/m3]', 'molar flow [mol/s]']:
            for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
                outlet_label = comp + ' ' + measurement
                if measurement != 'molar flow [mol/s]':
                    inlet_label = comp + ' inlet ' + measurement
                else:
                    inlet_label = comp + ' inlet flow [mol/s]'
                new_label = 'Normalised ' + outlet_label
                df_breakthrough.loc[:,new_label] = pd.Series([df_breakthrough[outlet_label][i]/df_breakthrough[inlet_label][i] if df_breakthrough[inlet_label][i] > 0 else 0 for i in range(len(df_breakthrough['Breakthrough time [s]']))], index=df_breakthrough.index)
        #creating ordered list for below normalisation loops to access columns easier
        order = ['Breakthrough time [s]']
        for result in [' inlet flow [mol/s]', ' inlet mole fraction [-]', ' inlet concentration [mol/m3]', 
                        ' molar flow [mol/s]', ' mole fraction [-]', ' concentration [mol/m3]',
                        'Nmolar flow [mol/s]', 'Nmole fraction [-]', 'Nconcentration [mol/m3]']:
            if (result != 'Nmolar flow [mol/s]') & (result != 'Nmole fraction [-]') & (result != 'Nconcentration [mol/m3]'):
                for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
                    order.append(comp + result)
            else:
                for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
                    order.append('Normalised ' + comp + ' ' + result[1:]) 
        #Finding which row in the dataframe the smoothing should start
        startlist = {}
        for comp in ['CO2', 'N2', 'He', 'H2O']:
            for i in range(1,len(df_breakthrough['Breakthrough time [s]'])):
                index = i
                if df_breakthrough['Breakthrough time [s]'][i] > self.conditions['smoothing_start'][comp]:
                    break
            startlist[comp] = index
        #Here we smooth the data
        for i in range(1,len(order)):
            label = order[i]
            new_label = 'Smoothed ' + label
            new_list = []
            for comp in ['CO2', 'N2', 'He', 'H2O']:
                if comp in label:
                    component = comp
            filtered_data = uniform_filter1d(pd.Series(df_breakthrough[label],index=df_breakthrough.index)[startlist[component]:], size=self.conditions['filter_window'])
            for comp in ['CO2', 'N2', 'He', 'H2O']:
                if comp in label: 
                    for q in range(startlist[comp]):
                        new_list.append(df_breakthrough[label][q])
                        f=q
                    for q in range(len(filtered_data)):
                        new_list.append(filtered_data[q])
                    df_breakthrough.loc[:,new_label] = pd.Series(new_list)
        #We can normalise all the values to the final value now for the smoothed data if we like
        if self.conditions['extra_normalisation'] == True:
            for comp in ['CO2', 'N2', 'He', 'H2O']:
                label = 'Smoothed Normalised ' + comp + ' molar flow [mol/s]'
                series = pd.Series([i/df_breakthrough[label].iloc[-1] for i in df_breakthrough[label]], index=df_breakthrough.index)
                df_breakthrough.loc[:,'Smoothed renormalised ' + comp + ' molar flow [mol/s]'] = series
        # this is to find the standard form of results as reccomended by IAS lecture which is y(t) * Q(t) / (yin * Qin)
        # where Q are the total volume flow rate in mL/min
        # where y is the molar fraction (He is not neglected)
        # we are finding the yinQin column for N2 and CO2, this column is equal to the inlet volumetric flow for the respective compound as the pressures and temps are the same for all MFCs so molar fraction is equal to volumetric fraction 
        # yinQin is in mL/min 
        MpS_to_mLpS = 273 * 8.314 * 6E7 / 1.01325E5 # used to convert from mol/s to mL/min
        df_breakthrough.loc[:, 'Q(t)'] = sum(df_breakthrough.loc[:, 'True ' + comp + ' molar flow [mol/s]'] for comp in ['CO2', 'N2', 'He']) * MpS_to_mLpS
        for comp in ['CO2', 'N2']:
            df_breakthrough.loc[:,'yinQin ' + comp] = df_breakthrough.loc[:, comp + ' inlet flow [mol/s]'] * MpS_to_mLpS
            # we can find Qt by the ideal gas law - average is used from CO2 and N2 (not He as it can be tricky)
            df_breakthrough.loc[:, 'y(t) ' + comp] = df_breakthrough.loc[:, 'True ' + comp + ' molar flow [mol/s]']/sum(df_breakthrough.loc[:, 'True ' + comp2 + ' molar flow [mol/s]'] for comp2 in ['CO2', 'N2', 'He'])
            for measurement in ['yinQin ', 'y(t) ']:
                df_breakthrough.loc[:, measurement + comp] = pd.Series(uniform_filter1d(df_breakthrough.loc[:, measurement + comp], size = self.conditions['filter_window']), index = df_breakthrough.index)
            df_breakthrough.loc[:, 'y(t)Q(t)/yinQin ' + comp] = df_breakthrough.loc[:, 'y(t) ' + comp] * df_breakthrough.loc[:, 'Q(t)'] / df_breakthrough.loc[:,'yinQin ' + comp]
            df_breakthrough.loc[:, 'y(t)Q(t)/yinQin ' + comp] = uniform_filter1d(df_breakthrough.loc[:, 'y(t)Q(t)/yinQin ' + comp], size = self.conditions['filter_window'])
        return df_breakthrough


    def plot(self, Columns_plotted = 'y(t)Q(t)/yinQin CO2', Plot_Title = 'y(t)Q(t)/yinQin CO2 at '):
        '''
        Simple quick plotting function. 
        By default it will plot the standard results y(t)Q(t)/yinQin for CO2
        Can change what is plotted however by specifying in the Columns_plotted section, this can take lists of what you want to plot.
        argument to specify title is present too 'Plot_Title
        '''
        fig, ax = plt.subplots(figsize=(16,5))
        if Columns_plotted == 'y(t)Q(t)/yinQin CO2':
            ax.scatter(self.sorted_data['Breakthrough time [s]'], self.sorted_data[Columns_plotted], color='tab:blue', s=10, alpha=0.5)
            ax.plot(self.sorted_data['Breakthrough time [s]'], self.sorted_data[Columns_plotted], color='tab:orange', linestyle='-')
            ax.plot(self.sorted_data['Breakthrough time [s]'], [1] * len(self.sorted_data['Breakthrough time [s]']), color = 'tab:grey', linestyle='-.')
            plt.title(label = (Plot_Title, (10* round(sum(self.sorted_data['yinQin CO2'])/len(self.sorted_data), 1)), ' percent'))
        else:
            if type(Columns_plotted) == list:
                for i in Columns_plotted:
                    ax.scatter(self.sorted_data['Breakthrough time [s]'], self.sorted_data[i], color='tab:blue', s=10, alpha=0.5)
                    ax.plot(self.sorted_data['Breakthrough time [s]'], self.sorted_data[i], color='tab:orange', linestyle='-.')
            else:
                    ax.scatter(self.sorted_data['Breakthrough time [s]'], self.sorted_data[Columns_plotted], color='tab:blue', s=10, alpha=0.5)
                    ax.plot(self.sorted_data['Breakthrough time [s]'], self.sorted_data[Columns_plotted], color='tab:orange', linestyle='-')
            plt.title(label = (Plot_Title, (10* round(sum(self.sorted_data['yinQin CO2'])/len(self.sorted_data), 1)), ' percent'))
        plt.show()


    def export_data(self, fileName):
        '''
        Exports any dataframe to csv, must specify file name and dataframe
        Working as required
        '''
        self.sorted_data.to_csv(fileName + '.csv', index=False)
        pass

    def calculate_loading(self,integration_end =1E6, density=1):
        '''
        Calculates the loading of the object.
        Note to find true loading for a sample take the value of the sample from this and subtract the blank (dead volume) loading value.
        As of now this only works for analysing breakthrough curves (NOT desorption surves). 
        Return the loading capacities volume and mass based adn the selectivity of CO2/N2
        '''
        #Now our loading calculation
        bed_length, bed_diameter, bed_mass = 0.30, 0.0067, 250E-3 #mass in kg and dimensions in m
        T_exp = self.conditions['T_exp']
        bed_porosity=0.4
        P_atm = 1.01325E5 #[Pa]
        P_exp = P_atm #[Pa]        
        R = 8.314
        bed_area = (np.pi*bed_diameter**2)/4
        bed_volume = bed_area*bed_length
        #Calculating the bed density.
        density = bed_mass/bed_volume
        pellet_density = density/(1-bed_porosity) 
        #Deleting all the data after the integration end time
        df = self.sorted_data.loc[((self.sorted_data.loc[:,'Breakthrough time [s]'] < integration_end)) ]
        #Calculating volumetric flow - new tested code
        df.loc[:,'Volumetric flow in [m3/s]'] = pd.Series(sum(df.loc[:,comp+ ' inlet flow [mol/s]'] for comp in ['CO2', 'N2', 'H2O'])*R*T_exp/P_exp, index=df.index)
        df.loc[:,'Volumetric flow out [m3/s]'] = pd.Series(sum(df.loc[:,comp + ' molar flow [mol/s]'] for comp in ['CO2', 'N2', 'H2O'])*R*T_exp/P_exp, index=df.index)
        #Calculating inlet concentrations
        c_avgin = {}
        for comp in ['CO2', 'N2', 'He', 'H2O', 'O2']:
            label = comp + ' inlet concentration [mol/m3]'
            c_avgin[comp] = df.loc[:,label].mean()
        #Calculating the average institial inlet velocity
        v_avgin = df.loc[:,'Volumetric flow in [m3/s]'].mean()/(bed_area*bed_porosity)
        #Now calculating the term inside the integral for each time step
        for comp in ['CO2', 'N2', 'He', 'H2O']:
            label = 'Smoothed Normalised ' + comp + ' molar flow [mol/s]'
            inlet_label = comp + ' inlet mole fraction [-]'
            new_label = comp + ' integral term'
            df.loc[:,new_label] = pd.Series([1]*len(df) - df.loc[:,label]/df.loc[len(df) - 1,label], index=df.index)
        #Doing the trapezium rule to calculate the area above the curve for each time step
        for comp in ['CO2', 'N2', 'He', 'H2O']:
            label = comp + ' integral term'
            new_label = comp + ' area integral'
            df.loc[:,new_label] = pd.Series([(df.loc[i,'Breakthrough time [s]']-df.loc[i-1,'Breakthrough time [s]'])*0.5*(df.loc[i,label] + df.loc[i-1,label]) if i > 0 else 0 for i in range(len(df))] , index=df.index)
        #Summing the area
        sum_area = {}
        for comp in ['CO2', 'N2', 'He', 'H2O']:
            label = comp + ' area integral'
            sum_area[comp] = df[label].sum()
        #Now calculating the volume based loading of the bed
        loading_volume_based = {}
        for comp in ['CO2', 'N2', 'He', 'H2O']:
            loading_volume_based[comp] = bed_porosity*c_avgin[comp]*(sum_area[comp]*v_avgin/bed_length - 1)/(1-bed_porosity)
        #And now converting the mass based loading of the bed
        loading_mass_based = {}
        for comp in ['CO2', 'N2', 'He', 'H2O']:
            loading_mass_based[comp] = loading_volume_based[comp]/pellet_density 
            # this is finding purely mass loading 
        for comp in ['CO2', 'N2', 'He', 'H2O']:
            BoxArea = df.loc[len(df)-1,'Breakthrough time [s]'] * sum(df[comp + ' inlet flow [mol/s]']) /len(df) # finding the box area from which to subtract area under curve - this is in mol
            AreaA = np.trapz(y = df.loc[:,'Smoothed ' + comp + ' molar flow [mol/s]'], x = df.loc[:,'Breakthrough time [s]']) # numerical integration to find area under the curve - this is also in mol
            loading_mass_based[comp] = (BoxArea - AreaA)/(self.conditions['bed_mass']) # bed mass is in kg, BoxArea is in mol so leaving as is is equivalent to mg/g
        # the issue with this is that adsorption cpaacity is measured by the difference between sample and blank. Causing an issue as we need a method in the class which can act on two objetc
        # selectivity = (loading_mass_based['CO2'] / loading_mass_based['N2']) * (sum(df.loc[:, 'CO2 inlet mole fraction [-]']) / sum(df.loc[:, 'N2 inlet mole fraction [-]']))
        return ['Mass based loading (mg/g sample)', loading_mass_based], ['Volume based loading (mmol/g sample)',loading_volume_based]


def standard_output(sample_object , blank_object,export_file_name, material_name, whether_to_export,y_axis = 'y(t)Q(t)/yinQin CO2', width = 16, height = 11):
    sample_df = sample_object.sorted_data
    blank_df = blank_object.sorted_data
    x_axis = 'Breakthrough time [s]'
    sns.set_theme(style='white')
    f, axs = plt.subplots(3,1, figsize = (width, height))
    f.suptitle((material_name + ' readings at: ', sample_object.conditions['T_exp'], 'K. At CO2 percent of: ', 10* round(sum(sample_df['yinQin CO2'])/len(sample_df), 1), '%'))
    axs[0].set_title('blank reading' + y_axis)
    sns.lineplot(data = blank_df, x=x_axis, y = y_axis, ax = axs[0], legend = 'auto')
    if (y_axis[:10] == 'Normalised') or (y_axis[9:19] == 'Normalised') or (y_axis[:15] == 'y(t)Q(t)/yinQin'):
        sns.lineplot(data =blank_df, x=x_axis, y = 1, ax = axs[0], dashes = True)
    axs[1].set_title('sample reading ' + y_axis)
    sns.lineplot(data = sample_df, x=x_axis, y = y_axis, ax = axs[1], legend = 'auto')
    if (y_axis[:10] == 'Normalised') or (y_axis[9:19] == 'Normalised') or (y_axis[:15] == 'y(t)Q(t)/yinQin'):
        sns.lineplot(data =sample_df, x=x_axis, y = 1, ax = axs[1], dashes = True)
    axs[2].set_title('blank and sample reading' + y_axis)
    sns.lineplot(data = blank_df, x=x_axis, y = y_axis, ax = axs[2], legend = 'auto')
    sns.lineplot(data = sample_df, x=x_axis, y = y_axis, ax = axs[2], legend = 'auto')
    if (y_axis[:10] == 'Normalised') or (y_axis[9:19] == 'Normalised') or (y_axis[:15] == 'y(t)Q(t)/yinQin'):
        if len(sample_df) > len(blank_df):
            sns.lineplot(data =sample_df, x=x_axis, y = 1, ax = axs[2], dashes = True)
        else:
            sns.lineplot(data = blank_df, x= x_axis, y=1, ax =axs[2], dashes=True)
    plt.tight_layout()
    print('Below are the CO2 adsorptions for blank and sample: ')
    print('CO2 adsorbed by the blank was: ', blank_object.loading_data[0][1]['CO2'], ' mmol/g')
    print('CO2 adsorbed by the sample was: ', sample_object.loading_data[0][1]['CO2'], ' mmol/g')
    print('')
    print('This gives a capture capacity of: ', (sample_object.loading_data[0][1]['CO2'] - blank_object.loading_data[0][1]['CO2']), ' mmol/g')
    print('')
    xCO2_xN2 = (sample_object.loading_data[0][1]['CO2'] - blank_object.loading_data[0][1]['CO2'])/(sample_object.loading_data[0][1]['N2'] - blank_object.loading_data[0][1]['N2'])
    yCO2_yN2 = sum(sum(df.loc[:, 'CO2 inlet mole fraction [-]'])/len(df) for df in [sample_df, blank_df])/sum(sum(df.loc[:, 'N2 inlet mole fraction [-]'])/len(df) for df in [sample_df, blank_df])
    print('Selectivity (x_CO2 * y_CO2 / x_N2 * y_N2) of: ', round((xCO2_xN2 * yCO2_yN2), 2))
    if whether_to_export == True:
        sample_object.export_data(fileName = 'sample_' + export_file_name)
        blank_object.export_data(fileName = 'blank_' + export_file_name)
