# Dictionaries used in setup.

# The commands necessary to get your mac address:
# Start a command line with iPython
# from uuid import getnode
# getnode()       <- this is the number you need
# In order to double-check with hex mac-address of ipconfig \all, run this:
# mac_address = ("".join(c + "-" if i % 2 else c for i, c in \
#                                     enumerate(hex(uuid.getnode())[2:].zfill(12)))[:-1]).upper()
mac_dict = {'203178706891063': 'CDickel_Desktop',
            '203308017140376': 'Adriaans_Macbook',
            '963460802314': 'Pagani_S17',
            '215977245841658': 'La_Maserati_JrJr',
            '215977245830009': 'La_Vespa',
            '13795386264098': 'Serwans_Laptop',
            '215977245834050': 'Xiang_PC',
            '215977245834050': 'La_Ducati',
            '203050745808564': 'La_Ducati_Jr',
            '57277341811788': 'Simulation_PC',
            '272774795670508': 'Nathans_Laptop',
            '46390847630': 'Aprilia_Jr',
            '198690273946987': 'Bart_Laptop',
            '167746772205643': 'NuovaFerrari',
            '167746772714689': 'Xiang_PC',
            '180725258210527': 'Niels_macbook',
            '109952948723616': 'Ramiro_Desktop',
            '23213':'Malay_Laptop',
            '31054844829911': 'Sjoerd_laptop',
            '26830024075025': 'Qudev_testbench',
            '88623634748008':'LaAprilia_1',
            '215977245830009': 'LaVespa',
            '79497677591501':'PaganiMeas',
            }

data_dir_dict = {'tud276606_FPGA_PC': 'D:\Experiments/CBox_Testing/Data',
                 'CDickel_Desktop': 'D:\Experiments/ExperimentName/Data',
                 'Sjoerd_laptop': 'D:\data',
                 'Malay_Laptop':'D:\Tomo datasets',
                 'Adriaans_Macbook': ('/Users/Adriaan/Documents/Testing/Data'),
                 'Niels_macbook': '/Users/nbultink/temp_data',
                 'Pagani_S17':  'D:\Experiments\V2_Sifaka B\Data',
                 'TUD277449': 'D:\Experiments/1710_FlipchipS7/Data',
                 'La_Maserati_Jr': 'D:\\Experiments\\1610_QcodesTests\\Data',
                 'La_Maserati_JrJr': 'D:\\Experiments\\1810_Purcell_3Q_O4_3_and_O2_1\\data',
                 'La_Vespa': 'D:\\Experiments\\161111_LaVespa_Intel_HR\\Data',
                 'Xiang_PC': 'D:\\data\\IntelDemo',
                 'Serwans_Laptop': 'W:/tnw/NS/qt/Serwan/MuxMon/',
                 # 'La_Ducati': 'D:\\Experiments\\1704_NWv74_Magnet\\Data',
                 # 'La_Ducati': 'F:\\Experiments\\1805_NW_Cheesymon_P4\\Data',
                 'La_Ducati': 'D:\\Experiments\\1909_Cheesymon_v8_E2\\Data',
                # 'La_Ducati': 'D:\\Experiments\\1810_Cheesymon_py3_C3\\Data',

                 # 'La_Ducati': 'D:\\Experiments\\1907_Cheesymon_v6_F4\\Data',

                 'La_Ducati_Jr': 'D:\\Experiments\\1805_NW_Cheesymon_P4\\Data',
                 'Simulation_PC': r'D:\Experiments/testSingleShotFidelityAnalysis/Data',
                 # 'Ramiro_Desktop': r'D:\\PhD_RS\\data_local',
                 'Ramiro_Desktop': r'\\TUD277449\Experiments\1801_QECVQE\Data',
                 # 'Ramiro_Desktop': r'\\131.180.82.81\\Experiments\\1702_Starmon\\data',
                 # 'Ramiro_Desktop': r'\\131.180.82.81\\data',
                 # 'Ramiro_Desktop': r'\\131.180.82.190\\Experiments\\1611_Starmon\\Data',
                 'Nathans_Laptop': r'D:/nlangford\My Documents\Projects\Rabi Model\Experiment_1504\Data',
                 'Bart_Laptop': r'C:\Experiments/NumericalOptimization/Data',
                 'Qudev_testbench' : r'E:\Control software\data',
                 'Luthi_Desktop': r'\\TUD277620\\Experiments\\1805_NW_Cheesymon_P4\\Data',
                 'Thijs_laptop' : 'C:\\Users\\Thijs\\Documents\\TUDelft\\PhD\\Data',
                 'Thijs_Desktop': r'\\TUD277620\\Experiments\\1805_NW_Cheesymon_P4\\Data',
                 'LaVespa': r'D:\Experiments\18031_Intel_resonators',
                 'LaAprilia_1' : r'D:\\Experiments\\1812_CZsims\\data',
                 'PaganiMeas':r'D:\\Experiments\\1903_S7_VIO_W29_C4\\data',
                 }
