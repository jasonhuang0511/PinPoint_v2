###########################################################
#
# future Classification information
#
###########################################################
eq_generics_onshore_production = ['FFB', 'FFD', 'IFB']

fi_generics_onshore_producion = ['TFC', 'TFT', 'TFS']

cm_generics_onshore_production = ['RBT', 'PT', 'ROC', 'APW', 'PKR', 'ZME', 'FO', 'IOE', 'AE', 'PAL', 'ZRR', 'SH', 'PYL',
                                  'ZSY', 'VV', 'BIT', 'ZRO', 'LHD', 'IRE', 'AC', 'CKC', 'IMR', 'PVC', 'POL', 'CB',
                                  'KEE', 'FGL', 'RT', 'AK', 'DCE', 'LGE', 'ZZP', 'TRC', 'JCI', 'DCS', 'URI', 'PGR',
                                  'CU', 'XII', 'AA', 'ZST', 'XOO', 'PBL', 'SYR', 'ZNA', 'KSP']

key_wind_generic = 'CODE'
key_wind_exchange = 'exchange'
''' Chinese futures WIND and Bloomberg ticker '''
generic_wind_dict = {
    'AA': {key_wind_generic: 'AL', key_wind_exchange: 'SHF'},
    'AC': {key_wind_generic: 'C', key_wind_exchange: 'DCE'},
    'AE': {key_wind_generic: 'M', key_wind_exchange: 'DCE'},
    'AK': {key_wind_generic: 'A', key_wind_exchange: 'DCE'},
    'APW': {key_wind_generic: 'AP', key_wind_exchange: 'CZC'},
    'BIT': {key_wind_generic: 'BU', key_wind_exchange: 'SHF'},
    'CB': {key_wind_generic: 'SR', key_wind_exchange: 'CZC'},
    'CKC': {key_wind_generic: 'JM', key_wind_exchange: 'DCE'},
    'CU': {key_wind_generic: 'CU', key_wind_exchange: 'SHF'},
    'DCE': {key_wind_generic: 'JD', key_wind_exchange: 'DCE'},
    'DCS': {key_wind_generic: 'CS', key_wind_exchange: 'DCE'},
    'FGL': {key_wind_generic: 'FG', key_wind_exchange: 'CZC'},
    'FO': {key_wind_generic: 'FU', key_wind_exchange: 'SHF'},
    'IMR': {key_wind_generic: 'SM', key_wind_exchange: 'CZC'},
    'IOE': {key_wind_generic: 'I', key_wind_exchange: 'DCE'},
    'IRE': {key_wind_generic: 'SF', key_wind_exchange: 'CZC'},
    'JCI': {key_wind_generic: 'CJ', key_wind_exchange: 'CZC'},
    'KEE': {key_wind_generic: 'J', key_wind_exchange: 'DCE'},
    'KSP': {key_wind_generic: 'SP', key_wind_exchange: 'SHF'},
    'LGE': {key_wind_generic: 'EG', key_wind_exchange: 'DCE'},
    'LHD': {key_wind_generic: 'LH', key_wind_exchange: 'DCE'},
    'PAL': {key_wind_generic: 'P', key_wind_exchange: 'DCE'},
    'PBL': {key_wind_generic: 'PB', key_wind_exchange: 'SHF'},
    'PGR': {key_wind_generic: 'PG', key_wind_exchange: 'DCE'},
    'PKR': {key_wind_generic: 'PK', key_wind_exchange: 'CZC'},
    'POL': {key_wind_generic: 'L', key_wind_exchange: 'DCE'},
    'PT': {key_wind_generic: 'TA', key_wind_exchange: 'CZC'},
    'PVC': {key_wind_generic: 'V', key_wind_exchange: 'DCE'},
    'PYL': {key_wind_generic: 'PP', key_wind_exchange: 'DCE'},
    'RBT': {key_wind_generic: 'RB', key_wind_exchange: 'SHF'},
    'ROC': {key_wind_generic: 'HC', key_wind_exchange: 'SHF'},
    'RT': {key_wind_generic: 'RU', key_wind_exchange: 'SHF'},
    'SH': {key_wind_generic: 'Y', key_wind_exchange: 'DCE'},
    'SYR': {key_wind_generic: 'EB', key_wind_exchange: 'DCE'},
    'TRC': {key_wind_generic: 'ZC', key_wind_exchange: 'CZC'},
    'URI': {key_wind_generic: 'UR', key_wind_exchange: 'CZC'},
    'VV': {key_wind_generic: 'CF', key_wind_exchange: 'CZC'},
    'XII': {key_wind_generic: 'NI', key_wind_exchange: 'SHF'},
    'XOO': {key_wind_generic: 'SN', key_wind_exchange: 'SHF'},
    'ZME': {key_wind_generic: 'MA', key_wind_exchange: 'CZC'},
    'ZNA': {key_wind_generic: 'ZN', key_wind_exchange: 'SHF'},
    'ZRO': {key_wind_generic: 'OI', key_wind_exchange: 'CZC'},
    'ZRR': {key_wind_generic: 'RM', key_wind_exchange: 'CZC'},
    'ZST': {key_wind_generic: 'SS', key_wind_exchange: 'SHF'},
    'ZSY': {key_wind_generic: 'SA', key_wind_exchange: 'CZC'},
    'ZZP': {key_wind_generic: 'PF', key_wind_exchange: 'CZC'},

    # financials
    # equity index
    'FFB': {key_wind_generic: 'IH', key_wind_exchange: 'CFE'},
    'FFD': {key_wind_generic: 'IC', key_wind_exchange: 'CFE'},
    'IFB': {key_wind_generic: 'IF', key_wind_exchange: 'CFE'},

    # FI
    'TFC': {key_wind_generic: 'TF', key_wind_exchange: 'CFE'},
    'TFT': {key_wind_generic: 'T', key_wind_exchange: 'CFE'},
    'TFS': {key_wind_generic: 'TS', key_wind_exchange: 'CFE'}
}

cm_cn_group_grains_oilseeds = ['M', 'P', 'RM', 'Y', 'OI', 'C', 'A', 'CS']
cm_cn_group_livestock = ['JD', 'LH']
cm_cn_group_softs = ['AP', 'PK', 'CF', 'SR', 'RU', 'CJ', 'SP']
cm_cn_group_base_metal = ['CU', 'NI', 'AL', 'SS', 'SN', 'PB', 'ZN']
cm_cn_group_black = ['RB', 'HC', 'I', 'SA', 'SF', 'JM', 'SM', 'J', 'FG', 'ZC']
cm_cn_group_chemicals = ['TA', 'MA', 'PP', 'V', 'L', 'EG', 'PF', 'UR', 'EB']
cm_cn_group_energy = ['FU', 'BU', 'PG']
cm_cn_group_stock_index = ['IC', 'IH', 'IF']
cm_cn_group_interest_rate = ['T', 'TS', 'TF']

cm_cn_sector_financial = cm_cn_group_stock_index + cm_cn_group_interest_rate
cm_cn_sector_agriculture = cm_cn_group_grains_oilseeds + cm_cn_group_livestock + cm_cn_group_softs
cm_cn_sector_industrials = cm_cn_group_base_metal + cm_cn_group_black
cm_cn_sector_refineries = cm_cn_group_chemicals + cm_cn_group_energy
cm_cn_all = cm_cn_sector_financial + cm_cn_sector_agriculture + cm_cn_sector_industrials + cm_cn_sector_refineries
sector_group_name_list = ['cm_cn_group_grains_oilseeds', 'cm_cn_group_livestock', 'cm_cn_group_softs',
                          'cm_cn_group_base_metal', 'cm_cn_group_black', 'cm_cn_group_chemicals', 'cm_cn_group_energy',
                          'cm_cn_group_stock_index', 'cm_cn_group_interest_rate', 'cm_cn_sector_agriculture',
                          'cm_cn_sector_industrials', 'cm_cn_sector_refineries', 'cm_cn_sector_financial', 'cm_cn_all']

cm_cn_product_spread_dict = {
    'M_RM': ['M', 'RM'],
    'Y_P': ['Y', 'P'],
    'OI_P': ['OI', 'P'],
    'Y_OI': ['Y', 'OI'],
    'CS_C': ['CS', 'C'],
    'RB_HC': ['RB', 'HC'],
    'L_PP': ['L', 'PP'],
    'PP_MA': ['PP', 'MA'],

}

cm_cn_product_ratio_dict = {
    'Y/M': ['Y', 'M'], 'OI/RM': ['OI', 'RM'], 'RB/I': ['RB', 'I'],
    'J/JM': ['J', 'JM'], 'JM/I': ['JM', 'I'],
    'AL/ZN': ['AL', 'ZN'], 'CU/ZN': ['CU', 'ZN']
}
###########################################################


###########################################################
#
# future basic information
#
###########################################################

# 交易时间段
fut_code_trading_min_time_dict = {
    'A.DCE': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'AL.SHF': [['21:00', '01:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'AP.CZC': [['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'BU.SHF': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'C.DCE': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'CF.CZC': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'CJ.CZC': [['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'CS.DCE': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'CU.SHF': [['21:00', '01:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'EB.DCE': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'EG.DCE': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'FG.CZC': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'FU.SHF': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'HC.SHF': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'IC.CFE': [['09:30', '11:30'], ['13:00', '15:00']],
    'I.DCE': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'IF.CFE': [['09:30', '11:30'], ['13:00', '15:00']],
    'IH.CFE': [['09:30', '11:30'], ['13:00', '15:00']],
    'J.DCE': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'JD.DCE': [['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'JM.DCE': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'L.DCE': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'LH.DCE': [['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'MA.CZC': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'M.DCE': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'NI.SHF': [['21:00', '01:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'OI.CZC': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'PB.SHF': [['21:00', '01:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'P.DCE': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'PF.CZC': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'PG.DCE': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'PK.CZC': [['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'PP.DCE': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'RB.SHF': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'RM.CZC': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'RU.SHF': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'SA.CZC': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'SF.CZC': [['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'SM.CZC': [['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'SN.SHF': [['21:00', '01:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'SP.SHF': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'SR.CZC': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'SS.SHF': [['21:00', '01:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'TA.CZC': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'T.CFE': [['09:30', '11:30'], ['13:00', '15:15']],
    'TF.CFE': [['09:30', '11:30'], ['13:00', '15:15']],
    'TS.CFE': [['09:30', '11:30'], ['13:00', '15:15']],
    'UR.CZC': [['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'V.DCE': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'Y.DCE': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'ZC.CZC': [['21:00', '23:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']],
    'ZN.SHF': [['21:00', '01:00'], ['09:00', '10:15'], ['10:30', '11:30'], ['13:30', '15:00']]}

# 国内期货品种清单
fut_code_list = ['A.DCE', 'AL.SHF', 'AP.CZC', 'BU.SHF', 'C.DCE', 'CF.CZC', 'CJ.CZC', 'CS.DCE', 'CU.SHF', 'EB.DCE',
                 'EG.DCE', 'FG.CZC', 'FU.SHF', 'HC.SHF', 'IC.CFE', 'I.DCE', 'IF.CFE', 'IH.CFE', 'J.DCE', 'JD.DCE',
                 'JM.DCE', 'L.DCE', 'LH.DCE', 'MA.CZC', 'M.DCE', 'NI.SHF', 'OI.CZC', 'PB.SHF', 'P.DCE', 'PF.CZC',
                 'PG.DCE', 'PK.CZC', 'PP.DCE', 'RB.SHF', 'RM.CZC', 'RU.SHF', 'SA.CZC', 'SF.CZC', 'SM.CZC', 'SN.SHF',
                 'SP.SHF', 'SR.CZC', 'SS.SHF', 'TA.CZC', 'T.CFE', 'TF.CFE', 'TS.CFE', 'UR.CZC', 'V.DCE', 'Y.DCE',
                 'ZC.CZC', 'ZN.SHF']

# 期货英文名与wind code mapping
fut_simple_code_to_windcode_mapping_dict = {key.split('.')[0]: key for key in fut_code_list}
fut_windcode_to_simple_code_mapping_dict = {key: key.split('.')[0] for key in fut_code_list}

exchange_full_name_to_simple_name_mapping = {'SHFE': 'SHF', 'INE': 'INE', 'CZCE': 'CZC', 'DCE': 'DCE', 'CFFEX': 'CFE'}

# 期货品种与中文名称mapping
fut_code_to_chinese_name_mapping = {'T.CFE': 'CFFEX10年期国债期货', 'IC.CFE': '中证500期货', 'TF.CFE': 'CFFEX5年期国债期货',
                                    'TS.CFE': 'CFFEX2年期国债期货', 'IM.CFE': '中证1000期货', 'IF.CFE': '沪深300期货',
                                    'IH.CFE': '上证50期货', 'FG.CZC': '玻璃主力', 'CF.CZC': '棉花主力', 'RM.CZC': '菜粕主力',
                                    'PF.CZC': '短纤主力', 'AP.CZC': '苹果主力', 'CY.CZC': '棉纱主力', 'ER.CZC': '早籼稻主力',
                                    'ME.CZC': '甲醇主力', 'RO.CZC': '菜油主力', 'TC.CZC': '动力煤主力', 'WS.CZC': '强麦主力',
                                    'WT.CZC': '硬麦主力', 'SA.CZC': '纯碱主力', 'PK.CZC': '花生主力', 'CJ.CZC': '红枣主力',
                                    'UR.CZC': '尿素主力', 'TA.CZC': 'PTA主力', 'OI.CZC': '菜油主力', 'MA.CZC': '甲醇主力',
                                    'RS.CZC': '菜籽主力', 'ZC.CZC': '动力煤主力', 'PM.CZC': '普麦主力', 'LR.CZC': '晚籼稻主力',
                                    'SR.CZC': '白糖主力', 'RI.CZC': '早籼稻主力', 'SF.CZC': '硅铁主力', 'WH.CZC': '强麦主力',
                                    'JR.CZC': '粳稻主力', 'SM.CZC': '锰硅主力', 'FB.DCE': '纤维板主力', 'PG.DCE': 'LPG主力',
                                    'EB.DCE': '苯乙烯主力', 'CS.DCE': '玉米淀粉主力', 'C.DCE': '玉米主力', 'V.DCE': 'PVC主力',
                                    'J.DCE': '焦炭主力', 'BB.DCE': '胶合板主力', 'M.DCE': '豆粕主力', 'A.DCE': '豆一主力',
                                    'PP.DCE': '聚丙烯主力', 'P.DCE': '棕榈油主力', 'B.DCE': '豆二主力', 'JD.DCE': '鸡蛋主力',
                                    'JM.DCE': '焦煤主力', 'L.DCE': '塑料主力', 'I.DCE': '铁矿石主力', 'Y.DCE': '豆油主力',
                                    'RR.DCE': '粳米主力', 'EG.DCE': '乙二醇主力', 'LH.DCE': '生猪主力', 'NR.INE': '20号胶主力',
                                    'SC.INE': '原油主力', 'LU.INE': '低硫燃料油主力', 'BC.INE': '国际铜主力', 'SCTAS.INE': '原油TAS指令主力',
                                    'RB.SHF': '螺纹钢主力', 'SN.SHF': '沪锡主力', 'NI.SHF': '沪镍主力', 'WR.SHF': '线材主力',
                                    'FU.SHF': '燃油主力', 'SS.SHF': '不锈钢主力', 'PB.SHF': '沪铅主力', 'RU.SHF': '橡胶主力',
                                    'HC.SHF': '热轧卷板主力', 'BU.SHF': '沥青主力', 'SP.SHF': '纸浆主力', 'AU.SHF': '黄金主力',
                                    'AL.SHF': '沪铝主力', 'CU.SHF': '沪铜主力', 'ZN.SHF': '沪锌主力', 'AG.SHF': '白银主力'}

# 期货上市日期
fut_code_listing_date = {'KC.COMDTY': '2000-01-03', 'BO.COMDTY': '2000-01-03', 'CC.COMDTY': '2000-01-03',
                         'CT.COMDTY': '2000-01-03', 'SM.COMDTY': '2000-01-03', 'SB.COMDTY': '2000-01-03',
                         'LH.COMDTY': '2000-01-03', 'LC.COMDTY': '2000-01-03', 'KW.COMDTY': '2000-01-03',
                         'A.DCE': '2000-01-04', 'CO.COMDTY': '2000-01-04', 'GC.COMDTY': '2000-01-04',
                         'QS.COMDTY': '2000-01-04', 'HG.COMDTY': '2000-01-04', 'AL.SHF': '2000-01-04',
                         'HO.COMDTY': '2000-01-04', 'PL.COMDTY': '2000-01-04', 'CL.COMDTY': '2000-01-04',
                         'PA.COMDTY': '2000-01-04', 'CU.SHF': '2000-01-04', 'NG.COMDTY': '2000-01-04',
                         'SI.COMDTY': '2000-01-04', 'RU.SHF': '2000-01-04', 'M.DCE': '2000-07-17',
                         'CF.CZC': '2004-06-01',
                         'FU.SHF': '2004-08-25', 'C.DCE': '2004-09-22', 'XB.COMDTY': '2005-10-03',
                         'SR.CZC': '2006-01-06',
                         'Y.DCE': '2006-01-09', 'TA.CZC': '2006-12-18', 'ZN.SHF': '2007-03-26', 'OI.CZC': '2007-06-08',
                         'L.DCE': '2007-07-31', 'P.DCE': '2007-10-29', 'DF.COMDTY': '2008-01-14',
                         'RB.SHF': '2009-03-27',
                         'V.DCE': '2009-05-25', 'IF.CFE': '2010-04-16', 'PB.SHF': '2011-03-24', 'J.DCE': '2011-04-15',
                         'MA.CZC': '2011-10-28', 'FG.CZC': '2012-12-03', 'RM.CZC': '2012-12-28', 'JM.DCE': '2013-03-22',
                         'TF.CFE': '2013-09-06', 'ZC.CZC': '2013-09-26', 'BU.SHF': '2013-10-09', 'I.DCE': '2013-10-18',
                         'JD.DCE': '2013-11-08', 'PP.DCE': '2014-02-28', 'HC.SHF': '2014-03-21', 'SF.CZC': '2014-08-08',
                         'SM.CZC': '2014-08-08', 'CS.DCE': '2014-12-19', 'T.CFE': '2015-03-20', 'SN.SHF': '2015-03-27',
                         'NI.SHF': '2015-03-27', 'IC.CFE': '2015-04-16', 'IH.CFE': '2015-04-16', 'AP.CZC': '2017-12-22',
                         'TS.CFE': '2018-08-17', 'SP.SHF': '2018-11-27', 'EG.DCE': '2018-12-10', 'CJ.CZC': '2019-04-30',
                         'UR.CZC': '2019-08-09', 'SS.SHF': '2019-09-25', 'EB.DCE': '2019-09-26', 'SA.CZC': '2019-12-06',
                         'PG.DCE': '2020-03-30', 'PF.CZC': '2020-10-12', 'LH.DCE': '2021-01-08', 'PK.CZC': '2021-02-01'}

# 期货bpv
bpv = {'A': 10.0, 'AG': 15.0, 'AL': 5.0, 'AP': 10.0, 'AU': 1000.0, 'B': 10.0, 'BB': 500.0, 'BC': 5.0, 'BU': 10.0,
       'C': 10.0, 'CF': 5.0, 'CJ': 5.0, 'CS': 10.0, 'CU': 5.0, 'CY': 5.0, 'EB': 5.0, 'EG': 10.0, 'ER': 10.0, 'FB': 10.0,
       'FG': 20.0, 'FU': 10.0, 'HC': 10.0, 'I': 100.0, 'IC': 200.0, 'IF': 300.0, 'IH': 300.0, 'IM': 200.0, 'J': 100.0,
       'JD': 10.0, 'JM': 60.0, 'JR': 20.0, 'L': 5.0, 'LH': 16.0, 'LR': 20.0, 'LU': 10.0, 'M': 10.0, 'MA': 10.0,
       'ME': 50.0, 'NI': 1.0, 'NR': 10.0, 'OI': 10.0, 'P': 10.0, 'PB': 5.0, 'PF': 5.0, 'PG': 20.0, 'PK': 5.0,
       'PM': 50.0, 'PP': 5.0, 'RB': 10.0, 'RI': 20.0, 'RM': 10.0, 'RO': 5.0, 'RR': 10.0, 'RS': 10.0, 'RU': 10.0,
       'SA': 20.0, 'SC': 1000.0, 'SCTAS': 1000.0, 'SF': 5.0, 'SM': 5.0, 'SN': 1.0, 'SP': 10.0, 'SR': 10.0, 'SS': 5.0,
       'T': 10000.0, 'TA': 5.0, 'TC': 200.0, 'TF': 10000.0, 'TS': 20000.0, 'UR': 20.0, 'V': 5.0, 'WH': 20.0, 'WR': 10.0,
       'WS': 10.0, 'WT': 10.0, 'Y': 10.0, 'ZC': 100.0, 'ZN': 5.0}

fut_code_bpv = {key: bpv[fut_windcode_to_simple_code_mapping_dict[key]] for key in
                fut_windcode_to_simple_code_mapping_dict.keys()}

fut_code_sort_by_roll_month = {'12': ['AL.SHF', 'CU.SHF', 'SP.SHF', 'PB.SHF', 'PG.DCE', 'EB.DCE', 'NI.SHF', 'SN.SHF',
                                      'ZN.SHF', 'SS.SHF', 'IH.CFE',
                                      'IC.CFE', 'IF.CFE'],
                               '4': ['TF.CFE', 'T.CFE', 'TS.CFE'],
                               '3': ['PK.CZC', 'TA.CZC',
                                     'AP.CZC', 'RB.SHF', 'HC.SHF',
                                     'C.DCE', 'M.DCE', 'A.DCE', 'BU.SHF', 'SR.CZC', 'JM.DCE', 'JD.DCE', 'CS.DCE',
                                     'FG.CZC',
                                     'FU.SHF', 'SM.CZC', 'I.DCE',
                                     'SF.CZC', 'CJ.CZC', 'J.DCE', 'EG.DCE', 'LH.DCE', 'P.DCE', 'L.DCE', 'V.DCE',
                                     'PP.DCE',
                                     'RU.SHF', 'Y.DCE', 'ZC.CZC',
                                     'UR.CZC', 'CF.CZC', 'MA.CZC', 'OI.CZC', 'RM.CZC', 'SA.CZC', 'PF.CZC']}

fut_code_roll_instrument = {
    'AL.SHF': {'01': '02', '02': '03', '03': '04', '04': '05',
               '05': '06', '06': '07', '07': '08', '08': '09',
               '09': '10', '10': '11', '11': '12', '12': '01'},
    'C.DCE': {'01': '05', '05': '09', '09': '01'},
    'M.DCE': {'01': '05', '05': '09', '09': '01'},
    'A.DCE': {'01': '05', '05': '09', '09': '01'},
    'AP.CZC': {'01': '05', '05': '10', '10': '01'},
    'BU.SHF': {'01': '05', '05': '09', '09': '01'},
    'SR.CZC': {'01': '05', '05': '09', '09': '01'},
    'JM.DCE': {'01': '05', '05': '09', '09': '01'},
    'CU.SHF': {'01': '02', '02': '03', '03': '04', '04': '05',
               '05': '06', '06': '07', '07': '08', '08': '09',
               '09': '10', '10': '11', '11': '12', '12': '01'},
    'JD.DCE': {'01': '05', '05': '09', '09': '01'},
    'CS.DCE': {'01': '05', '05': '09', '09': '01'},
    'FG.CZC': {'01': '05', '05': '09', '09': '01'},
    'FU.SHF': {'01': '05', '05': '09', '09': '01'},
    'SM.CZC': {'01': '05', '05': '09', '09': '01'},
    'I.DCE': {'01': '05', '05': '09', '09': '01'},
    'SF.CZC': {'01': '05', '05': '09', '09': '01'},
    'CJ.CZC': {'01': '05', '05': '09', '09': '01'},
    'J.DCE': {'01': '05', '05': '09', '09': '01'},
    'SP.SHF': {'01': '02', '02': '03', '03': '04', '04': '05',
               '05': '06', '06': '07', '07': '08', '08': '09',
               '09': '10', '10': '11', '11': '12', '12': '01'},
    'EG.DCE': {'01': '05', '05': '09', '09': '01'},
    'LH.DCE': {'01': '05', '05': '09', '09': '01'},
    'P.DCE': {'01': '05', '05': '09', '09': '01'},
    'PB.SHF': {'01': '02', '02': '03', '03': '04', '04': '05',
               '05': '06', '06': '07', '07': '08', '08': '09',
               '09': '10', '10': '11', '11': '12', '12': '01'},
    'PG.DCE': {'01': '02', '02': '03', '03': '04', '04': '05',
               '05': '06', '06': '07', '07': '08', '08': '09',
               '09': '10', '10': '11', '11': '12', '12': '01'},
    'L.DCE': {'01': '05', '05': '09', '09': '01'},
    'V.DCE': {'01': '05', '05': '09', '09': '01'},
    'PP.DCE': {'01': '05', '05': '09', '09': '01'},
    'RB.SHF': {'01': '05', '05': '10', '10': '01'},
    'HC.SHF': {'01': '05', '05': '10', '10': '01'},
    'RU.SHF': {'01': '05', '05': '09', '09': '01'},
    'Y.DCE': {'01': '05', '05': '09', '09': '01'},
    'EB.DCE': {'01': '02', '02': '03', '03': '04', '04': '05',
               '05': '06', '06': '07', '07': '08', '08': '09',
               '09': '10', '10': '11', '11': '12', '12': '01'},
    'ZC.CZC': {'01': '05', '05': '09', '09': '01'},
    'UR.CZC': {'01': '05', '05': '09', '09': '01'},
    'CF.CZC': {'01': '05', '05': '09', '09': '01'},
    'NI.SHF': {'01': '02', '02': '03', '03': '04', '04': '05',
               '05': '06', '06': '07', '07': '08', '08': '09',
               '09': '10', '10': '11', '11': '12', '12': '01'},
    'SN.SHF': {'01': '02', '02': '03', '03': '04', '04': '05',
               '05': '06', '06': '07', '07': '08', '08': '09',
               '09': '10', '10': '11', '11': '12', '12': '01'},
    'MA.CZC': {'01': '05', '05': '09', '09': '01'},
    'ZN.SHF': {'01': '02', '02': '03', '03': '04', '04': '05',
               '05': '06', '06': '07', '07': '08', '08': '09',
               '09': '10', '10': '11', '11': '12', '12': '01'},
    'OI.CZC': {'01': '05', '05': '09', '09': '01'},
    'RM.CZC': {'01': '05', '05': '09', '09': '01'},
    'SS.SHF': {'01': '02', '02': '03', '03': '04', '04': '05',
               '05': '06', '06': '07', '07': '08', '08': '09',
               '09': '10', '10': '11', '11': '12', '12': '01'},
    'SA.CZC': {'01': '05', '05': '09', '09': '01'},
    'PF.CZC': {'01': '05', '05': '09', '09': '01'},
    'IH.CFE': {'01': '02', '02': '03', '03': '04', '04': '05',
               '05': '06', '06': '07', '07': '08', '08': '09',
               '09': '10', '10': '11', '11': '12', '12': '01'},
    'IC.CFE': {'01': '02', '02': '03', '03': '04', '04': '05',
               '05': '06', '06': '07', '07': '08', '08': '09',
               '09': '10', '10': '11', '11': '12', '12': '01'},
    'IF.CFE': {'01': '02', '02': '03', '03': '04', '04': '05',
               '05': '06', '06': '07', '07': '08', '08': '09',
               '09': '10', '10': '11', '11': '12', '12': '01'},
    'TF.CFE': {'03': '06', '06': '09', '09': '12', '12': '03'},
    'T.CFE': {'03': '06', '06': '09', '09': '12', '12': '03'},
    'TS.CFE': {'03': '06', '06': '09', '09': '12', '12': '03'},
    'PK.CZC': {'01': '04', '04': '10', '10': '01'},
    'TA.CZC': {'01': '05', '05': '09', '09': '01'}}

# roll group
group_cffex_stock = ['IC.CFE', 'IF.CFE', 'IH.CFE', 'IM.CFE']
group_monthly_roll = ['AL.SHF', 'CU.SHF', 'SP.SHF', 'PB.SHF', 'PG.DCE', 'EB.DCE', 'NI.SHF', 'SN.SHF', 'ZN.SHF',
                      'SS.SHF']
group_quarterly_roll = ['TF.CFE', 'T.CFE', 'TS.CFE']
group_three_times_yearly_roll = ['PK.CZC', 'TA.CZC', 'AP.CZC', 'RB.SHF', 'HC.SHF', 'C.DCE', 'M.DCE', 'A.DCE', 'BU.SHF',
                                 'SR.CZC', 'JM.DCE', 'JD.DCE', 'CS.DCE', 'FG.CZC', 'FU.SHF', 'SM.CZC', 'I.DCE',
                                 'SF.CZC', 'CJ.CZC', 'J.DCE', 'EG.DCE', 'LH.DCE', 'P.DCE', 'L.DCE', 'V.DCE', 'PP.DCE',
                                 'RU.SHF', 'Y.DCE', 'ZC.CZC', 'UR.CZC', 'CF.CZC', 'MA.CZC', 'OI.CZC', 'RM.CZC',
                                 'SA.CZC', 'PF.CZC']

window_monthly_roll = 12
window_yoy_monthly_roll = 3
window_non_monthly_roll = 9
window_yoy_non_monthly_roll = 4
###########################################################

###########################################################
#
# future backtest parameter
#
###########################################################

transaction_costs_default = 0.0005
transaction_costs_low = 0.0003
trading_schedule_rolling_param = 3
trading_schedule_vol_multiplier = 0.1

file_save_path_default = 'C:\\Users\\jason.huang\\research\\backtest\\'

tickers_group_mapping = {
    'cm_cn_group_grains_oilseeds': ['M.DCE', 'P.DCE', 'RM.CZC', 'Y.DCE', 'OI.CZC', 'C.DCE', 'A.DCE', 'CS.DCE'],
    'cm_cn_group_livestock': ['JD.DCE', 'LH.DCE'],
    'cm_cn_group_softs': ['AP.CZC', 'PK.CZC', 'CF.CZC', 'SR.CZC', 'RU.SHF', 'CJ.CZC', 'SP.SHF'],
    'cm_cn_group_base_metal': ['CU.SHF', 'NI.SHF', 'AL.SHF', 'SS.SHF', 'SN.SHF', 'PB.SHF', 'ZN.SHF'],
    'cm_cn_group_black': ['RB.SHF', 'HC.SHF', 'I.DCE', 'SA.CZC', 'SF.CZC', 'JM.DCE', 'SM.CZC', 'J.DCE', 'FG.CZC',
                          'ZC.CZC'],
    'cm_cn_group_chemicals': ['TA.CZC', 'MA.CZC', 'PP.DCE', 'V.DCE', 'L.DCE', 'EG.DCE', 'PF.CZC', 'UR.CZC', 'EB.DCE'],
    'cm_cn_group_energy': ['FU.SHF', 'BU.SHF', 'PG.DCE'],
    'cm_cn_group_stock_index': ['IC.CFE', 'IH.CFE', 'IF.CFE'],
    'cm_cn_group_interest_rate': ['T.CFE', 'TS.CFE', 'TF.CFE'],
    'cm_cn_sector_agriculture': ['M.DCE', 'P.DCE', 'RM.CZC', 'Y.DCE', 'OI.CZC', 'C.DCE', 'A.DCE', 'CS.DCE', 'JD.DCE',
                                 'LH.DCE', 'AP.CZC', 'PK.CZC', 'CF.CZC', 'SR.CZC', 'RU.SHF', 'CJ.CZC', 'SP.SHF'],
    'cm_cn_sector_industrials': ['CU.SHF', 'NI.SHF', 'AL.SHF', 'SS.SHF', 'SN.SHF', 'PB.SHF', 'ZN.SHF', 'RB.SHF',
                                 'HC.SHF', 'I.DCE', 'SA.CZC', 'SF.CZC', 'JM.DCE', 'SM.CZC', 'J.DCE', 'FG.CZC',
                                 'ZC.CZC'],
    'cm_cn_sector_refineries': ['TA.CZC', 'MA.CZC', 'PP.DCE', 'V.DCE', 'L.DCE', 'EG.DCE', 'PF.CZC', 'UR.CZC', 'EB.DCE',
                                'FU.SHF', 'BU.SHF', 'PG.DCE'],
    'cm_cn_sector_financial': ['IC.CFE', 'IH.CFE', 'IF.CFE', 'T.CFE', 'TS.CFE', 'TF.CFE'],
    'cm_cn_sector_commodity': ['M.DCE', 'P.DCE', 'RM.CZC', 'Y.DCE', 'OI.CZC', 'C.DCE', 'A.DCE', 'CS.DCE', 'JD.DCE',
                               'LH.DCE', 'AP.CZC', 'PK.CZC', 'CF.CZC', 'SR.CZC', 'RU.SHF', 'CJ.CZC', 'SP.SHF', 'CU.SHF',
                               'NI.SHF', 'AL.SHF', 'SS.SHF', 'SN.SHF', 'PB.SHF', 'ZN.SHF', 'RB.SHF', 'HC.SHF', 'I.DCE',
                               'SA.CZC', 'SF.CZC', 'JM.DCE', 'SM.CZC', 'J.DCE', 'FG.CZC', 'ZC.CZC', 'TA.CZC', 'MA.CZC',
                               'PP.DCE', 'V.DCE', 'L.DCE', 'EG.DCE', 'PF.CZC', 'UR.CZC', 'EB.DCE', 'FU.SHF', 'BU.SHF',
                               'PG.DCE'],
    'cm_cn_all': ['IC.CFE', 'IH.CFE', 'IF.CFE', 'T.CFE', 'TS.CFE', 'TF.CFE', 'M.DCE', 'P.DCE', 'RM.CZC', 'Y.DCE',
                  'OI.CZC', 'C.DCE', 'A.DCE', 'CS.DCE', 'JD.DCE', 'LH.DCE', 'AP.CZC', 'PK.CZC', 'CF.CZC', 'SR.CZC',
                  'RU.SHF', 'CJ.CZC', 'SP.SHF', 'CU.SHF', 'NI.SHF', 'AL.SHF', 'SS.SHF', 'SN.SHF', 'PB.SHF', 'ZN.SHF',
                  'RB.SHF', 'HC.SHF', 'I.DCE', 'SA.CZC', 'SF.CZC', 'JM.DCE', 'SM.CZC', 'J.DCE', 'FG.CZC', 'ZC.CZC',
                  'TA.CZC', 'MA.CZC', 'PP.DCE', 'V.DCE', 'L.DCE', 'EG.DCE', 'PF.CZC', 'UR.CZC', 'EB.DCE', 'FU.SHF',
                  'BU.SHF', 'PG.DCE'],
    'cm_cn_group_selected_1': ['RB.SHF', 'HC.SHF'],
    'cm_cn_group_selected_2': ['TA.CZC', 'EG.DCE'],
    'cm_cn_group_selected_3': ['AL.SHF', 'CU.SHF', 'ZN.SHF'],
    'cm_cn_group_selected_4': ['M.DCE', 'C.DCE', 'P.DCE', 'Y.DCE'],
    'cm_cn_group_selected_5': ['SR.CZC', 'CF.CZC'],
    'crypto_binance_all': ['XMRUSDT', 'EOSUSDT', 'BTCUSDT', 'ETHUSDT', 'QTUMUSDT', 'ADAUSDT', 'TRXUSDT', 'LTCUSDT',
                           'NEOUSDT', 'DOGEUSDT', 'XLMUSDT', 'LINKUSDT', 'DASHUSDT', 'ETCUSDT', 'BTTUSDT', 'ZECUSDT',
                           'BNBUSDT', 'XRPUSDT'],
    'crypto_binance_selected': ['BTCUSDT', 'ETHUSDT'],
    'crypto_binance_future_all': ['MATICUSDT', 'NEARUSDT', 'GRTUSDT', 'DENTUSDT', 'ANKRUSDT', 'RNDRUSDT', 'ENSUSDT',
                                  'YFIIUSDT', 'HOOKUSDT', '1000SHIBUSDT', 'ZILUSDT', 'IOSTUSDT', 'EOSUSDT', 'AKROUSDT',
                                  'SOLUSDT', 'GALUSDT', 'C98USDT', 'CELRUSDT', 'ATAUSDT', 'DYDXUSDT', 'JASMYUSDT',
                                  'BTCDOMUSDT', 'FXSUSDT', 'KSMUSDT', 'ALPHAUSDT', 'TUSDT',
                                  'SUSHIUSDT', 'BTSUSDT', 'PEOPLEUSDT', 'DUSKUSDT', 'SXPUSDT', 'SANDUSDT', 'RLCUSDT',
                                  'FETUSDT', 'NEOUSDT', 'BANDUSDT', 'MTLUSDT', 'MANAUSDT', 'IOTXUSDT', 'WAVESUSDT',
                                  'DGBUSDT', 'DODOUSDT', 'LENDUSDT', 'LITUSDT', 'MKRUSDT', 'CVCUSDT', 'DARUSDT',
                                  'BZRXUSDT', 'RAYUSDT', 'MASKUSDT', 'ANTUSDT', 'BCHUSDT', 'UNIUSDT', 'APTUSDT',
                                  'ROSEUSDT', 'QNTUSDT', 'LUNA2USDT', 'API3USDT', 'ADAUSDT', 'QTUMUSDT', 'HBARUSDT',
                                  'ICXUSDT', 'BTCSTUSDT', 'CHZUSDT', 'CELOUSDT', 'ONTUSDT', 'STGUSDT', 'ZENUSDT',
                                  'RENUSDT', 'ATOMUSDT', 'TLMUSDT', 'UNFIUSDT', 'STORJUSDT', 'LUNAUSDT', 'GTCUSDT',
                                  'FTTUSDT', 'XLMUSDT', 'KNCUSDT', 'LINKUSDT', 'BALUSDT', 'ALGOUSDT', 'BELUSDT',
                                  'AVAXUSDT', 'ARUSDT', 'ETCUSDT', 'BTTUSDT', 'OMGUSDT', 'CRVUSDT', 'BNBUSDT',
                                  'XRPUSDT', 'RVNUSDT', 'CTKUSDT', 'YFIUSDT', 'ENJUSDT', 'BATUSDT', 'RSRUSDT',
                                  '1000LUNCUSDT', 'TOMOUSDT', 'XMRUSDT', 'TRBUSDT', 'SFPUSDT', 'DOTECOUSDT', 'BTCUSDT',
                                  'ANCUSDT', 'LPTUSDT', 'CTSIUSDT', 'SRMUSDT', 'ALICEUSDT', '1000BTTCUSDT', 'ETHUSDT',
                                  'WOOUSDT', 'STMXUSDT', 'MAGICUSDT', 'RUNEUSDT', 'REEFUSDT', 'LRCUSDT', 'AXSUSDT',
                                  'AAVEUSDT', 'ARPAUSDT', 'EGLDUSDT', 'DEFIUSDT', 'KEEPUSDT', 'FLOWUSDT', 'FLMUSDT',
                                  'IMXUSDT', 'VETUSDT', 'NUUSDT', 'INJUSDT', 'ONEUSDT', 'CHRUSDT', '1000XECUSDT',
                                  'SKLUSDT', 'BAKEUSDT', 'FILUSDT', 'SNXUSDT', 'HNTUSDT', 'KAVAUSDT', 'IOTAUSDT',
                                  'KLAYUSDT', 'BNXUSDT', 'OCEANUSDT', 'SCUSDT', 'OPUSDT', 'HOTUSDT', 'COMPUSDT',
                                  'BLUEBIRDUSDT', 'APEUSDT', '1INCHUSDT', 'SPELLUSDT', 'TRXUSDT', 'GMTUSDT', 'BLZUSDT',
                                  'LTCUSDT', 'XEMUSDT', 'LDOUSDT', 'COTIUSDT', 'DOGEUSDT', 'ICPUSDT', 'THETAUSDT',
                                  'NKNUSDT', 'LINAUSDT', 'GALAUSDT', 'FOOTBALLUSDT', 'DASHUSDT', 'DOTUSDT', 'AUDIOUSDT',
                                  'XTZUSDT', 'OGNUSDT', 'CVXUSDT', 'ZRXUSDT', 'FTMUSDT', 'ZECUSDT']
}

stats_calculation_constant = {'1D': [252, 15.8], '4H': [3 * 252, 27.5], '1H': [24 * 365, 93.59]}

# COVID-19 after spring festival (2022-02-03)
# ret of IC.CFE	IF.CFE IH.CFE is -10.12% -10.09% -9.59%
daily_strategy_future_delete_date_dict = {
    'IH.CFE': ['2020-02-03', '2020-02-04', '2020-02-05', '2020-02-06', '2020-02-07'],
    'IC.CFE': ['2020-02-03', '2020-02-04', '2020-02-05', '2020-02-06', '2020-02-07'],
    'IF.CFE': ['2020-02-03', '2020-02-04', '2020-02-05', '2020-02-06', '2020-02-07']}
