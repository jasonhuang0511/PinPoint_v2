import warnings

import data.SQL.extract_data_from_postgre as ExtractDataPostgre
import model.constants.futures as ConstFutBasic
import model.constants.path as ConstPath

warnings.filterwarnings('ignore')

tickers_list = ConstFutBasic.fut_code_list
start_date = '2018-01-01'
end_date = '2022-09-30'

# predict percent
pct = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                        key_word='close', freq='D', index=1, ret_index=True)
pct['Fut_code'] = pct['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
pct = pct.pivot_table(index='Trade_DT', columns='Fut_code', values=pct.columns[2])
pct.to_csv(ConstPath.input_data_return_pct_path + 'freq_1D_close_ret.csv')

# basic price volume factor

close = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                          key_word='close', freq='D', index=1, ret_index=False)
close['Fut_code'] = close['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
close = close.pivot_table(index='Trade_DT', columns='Fut_code', values=close.columns[2])
close.to_csv(ConstPath.input_data_feature_path + 'freq_1D_close.csv')

open = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                         key_word='open', freq='D', index=1, ret_index=False)
open['Fut_code'] = open['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
open = open.pivot_table(index='Trade_DT', columns='Fut_code', values=open.columns[2])
open.to_csv(ConstPath.input_data_feature_path + 'freq_1D_open.csv')

high = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                         key_word='high', freq='D', index=1, ret_index=False)
high['Fut_code'] = high['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
high = high.pivot_table(index='Trade_DT', columns='Fut_code', values=high.columns[2])
high.to_csv(ConstPath.input_data_feature_path + 'freq_1D_high.csv')

low = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date, key_word='low',
                                        freq='D', index=1, ret_index=False)
low['Fut_code'] = low['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
low = low.pivot_table(index='Trade_DT', columns='Fut_code', values=low.columns[2])
low.to_csv(ConstPath.input_data_feature_path + 'freq_1D_low.csv')

settle = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                           key_word='settle',
                                           freq='D', index=1, ret_index=False)
settle['Fut_code'] = settle['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
settle = settle.pivot_table(index='Trade_DT', columns='Fut_code', values=settle.columns[2])
settle.to_csv(ConstPath.input_data_feature_path + 'freq_1D_settle.csv')

volume = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                           key_word='vol', freq='D', index=1, ret_index=False)
volume['Fut_code'] = volume['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
volume = volume.pivot_table(index='Trade_DT', columns='Fut_code', values=volume.columns[2])
volume.to_csv(ConstPath.input_data_feature_path + 'freq_1D_volume.csv')

oi = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                       key_word='openinterest', freq='D', index=1, ret_index=False)
oi['Fut_code'] = oi['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
oi = oi.pivot_table(index='Trade_DT', columns='Fut_code', values=oi.columns[2])

oi.to_csv(ConstPath.input_data_feature_path + 'freq_1D_oi.csv')

vwap_amount = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                key_word=['amount'], freq='D', index=1, ret_index=False)
vwap_amount['Fut_code'] = vwap_amount['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
vwap_amount['bpv'] = vwap_amount['Fut_code'].map(ConstFutBasic.fut_code_bpv)
vwap_amount['amount'] = vwap_amount['amount'] / vwap_amount['bpv'] * 10000
vwap_amount = vwap_amount.pivot_table(index='Trade_DT', columns='Fut_code', values=vwap_amount.columns[2])
vwap_volume = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                key_word='vol', freq='D', index=1, ret_index=False)
vwap_volume['Fut_code'] = vwap_volume['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
vwap_volume = vwap_volume.pivot_table(index='Trade_DT', columns='Fut_code', values=vwap_volume.columns[2])

vwap = vwap_amount / vwap_volume

vwap.to_csv(ConstPath.input_data_feature_path + 'freq_1D_vwap.csv')

amount = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                           key_word='amount', freq='D', index=1, ret_index=False)
amount['Fut_code'] = amount['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
amount = amount.pivot_table(index='Trade_DT', columns='Fut_code', values=amount.columns[2])

amount.to_csv(ConstPath.input_data_feature_path + 'freq_1D_amount.csv')

carry_pct = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                              key_word='annualized_carry_rate_nxt_main', freq='D', index=1,
                                              ret_index=False)
carry_pct['Fut_code'] = carry_pct['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
carry_pct = carry_pct.pivot_table(index='Trade_DT', columns='Fut_code', values=carry_pct.columns[2])

carry_pct.to_csv(ConstPath.input_data_feature_path + 'freq_1D_carry_pct.csv')

historical_vol = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                                   key_word='historical_volatility', freq='D', index=1, ret_index=False)
historical_vol['Fut_code'] = historical_vol['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
historical_vol = historical_vol.pivot_table(index='Trade_DT', columns='Fut_code', values=historical_vol.columns[2])

historical_vol.to_csv(ConstPath.input_data_feature_path + 'freq_1D_historical_vol.csv')

dastd = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                          key_word='dastd', freq='D', index=1, ret_index=False)
dastd['Fut_code'] = dastd['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
dastd = dastd.pivot_table(index='Trade_DT', columns='Fut_code', values=dastd.columns[2])

dastd.to_csv(ConstPath.input_data_feature_path + 'freq_1D_dastd.csv')

# pct 1 close T / close T-1  -1
# pct 2 open T / open T-1    -1
# pct 3  high T / low T      -1
# pct 4  close T / open T    -1

pct1 = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                         key_word='close', freq='D', index=1, ret_index=True)
pct1['Fut_code'] = pct1['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
pct1 = pct1.pivot_table(index='Trade_DT', columns='Fut_code', values=pct1.columns[2])

pct1.to_csv(ConstPath.input_data_feature_path + 'freq_1D_pct1.csv')

pct2 = ExtractDataPostgre.get_syn_con_ts(tickers=tickers_list, start_date=start_date, end_date=end_date,
                                         key_word='open', freq='D', index=1, ret_index=True)
pct2['Fut_code'] = pct2['Code'].map(ExtractDataPostgre.get_code_instrument_mapping())
pct2 = pct2.pivot_table(index='Trade_DT', columns='Fut_code', values=pct2.columns[2])

pct2.to_csv(ConstPath.input_data_feature_path + 'freq_1D_pct2.csv')

pct3 = high.div(low) - 1
pct3.to_csv(ConstPath.input_data_feature_path + 'freq_1D_pct3.csv')
pct4 = close.div(open) - 1
pct4.to_csv(ConstPath.input_data_feature_path + 'freq_1D_pct4.csv')
