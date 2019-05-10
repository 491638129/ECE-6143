# -*- coding: utf-8 -*-
lending_useless_cols = [
    'id',
    'next_pymnt_d',  # high null rate
    'mths_since_last_record',  # high null rate
    'desc',  # high null rate
    'mths_since_last_major_derog',  # high null rate
    'mths_since_last_delinq',  # high null rate
    'pymnt_plan',  # single class
    'application_type',  # single class
    'disbursement_method',  # single class
    'installment',
    'grade',
    'funded_amnt',
    'out_prncp',
    'total_pymnt'

]
lending_fill_cols = [
    'avg_cur_bal',
    'total_rev_hi_lim',
    'tot_cur_bal',
    'tot_coll_amt',
    'emp_title',
    'emp_length',
    'bc_util',
    'bc_open_to_buy',
    'acc_open_past_24mths',
    'last_pymnt_d',
    'revol_util',
    'last_credit_pull_d',
    'title'
]
lending_cate_cols = [
    'sub_grade', 'emp_title',
    'emp_length', 'home_ownership', 'verification_status', 'term',
    'loan_status', 'purpose',
    'title', 'zip_code', 'addr_state', 'initial_list_status',
    'issue_d', 'last_pymnt_d',
    'earliest_cr_line',
    'last_credit_pull_d'
]
lending_num_cols = [
    'annual_inc', 'loan_amnt',
    'funded_amnt_inv',
    'int_rate',
    'dti', 'inq_last_6mths',
    'open_acc', 'pub_rec',
    'revol_bal', 'revol_util',
    'total_acc',
    'out_prncp_inv',
    'total_pymnt_inv', 'total_rec_prncp',
    'total_rec_int', 'total_rec_late_fee',
    'recoveries', 'collection_recovery_fee',
    'last_pymnt_amnt', 'collections_12_mths_ex_med', 'acc_now_delinq',
    'tot_coll_amt', 'tot_cur_bal',
    'total_rev_hi_lim', 'acc_open_past_24mths',
    'avg_cur_bal', 'bc_open_to_buy',
    'bc_util', 'chargeoff_within_12_mths',
    'delinq_amnt', 'pub_rec_bankruptcies',
    'tax_liens'
]
lending_lower_cols = [
    'title', 'emp_title',
]
