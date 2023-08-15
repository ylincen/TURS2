rm(list=ls())
system("scp -r rivium:/home/yangl3/projects/turs/turs2/exp_uci_20230807/*.csv ./exp_uci_20230807/")
files = list.files("./exp_uci_20230807/")

d = data.frame()
for(f in files){
  ff = paste("./exp_uci_20230807/", f, sep="")
  dd = read.csv(ff)
  d = rbind(d, dd)
}

require(data.table)

dt = data.table(d)

dt[order(data_name),.(test_auc=mean(roc_auc_test), 
      test_rd_auc=mean(random_picking_roc_auc), 
      train_auc=mean(roc_auc_train)),by=.(data_name)]

dt[order(data_name),.(num_rules=mean(num_rules), 
      avg_rule_len=mean(avg_rule_length),
      overlap_perc=mean(overlap_perc), 
      explainability=mean(avg_num_literals_for_each_datapoint)), by=.(data_name)]

dt[,.(tr_ts_prob_diff=mean(train_test_prob_diff),
      rule_sig_perc=mean(rules_p_value_permutations_significance_perc),
      overlap_sig_perc=mean(overlap_significances_perc,na.rm=T)), by=.(data_name)]

dt[order(data_name),.(brier_score_test=mean(Brier_test)),by=.(data_name)]
