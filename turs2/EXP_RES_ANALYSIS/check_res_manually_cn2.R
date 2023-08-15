rm(list=ls())
require(dplyr)
require(data.table)
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/cn2/exp_uci_20230723/*.csv ./cn2_uci/")

folders = c("turs_uci", "cn2_uci", "cart_uci", "brs_uci", "classy_uci", "drs_uci", "ids_uci")

d = data.frame()

folder = "cn2_uci"
alg = strsplit(folder, split="_")[[1]][1]
files = list.files(folder)
for(f in files){
  ff = paste(folder, "/",  f, sep="")
  dd = read.csv(ff)
  if(folder == "cn2_uci" | folder =="brs_uci"){
    avg_rule_len = rep(0, nrow(dd))
    for(i in 1:nrow(dd)){
      char_vector = dd$avg_rule_length[i]
      num_vec = as.numeric(unlist(strsplit(substr(char_vector, 2, nchar(char_vector) - 1), ", ")))
      avg_rule_len[i] = mean(num_vec[1:(length(num_vec) - 1)])
    }
    dd$avg_rule_length=avg_rule_len
  }
  dd$alg = alg
  d = bind_rows(d, dd)
}  

dt = data.table(d)
dt$alg = NULL
dt$rules_p_value_permutations = NULL
# roc_auc
dt[, .(roc_auc_test = mean(roc_auc_test), roc_auc_train = mean(roc_auc_train)), by = .(data_name)]

# roc_auc_random_picking
means_by_group <- dt[, lapply(.SD, mean, na.rm=T), by = .(data_name)]

# ad bench results
folder = "~/projects/turs_supllementary_scripts/competitors/cn2/exp_adbench20230723/"
files = list.files(folder)
d_ad = data.frame()
for(f in files){
  ff = paste(folder, f, sep="")
  dd = read.csv(ff)
  d_ad = rbind(d_ad, dd)
}
d_ad$"data_name_fold" = paste(d_ad$data_name, "FOLD", as.character(d_ad$fold), sep="")
avg_rule_len = rep(0, nrow(d_ad))
for(i in 1:nrow(d_ad)){
  char_vector = d_ad$avg_rule_length[i]
  num_vec = as.numeric(unlist(strsplit(substr(char_vector, 2, nchar(char_vector) - 1), ", ")))
  avg_rule_len[i] = mean(num_vec[1:(length(num_vec) - 1)])
}
d_ad$avg_rule_length=avg_rule_len
dt_ad = data.table(d_ad)
dt_ad$data_name_fold = NULL
dt_ad$rules_p_value_permutations = NULL
means_ad <- dt_ad[, lapply(.SD, mean, na.rm=T), by = .(data_name)]
