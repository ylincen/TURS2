rm(list=ls())
require(dplyr)
require(data.table)
# system("scp -r rivium:/home/yangl3/projects/turs/turs2/exp_uci_20230807/*.csv ./turs_uci/")
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/cn2/exp_uci_20230723/*.csv ./cn2_uci/")
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/cart/exp_uci_20230809/*.csv ./cart_uci/")
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/BOA/exp_uci_20230728/*.csv ./brs_uci/")
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/classy/exp_uci_20230810/*.csv ./classy_uci/")
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/drs/exp_uci_20230809/*.csv ./drs_uci/")
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/ids/pyIDS/exp_uci_20230809/*.csv ./ids_uci/")

folders = c("turs_uci", "cn2_uci", "cart_uci", "brs_uci", "classy_uci", "drs_uci", "ids_uci")

d = data.frame()
for(folder in folders){
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
}

d_turs = d[d$alg == "turs",]

dt_turs = data.table(d_turs)
dt_turs$alg = NULL
dt_turs$rules_p_value_permutations = NULL
# roc_auc
dt_turs[, .(roc_auc_test = mean(roc_auc_test), roc_auc_train = mean(roc_auc_train)), by = .(data_name)]

# roc_auc_random_picking
means_by_group <- dt_turs[, lapply(.SD, mean, na.rm=T), by = .(data_name)]

# ad bench results

system("scp -r rivium:/home/yangl3/projects/turs/turs2/exp_adbench_20230807/*.csv ./exp_adbench_20230807_tmp/")
files = list.files("~/projects/TURS/turs2/EXP_RES_ANALYSIS/exp_adbench_20230807_tmp/")
d_ad = data.frame()
for(f in files){
  ff = paste("~/projects/TURS/turs2/EXP_RES_ANALYSIS/exp_adbench_20230807_tmp/", f, sep="")
  dd = read.csv(ff)
  d_ad = rbind(d_ad, dd)
}
d_ad$"data_name_fold" = paste(d_ad$data_name, "FOLD", as.character(d_ad$fold), sep="")
dt_ad = data.table(d_ad)
dt_ad$data_name_fold = NULL
dt_ad$rules_p_value_permutations = NULL
means_ad <- dt_ad[, lapply(.SD, mean, na.rm=T), by = .(data_name)]
