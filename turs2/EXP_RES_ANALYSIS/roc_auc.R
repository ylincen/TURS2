rm(list=ls())
require(dplyr)
require(data.table)
## FOR TURS
# system("scp -r rivium:/home/yangl3/projects/turs/turs2/exp_uci_20230807/*.csv ./turs_uci/")
# system("scp -r rivium:/home/yangl3/projects/turs/turs2/exp_uci_20230811/*.csv ./turs_uci/")
# system("scp -r rivium:/home/yangl3/projects/turs/turs2/exp_uci_20230811/*.csv ./turs_uci/")
# system("scp -r rivium:/home/yangl3/projects/turs/turs2/exp_adbench_20230814/*.csv ./turs_uci/")

## CN2
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/cn2/exp_uci_20230723/*.csv ./cn2_uci/")
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/cn2/exp_uci_20230811/*.csv ./cn2_uci/")
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/cn2/exp_adbench20230723/*.csv ./cn2_uci/")

## CART
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/cart/exp_uci_20230809/*.csv ./cart_uci/")
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/cart/exp_uci_20230811/*.csv ./cart_uci/")

## BRS
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/BOA/exp_uci_20230728/*.csv ./brs_uci/")

## CLASSY
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/classy/exp_uci_20230810/*.csv ./classy_uci/")
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/classy/exp_uci_20230811/*.csv ./classy_uci/")

## DRS
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/drs/exp_uci_20230809/*.csv ./drs_uci/")
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/drs/exp_uci_20230811/*.csv ./drs_uci/")

## IDS
# system("scp -r rivium:/home/yangl3/projects/turs_supplementary/competitors/ids/pyIDS/exp_uci_20230815/*.csv ./ids_uci/")

folders = c("turs_uci", "cn2_uci", "cart_uci", "brs_uci", "classy_uci", "drs_uci", "ids_uci")

d = data.frame()
for(folder in folders){
  alg = strsplit(folder, split="_")[[1]][1]
  files = list.files(folder)
  for(f in files){
    ff = paste(folder, "/",  f, sep="")
    try(
      {dd = read.csv(ff)}, silent = T
    )
    if(folder == "cn2_uci" | folder =="brs_uci"){
      avg_rule_len = rep(0, nrow(dd))
      for(i in 1:nrow(dd)){
        char_vector = dd$avg_rule_length[i]
        num_vec = as.numeric(unlist(strsplit(substr(char_vector, 2, nchar(char_vector) - 1), ", ")))
        avg_rule_len[i] = mean(num_vec[1:(length(num_vec) - 1)])
      }
      dd$avg_rule_length=avg_rule_len
    }
    if(folder == "drs" | folder == "cart" | folder == "ids"){
      d$Brier_test = d$Brier_test / (d$nrow * 0.8)
      d$Brier_train = d$Brier_train / (d$nrow * 0.2)
    }
    dd$alg = alg
    d = bind_rows(d, dd)
  }  
}


dd = read.csv("~/projects/turs_supllementary_scripts/competitors/ripper/uci_ripper.csv")
colnames(dd)[21] = "data_name"
dd$alg = "ripper"
dd$rules_p_value_permutations = ""  # tmp solution; check the code
d = bind_rows(d, dd)
dd = read.csv("~/projects/turs_supllementary_scripts/competitors/ripper/uci_ripper_extra.csv")
colnames(dd)[21] = "data_name"
dd$alg = "ripper"
dd$rules_p_value_permutations = ""  # tmp solution; check the code
d = bind_rows(d, dd)
dd = read.csv("~/projects/turs_supllementary_scripts/competitors/ripper/adbench_ripper.csv")
colnames(dd)[21] = "data_name"
dd$alg = "ripper"
dd$rules_p_value_permutations = ""  # tmp solution; check the code
d = bind_rows(d, dd)
dd = read.csv("~/projects/turs_supllementary_scripts/competitors/ripper/adbench_ripper_extra.csv")
colnames(dd)[21] = "data_name"
dd$alg = "ripper"
dd$rules_p_value_permutations = ""  # tmp solution; check the code
d = bind_rows(d, dd)

dd = read.csv("~/projects/turs_supllementary_scripts/competitors/c45/uci_c45.csv")
dd$alg = "c45"
dd$rules_p_value_permutations = ""  # tmp solution; check the code
colnames(dd)[16] = "data_name"
d = bind_rows(d, dd)
dd = read.csv("~/projects/turs_supllementary_scripts/competitors/c45/uci_c45_extra.csv")
dd$alg = "c45"
dd$rules_p_value_permutations = ""  # tmp solution; check the code
colnames(dd)[16] = "data_name"
d = bind_rows(d, dd)
dd = read.csv("~/projects/turs_supllementary_scripts/competitors/c45/adbench_c45.csv")
dd$alg = "c45"
dd$rules_p_value_permutations = ""  # tmp solution; check the code
colnames(dd)[16] = "data_name"
d = bind_rows(d, dd)
dd = read.csv("~/projects/turs_supllementary_scripts/competitors/c45/adbench_c45_backdoor_and_waveform2.csv")
dd$alg = "c45"
dd$rules_p_value_permutations = ""  # tmp solution; check the code
colnames(dd)[16] = "data_name"
d = bind_rows(d, dd)



dt = data.table(d)
auc_res = dt[order(data_name),.(test_auc=mean(roc_auc_test, na.rm=T), 
                                test_rd_auc=mean(random_picking_roc_auc), 
                                train_auc=mean(roc_auc_train)),by=.(data_name, alg)]
auc_res_wide = dcast(auc_res, data_name ~ alg, value.var="test_auc")
auc_res_wide[, (names(auc_res_wide)[-1]) := lapply(.SD, round, 4), .SDcols = -1]
auc_res_wide = auc_res_wide[!is.na(auc_res_wide$turs) ]

for(i in 1:nrow(auc_res_wide)){
  if(auc_res_wide$data_name[i] == "14_glass.npz"){
    auc_res_wide$data_name[i] = "14_glassanomaly.npz"
  }
  
  if(auc_res_wide$data_name[i] == "41_Waveform.npz"){
    auc_res_wide$data_name[i] = "41_waveform2.npz"
  }
  split_name = strsplit(auc_res_wide$data_name[i], split="_")[[1]]
  if(length(split_name) > 1){
    auc_res_wide$data_name[i] = strsplit(split_name[2], split=".npz")[[1]][1]
  }
  
  
}
auc_res_wide$data_name = tolower(auc_res_wide$data_name)
auc_res_wide = auc_res_wide[order(is.na(auc_res_wide$brs), data_name)]
desired_order = c("data_name", "brs", "c45", "cart", "classy", 
                  "ripper", "cn2", "drs", 
                  "ids", "turs")
setcolorder(auc_res_wide, desired_order)

best_each_data = rep(0, nrow(auc_res_wide))
for(i in 1:nrow(auc_res_wide)){
  best_each_data[i] = max(auc_res_wide[i, 2:ncol(auc_res_wide)], na.rm=T)
}

gaps_to_best = data.frame(auc_res_wide)
for(i in 1:nrow(gaps_to_best)){
  gaps_to_best[i, 2:ncol(gaps_to_best)] = 
    gaps_to_best[i, 2:ncol(gaps_to_best)] - best_each_data[i]  
}
gaps_to_best = data.table(gaps_to_best)
gaps_to_best = melt(gaps_to_best, id.vars = colnames(gaps_to_best)[1],
     value.name = "roc_auc", variable.name = "algorithm",
     measure.vars = colnames(gaps_to_best)[2:length(gaps_to_best)])
ggplot(gaps_to_best) + geom_boxplot(aes(x=algorithm, y=roc_auc)) + 
  labs(x="Algorithm", y="Diff to the best ROC-AUC") 

ggplot(gaps_to_best) + geom_point(aes(x=algorithm, y=roc_auc)) +
  labs(x="Algorithms", y="Diff to the best ROC-AUC") 




