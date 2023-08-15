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

dd = read.csv("~/projects/turs_supllementary_scripts/competitors/c45/uci_c45.csv")
dd$alg = "c45"
dd$rules_p_value_permutations = ""  # tmp solution; check the code
colnames(dd)[16] = "data_name"
d = bind_rows(d, dd)


dt = data.table(d)
auc_res = dt[order(data_name),.(test_auc=mean(roc_auc_test), 
                      test_rd_auc=mean(random_picking_roc_auc), 
                      train_auc=mean(roc_auc_train)),by=.(data_name, alg)]
auc_res_wide = dcast(auc_res, data_name ~ alg, value.var="test_auc")
uci_auc = auc_res_wide[16:nrow(auc_res_wide),]
uci_auc$turs_diff_to_best = uci_auc$turs - apply(uci_auc[,2:ncol(uci_auc)], 1, max, na.rm=T)


run_time = dt[order(data_name),.(runtime=runtime),by=.(data_name, alg)]
run_time_wide = dcast(run_time, data_name ~ alg, value.var="runtime")
