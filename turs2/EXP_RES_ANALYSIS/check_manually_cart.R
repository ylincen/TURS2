rm(list = ls())
folder = "./cart_uci/"

d = data.frame()
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
