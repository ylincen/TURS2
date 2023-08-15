rm(list=ls())
require(dplyr)
require(data.table)

folders = c("turs_uci", "cn2_uci", "drs_uci", "ids_uci")

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


dt = data.table(d)
auc_res = dt[order(data_name),
             .(test_auc=mean(roc_auc_test), 
              test_rd_auc=mean(random_picking_roc_auc),
              overlap_perc=mean(overlap_perc)),
             by=.(data_name, alg)]
auc_res$overlap_perc = round(auc_res$overlap_perc, 2)

auc_res_wide1 = dcast(auc_res, data_name ~ alg, value.var="test_auc")
auc_res_wide2 = dcast(auc_res, data_name ~ alg, value.var="test_rd_auc")
auc_res_wide3 = dcast(auc_res, data_name ~ alg, value.var="overlap_perc")
auc_res_wide = cbind(auc_res_wide1, auc_res_wide2, auc_res_wide3)[,c(1:5, 7:10, 12:15)]
auc_res_wide = auc_res_wide[!is.na(auc_res_wide$turs) ]

colnames(auc_res_wide)[6:9] = paste(colnames(auc_res_wide)[6:9], "_RP", sep="")
colnames(auc_res_wide)[10:13] = paste(colnames(auc_res_wide)[10:13], " %overlap", sep="")

desired_order = c("data_name","turs", "turs_RP", "turs %overlap", 
                  "cn2", "cn2_RP", "cn2 %overlap", 
                  "drs", "drs_RP", "drs %overlap", 
                  "ids", "ids_RP", "ids %overlap")
setcolorder(auc_res_wide, desired_order)

print(data.frame(auc_res_wide)[,1:7],digits=4) 






