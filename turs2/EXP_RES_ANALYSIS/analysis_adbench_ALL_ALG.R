rm(list = ls())
require(data.table)
require(dplyr)
system(
  "scp -r rivium:/home/yangl3/projects/turs/turs2/exp_adbench_20230807/*.csv ./exp_adbench_20230807")

files = list.files("~/projects/TURS/turs2/EXP_RES_ANALYSIS/exp_adbench_20230807/")
d = data.frame()
for(f in files){
  ff = paste("~/projects/TURS/turs2/EXP_RES_ANALYSIS/exp_adbench_20230807/", f, sep="")
  dd = read.csv(ff)
  d = rbind(d, dd)
}
d$"data_name_fold" = paste(d$data_name, "FOLD", as.character(d$fold), sep="")
d_merge = d


d_c45 = read.csv("~/projects/turs_supllementary_scripts/competitors/c45/adbench_c45.csv")
colnames(d_c45)[16] = "data_name"
d_c45$fold = d_c45$fold - 1
d_c45$"data_name_fold" = paste(d_c45$data_name, "FOLD", as.character(d_c45$fold), sep="")
d_merge = merge(d_merge, d_c45, by="data_name_fold", all=T, suffixes = c("_turs", "_c45"))

d_cn2 = data.frame()
files = list.files("~/projects/turs_supllementary_scripts/competitors/cn2/exp_adbench20230723/")
for(f in files){
  ff = paste("~/projects/turs_supllementary_scripts/competitors/cn2/exp_adbench20230723/", f, sep="")
  dd = read.csv(ff)
  d_cn2 = rbind(d_cn2, dd)
}
d_cn2$"data_name_fold" = paste(d_cn2$data_name, "FOLD", as.character(d_cn2$fold), sep="")
colnames(d_cn2) = sapply(colnames(d_cn2), function(s){
  if(s != "data_name_fold"){
    paste(s, "_cn2", sep="")
  } else{
    s
  }
})
d_merge = merge(d_merge, d_cn2, by="data_name_fold", all=T)

d_ripper = read.csv("~/projects/turs_supllementary_scripts/competitors/ripper/adbench_ripper.csv")
d_ripper$fold = d_ripper$fold - 1
d_ripper$"data_name_fold" = paste(d_ripper$data, "FOLD", as.character(d_ripper$fold), sep="")
colnames(d_ripper) = sapply(colnames(d_ripper), function(s){
  if(s != "data_name_fold"){
    paste(s, "_ripper", sep="")
  } else{
    s
  }
})
d_merge = merge(d_merge, d_ripper, by="data_name_fold", all=T)


dt = data.table(d_merge)

res_roc_auc = dt[,.(TURS=mean(roc_auc_test_turs), C45=mean(roc_auc_test_c45),
      CN2 = mean(roc_auc_test_cn2),
      Ripper = mean(roc_auc_test_ripper)), by=.(data_name_c45)]
(res_roc_auc = res_roc_auc[-8][-10])

max_roc_auc = res_roc_auc[,2:ncol(res_roc_auc)] %>% as.matrix() %>% 
  apply(1, max)
diff_to_max = as.matrix(res_roc_auc[,2:ncol(res_roc_auc)] ) - max_roc_auc


apply(res_roc_auc[,2:ncol(res_roc_auc)], 1, rank) %>% apply(1, mean)

dt[,.(TURS_rp=mean(random_picking_roc_auc),
      CN2_rp = mean(random_picking_roc_auc_cn2)), by=.(data_name_c45)]
