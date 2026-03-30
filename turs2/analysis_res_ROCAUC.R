rm(list=ls())
require(data.table)

d1 = read.csv("~/projects/TURS/turs2/res.csv")
d1$"alg" = "TURS2"
dt1 = data.table(d1)
ddt = dt1[,.(auc_turs2=mean(roc_auc)),by=.(data, alg)]
ddt = ddt[,.(data, auc_turs2)]

d2 = read.csv("~/projects/TURS/turs2/ad_bench_competitors/res_CART_withCV.csv")
dt2 = data.table(d2)
ddt2 = dt2[,.(auc_CART=mean(roc_auc)),by=.(data, alg)]
ddt2 = ddt2[,.(data, auc_CART)]
ddt = ddt[ddt2, on='data']

d4 = read.csv("~/projects/TURS/turs2/ad_bench_competitors/RIPPER/res_RIPPER.csv")
d4$"alg" = "RIPPER"
colnames(d4) = c( "X","data","time","roc_auc","pr_auc","f1", "alg")
d4 = rbind(d4, d2[d2$data == "3_backdoor.npz",])
d4[d4$data == "3_backdoor.npz", c("roc_auc", "pr_auc", "f1")] =NA
d4[d4$data == "3_backdoor.npz", "alg"] = "RIPPER"
dt4 = data.table(d4)
ddt4 = dt4[,.(auc_RIPPER=mean(roc_auc)),by=.(data, alg)]
ddt4 = ddt4[,.(data, auc_RIPPER)]
ddt = ddt[ddt4, on='data']

d5 = read.csv("~/projects/TURS/turs2/ad_bench_competitors/RIPPER/res_PART.csv")
d5$"alg" = "PART"
colnames(d5) = c( "X","data","time","roc_auc","pr_auc","f1", "alg")
dt5 = data.table(d5)
ddt5 = dt5[,.(auc_PART=mean(roc_auc)),by=.(data, alg)]
ddt5 = ddt5[,.(data, auc_PART)]
ddt = ddt[ddt5, on='data']

d_description = read.csv("~/projects/TURS/turs2/ad_bench_competitors/adbench_data_description.csv")
d_description = d_description[,2:ncol(d_description)]
dt_description = data.table(d_description)

ddt = ddt[dt_description, on='data']

ddt = ddt[data != "34_smtp.npz"] # this dataset has discrete-continuous mixture features

auc_diff = rep(0, nrow(ddt))
for(i in 1:nrow(ddt)){
  auc_diff[i] = ddt$auc_turs2[i] - max(ddt$auc_RIPPER[i], ddt$auc_PART[i], na.rm = T)
}


