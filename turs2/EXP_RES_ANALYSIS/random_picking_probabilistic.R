rm(list=ls())
require(dplyr)
require(data.table)

folders = c("turs_uci")

read_exp_data = function(folders){
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
  return(d)
}

d = read_exp_data(folders)
d2 = read_exp_data(c("brs_uci", "turs_uci"))
require(dplyr)
dt2 = data.table(d2)
brs_res = dt2[,.(test_auc=mean(roc_auc_test)),by=.(data_name, alg)]
brs_res_wide = dcast(brs_res, data_name ~ alg, value.var="test_auc")
brs_res_wide = brs_res_wide[!is.na(brs_res_wide$turs),]

dt = data.table(d)

pvalues_all = list()
for(i in 1:nrow(d)){
  tmp = d$rules_p_value_permutations[i] %>%
    strsplit(split=",") %>% .[[1]]
  if(length(tmp) == 1){
    pvalues = strsplit(tmp, split="\\[")[[1]][2] %>% strsplit(split="\\]") %>% .[[1]] %>% as.numeric
  } else{
    pvalues = rep(0, length(tmp))
    pvalues[1] = tmp[1] %>% strsplit(split="\\[") %>% .[[1]] %>% .[2] %>%
      as.numeric()
    pvalues[length(pvalues)] = tmp[length(tmp)] %>% strsplit(split="\\]") %>% .[[1]] %>% .[1] %>%
      as.numeric()
    if(length(pvalues) > 2){
      pvalues[2:(length(pvalues)-1)] = as.numeric(tmp[2:(length(tmp)-1)])
    }
  }
  pvalues_all[[i]] = pvalues  
}
sig_Bonferroni = rep(F, length(pvalues_all))
sig_HonnBonferroni = rep(F, length(pvalues_all))
for(i in 1:length(pvalues_all)){
  sig_Bonferroni[i] = mean(pvalues_all[[i]] < (0.05 / length(pvalues_all[[i]])))
  
  ranks = rank(-pvalues_all[[i]], ties.method = "random")
  sig_HonnBonferroni[i] = mean(pvalues_all[[i]] < (0.05 / ranks))
  
}

for(i in 1:nrow(d)){
  if(d$data_name[i] == "14_glass.npz"){
    d$data_name[i] = "14_glass-2.npz"
  }
  
  if(d$data_name[i] == "41_Waveform.npz"){
    d$data_name[i] = "41_waveform-2.npz"
  }
  
  if(d$data_name[i] == "28_pendigits.npz"){
    d$data_name[i] = "28_pendigits-2.npz"
  }
  
  
  
  split_name = strsplit(d$data_name[i], split="_")[[1]]
  if(length(split_name) > 1){
    d$data_name[i] = strsplit(split_name[2], split=".npz")[[1]][1]
  }
}
d$data_name = tolower(d$data_name)

d2 = data.frame(data_name = d$data_name, logloss=d$logloss_test, random_picking=F)
d3 = data.frame(data_name = d$data_name, logloss=d$random_picking_logloss, random_picking=T)
dd = rbind(d2, d3)
ggplot(dd, aes(x=data_name, y=logloss, fill=random_picking)) +
  geom_col(position = "dodge") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(fill = "random picking\n for overlaps", x="", y="log loss")
ggsave("~/projects/TURS/turs2/paper_jmlr/logloss_randompicking.png",
       device = "png", width=20, height=7, units="cm", dpi=300)
