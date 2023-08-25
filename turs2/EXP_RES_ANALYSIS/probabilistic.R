rm(list=ls())
require(dplyr)
require(data.table)
require(ggplot2)
folders = c("turs_uci", "cn2_uci", "cart_uci", "brs_uci", "classy_uci", "drs_uci", "ids_uci")
read_all_data = function(folders){
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
      if(folder == "drs_uci" | folder == "cart_uci" | folder == "ids_uci"){
        dd$Brier_test = dd$Brier_test / (dd$nrow * 0.8)
        dd$Brier_train = dd$Brier_train / (dd$nrow * 0.2)
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
  
  d$brier_test[is.na(d$brier_test)] = 
    d$Brier_test[is.na(d$brier_test)]
  d$Brier_test[is.na(d$Brier_test)] = 
    d$brier_test[is.na(d$Brier_test)]
  return(d)
}

get_res_wide = function(dt, value_var){
  res = dt[order(data_name),.(brier=mean(Brier_test, na.rm=T), 
                              logloss=mean(logloss_test), na.rm=T,
                              train_test_prob_diff=mean(train_test_prob_diff,na.rm=T)),by=.(data_name, alg)]
  res_wide = dcast(res, data_name ~ alg, value.var=value_var)
  
  for(i in 1:nrow(res_wide)){
    if(res_wide$data_name[i] == "14_glass.npz"){
      res_wide$data_name[i] = "14_glass-2.npz"
    }
    
    if(res_wide$data_name[i] == "41_Waveform.npz"){
      res_wide$data_name[i] = "41_waveform-2.npz"
    }
    
    if(res_wide$data_name[i] == "28_pendigits.npz"){
      res_wide$data_name[i] = "28_pendigits-2.npz"
    }
    
    
    
    split_name = strsplit(res_wide$data_name[i], split="_")[[1]]
    if(length(split_name) > 1){
      res_wide$data_name[i] = strsplit(split_name[2], split=".npz")[[1]][1]
    }
    
    
  }
  res_wide$data_name = tolower(res_wide$data_name)
  res_wide = res_wide[order(is.na(res_wide$brs), data_name)]
  res_wide = res_wide[!is.na(res_wide$turs)]
  desired_order = c("data_name", "brs", "c45", "cart", "classy", 
                    "ripper", "cn2", "drs", 
                    "ids", "turs")
  setcolorder(res_wide, desired_order)
  return(res_wide)
}


# res_wide_long = melt(res_wide, id.vars = colnames(res_wide)[1],
#        value.name = "res_wide", variable.name = "algorithm",
#        measure.vars = colnames(res_wide)[2:length(res_wide)])

remove_those_with_bad_auc = function(res_wide){
  roc_auc = get_res_wide("roc_auc")
  roc_auc_long = melt(roc_auc, id.vars = colnames(roc_auc)[1],
                      value.name = "roc_auc", variable.name = "algorithm",
                      measure.vars = colnames(roc_auc)[2:length(roc_auc)])
  res_wide = data.frame(res_wide)
  for(i in 1:nrow(res_wide)){
    use_or_not = ((roc_auc[i, 2:ncol(roc_auc)] - as.numeric(roc_auc[i, "turs"])) > -0.05)
    
    for(j in 2:ncol(res_wide)){
      if(is.na(use_or_not[j-1])){
        next
      }
      if(!isTRUE(use_or_not[j-1])){
        res_wide[i,j] = Inf
      }
    }
  }
  return(res_wide)
}


get_best_num_lit = function(res_wide){
  check_best_mat = matrix(rep(F, (ncol(res_wide)-1) * nrow(res_wide)),
                          nrow=nrow(res_wide))
  for(i in 1:nrow(res_wide)){
    check_best_mat[i,] = res_wide[i, -1] == min(res_wide[i, -1], na.rm=T)
  }
  check_best_mat[is.na(check_best_mat)] = F
  return(check_best_mat)
}


get_xtable = function(res_wide, check_best_mat){
  roc_auc = get_res_wide("roc_auc")
  
  res_wide = data.frame(res_wide)
  
  for(i in 2:ncol(res_wide)){
    res_wide[,i] = as.character(round(res_wide[,i]), 1)
  }
  for(i in 1:nrow(res_wide)){
    use_or_not = ((roc_auc[i, 2:ncol(roc_auc)] - as.numeric(roc_auc[i, "turs"])) > -0.05)
    # if(check_best[i]){
    #   res_wide[i, ncol(res_wide)] = 
    #     paste("textbf{{", res_wide[i, ncol(res_wide)], "}}", sep="")
    # }
    for(j in 2:ncol(res_wide)){
      if(is.na(use_or_not[j-1])){
        next
      }
      if(!isTRUE(use_or_not[j-1])){
        res_wide[i,j] = paste("scriptsize{{", as.character(res_wide[i,j]),"}}", sep="")
      } else{
        if(check_best_mat[i,j-1]){
          res_wide[i,j] = paste("textbf{{", res_wide[i, j], "}}", sep="")
        }
      }
    }
  }
  
  res_wide[is.na(res_wide)] = "---"
  print(res_wide, digit=1)
  
  print(xtable(res_wide), include.rownames=F)
}

d = read_all_data(folders)
dt = data.table(d)
# brier_res = get_res_wide(dt, "brier")
# 
# brier_res_long = 
#   melt(brier_res, id.vars = colnames(brier_res)[1],
#       value.name = "brier", variable.name = "algorithm",
#       measure.vars = colnames(brier_res)[2:length(brier_res)])
# 
# require(ggplot2)
# 
# qs = quantile(brier_res_long[brier_res_long$algorithm == "turs","brier"] %>%
#            unlist())
# ggplot(brier_res_long) + geom_boxplot(aes(x=algorithm, y=brier)) + 
#   geom_hline(yintercept = qs[2:4], lty=5)


res = get_res_wide(dt, "train_test_prob_diff")
get_res_long = function(res){
  res$data_name = factor(res$data_name, levels = res$data_name)
  
  res = data.frame(res)
  res_long = res[,1:2]
  colnames(res_long) = c("data_name", "res")
  res_long$algorithm = "brs"
  for(i in 3:ncol(res)){
    res_long = rbind(res_long, 
                     data.frame(data_name=res[,1],
                                res=res[,i],
                                algorithm=rep(colnames(res)[i], nrow(res))))
  }
  return(res_long)
}
res_long = get_res_long(res)
ggplot(res_long) + geom_boxplot(aes(x=algorithm, y=res))

# alphas = seq(0.4,1,length.out=9)[order(apply(res[,-1], 2, mean, na.rm=T), decreasing = T)]
# res_long$alphas = 0
# res_long = data.frame(res_long)
# for(i in 1:length(unique(res_long$algorithm))){
#   alg=unique(res_long$algorithm)[i]
#   res_long[res_long$algorithm == alg, "alphas"] = alphas[i]
# }

res_long = res_long[!is.na(res_long$res),]
gp=ggplot(res_long) + 
  geom_line(aes(y=res, x=data_name, col=algorithm,
               group=algorithm, size=algorithm,
               alpha=algorithm)) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  # scale_size_manual(values = c(brs=0.6,
  #                              c45=0.6, cart=0.6, classy=0.6, 
  #                              ripper=0.6, cn2=0.6, drs=0.6, ids=0.6,
  #                              turs=1.3)) + 
  scale_size_manual(values = c(rep(0.6, 8), 1.3)) +
  scale_alpha_manual(values = c(rep(1, 8), 0.8)) +
  labs(x="", y="train/test prob. diff.")
gp
ggsave("~/projects/TURS/turs2/paper_jmlr/train_test_diff.png", gp,
       device = "png", width=20, height=10, units="cm", dpi=300)

  
gp2 = ggplot(res_long[res_long$algorithm %in% c("turs", "ids", "drs", "cn2", "brs"), ]) + 
  geom_line(aes(y=res, x=data_name, col=algorithm,
                group=algorithm, size=algorithm,
                alpha=algorithm)) + 
  geom_point(aes(y=res, x=data_name, col=algorithm)) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  # scale_size_manual(values = c(brs=0.6,
  #                              c45=0.6, cart=0.6, classy=0.6, 
  #                              ripper=0.6, cn2=0.6, drs=0.6, ids=0.6,
  #                              turs=1.3)) + 
  scale_size_manual(values = c(rep(0.6, 4), 1.3)) +
  scale_alpha_manual(values = c(rep(1, 4), 0.8)) +
  labs(x="", y="train/test prob. diff.")
gp2
ggsave("~/projects/TURS/turs2/paper_jmlr/train_test_diff_rulesets.png", gp2,
       device = "png", width=20, height=10, units="cm", dpi=300)


gp3 = ggplot(res_long[res_long$algorithm %in% c("turs", "c45", "cart", "classy", "ripper"), ]) + 
  geom_line(aes(y=res, x=data_name, col=algorithm,
                group=algorithm, size=algorithm,
                alpha=algorithm)) + 
  geom_point(aes(y=res, x=data_name, col=algorithm)) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  # scale_size_manual(values = c(brs=0.6,
  #                              c45=0.6, cart=0.6, classy=0.6, 
  #                              ripper=0.6, cn2=0.6, drs=0.6, ids=0.6,
  #                              turs=1.3)) + 
  scale_size_manual(values = c(rep(0.6, 4), 1.3)) +
  scale_alpha_manual(values = c(rep(1, 4), 0.8)) +
  labs(x="", y="train/test prob. diff.")
gp3
ggsave("~/projects/TURS/turs2/paper_jmlr/train_test_diff_treeandlist.png", gp3,
       device = "png", width=20, height=10, units="cm", dpi=300)


best_each_data = rep(0, nrow(res))
for(i in 1:nrow(res)){
  best_each_data[i] = min(res[i, 2:ncol(res)], na.rm=T)
}
gaps_to_best = data.frame(res)
for(i in 1:nrow(gaps_to_best)){
  gaps_to_best[i, 2:ncol(gaps_to_best)] = 
    gaps_to_best[i, 2:ncol(gaps_to_best)] - best_each_data[i]  
}
gaps_to_best = data.table(gaps_to_best)
gaps_to_best_wide = gaps_to_best
gaps_to_best = melt(gaps_to_best, id.vars = colnames(gaps_to_best)[1],
                    value.name = "roc_auc", variable.name = "algorithm",
                    measure.vars = colnames(gaps_to_best)[2:length(gaps_to_best)])
gaps_to_best_wide = data.frame(gaps_to_best_wide)
turs_diff = gaps_to_best_wide[,ncol(gaps_to_best_wide)]

res = data.frame(res)
for(i in 2:ncol(res)){
  res[,i] = round(res[, i], 3)
}
res = apply(res, 2, as.character)
res[is.na(res)] = "---"
for(i in 1:nrow(res)){
  if(turs_diff[i] != 0){
    res[i ,ncol(res)] = 
      paste(res[i, ncol(res)], " tiny{{(", round(turs_diff[i], 3), ")}",
            sep="")  
  }
}

print(xtable(res, caption="Train test probability estimation difference"), include.rownames = F)



# logloss_res = get_res_wide(dt, "logloss")
# 
# logloss_res_long = 
#   melt(logloss_res, id.vars = colnames(logloss_res)[1],
#        value.name = "logloss", variable.name = "algorithm",
#        measure.vars = colnames(logloss_res)[2:length(logloss_res)])
# 
# require(ggplot2)
# 
# qs = quantile(logloss_res_long[logloss_res_long$algorithm == "turs","logloss"] %>%
#                 unlist())
# ggplot(logloss_res_long) + geom_boxplot(aes(x=algorithm, y=logloss)) 
  # geom_hline(yintercept = qs[2:4], lty=5)


# res_long = data.frame(res_long)
# ggplot(res_long[res_long$algorithm == "brs",]) + 
#   geom_point(aes(x = data_name, y=res), na.rm=T) +
#   geom_line(aes(x = data_name, y=res, group=1), na.rm=T)
  
                      
