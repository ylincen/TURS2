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
}

get_res_wide = function(value_var){
  res = dt[order(data_name),.(avg_lit=mean(avg_num_literals_for_each_datapoint),
                              rule_length=mean(avg_rule_length, na.rm=T),
                              num_rules=mean(num_rules),
                              roc_auc=mean(roc_auc_test, na.rm=T)),by=.(data_name, alg)]
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

get_total_lit = function(){
  num_rules = get_res_wide("num_rules")
  avg_rule_length = get_res_wide("rule_length")
  total_lit = num_rules[, 2:ncol(num_rules)] * avg_rule_length[, 2:ncol(avg_rule_length)]
  total_lit = cbind(num_rules[,1], total_lit)
}

# total_lit_long = melt(total_lit, id.vars = colnames(total_lit)[1],
#        value.name = "total_lit", variable.name = "algorithm",
#        measure.vars = colnames(total_lit)[2:length(total_lit)])

remove_those_with_bad_auc = function(total_lit){
  roc_auc = get_res_wide("roc_auc")
  roc_auc_long = melt(roc_auc, id.vars = colnames(roc_auc)[1],
                      value.name = "roc_auc", variable.name = "algorithm",
                      measure.vars = colnames(roc_auc)[2:length(roc_auc)])
  total_lit2 = data.frame(total_lit)
  for(i in 1:nrow(total_lit2)){
    use_or_not = ((roc_auc[i, 2:ncol(roc_auc)] - as.numeric(roc_auc[i, "turs"])) > -0.1)
    
    for(j in 2:ncol(total_lit2)){
      if(is.na(use_or_not[j-1])){
        next
      }
      if(!isTRUE(use_or_not[j-1])){
        total_lit2[i,j] = Inf
      }
    }
  }
  return(total_lit2)
}


get_best_num_lit = function(total_lit2){
  check_best_mat = matrix(rep(F, (ncol(total_lit2)-1) * nrow(total_lit2)),
                          nrow=nrow(total_lit2))
  for(i in 1:nrow(total_lit2)){
    check_best_mat[i,] = total_lit2[i, -1] == min(total_lit2[i, -1], na.rm=T)
  }
  check_best_mat[is.na(check_best_mat)] = F
  return(check_best_mat)
}


get_xtable = function(total_lit, check_best_mat, exclude_bad_auc=T){
  roc_auc = get_res_wide("roc_auc")
  
  total_lit = data.frame(total_lit)
  
  for(i in 2:ncol(total_lit)){
    total_lit[,i] = as.character(round(total_lit[,i],1))
  }
  for(i in 1:nrow(total_lit)){
    use_or_not = ((roc_auc[i, 2:ncol(roc_auc)] - as.numeric(roc_auc[i, "turs"])) > -0.1)
    # if(check_best[i]){
    #   total_lit[i, ncol(total_lit)] = 
    #     paste("textbf{{", total_lit[i, ncol(total_lit)], "}}", sep="")
    # }
    for(j in 2:ncol(total_lit)){
      if(is.na(use_or_not[j-1])){
        next
      }
      if(!isTRUE(use_or_not[j-1])){
        if(exclude_bad_auc){
          total_lit[i,j] = paste("tiny{{", as.character(total_lit[i,j]),"}}", sep="")  
        }
      } else{
        if(check_best_mat[i,j-1]){
          total_lit[i,j] = paste("textbf{{", total_lit[i, j], "}}", sep="")
        }
      }
    }
  }
  
  total_lit[is.na(total_lit)] = "---"
  # print(total_lit, digit=1)
  
  print(xtable(total_lit), include.rownames=F)
}

calculate_ratio_against_best = function(total_lit){
  total_lit = data.frame(total_lit)
  
  ratio_mat = data.frame(matrix(rep(0, nrow(total_lit) * ncol(total_lit)), ncol = ncol(total_lit)))
  ratio_mat[, 1] = total_lit[,1]
  for(i in 1:nrow(total_lit)){
    total_lit[i, is.infinite(as.numeric(total_lit[i,]))] = NA
    ratio_mat[i, -1] = min(total_lit[i, -1], na.rm=T) / total_lit[i, -1]
  }
  return(ratio_mat)
}

get_heat_map = function(total_lit2){
  colnames(total_lit2[-1]) = toupper(colnames(total_lit2[-1]))
  total_lit2 = data.table(total_lit2)
  total_lit_long2 = melt(total_lit2, id.vars = colnames(total_lit2)[1],
                         value.name = "total_lit", variable.name = "algorithm",
                         measure.vars = colnames(total_lit2)[2:ncol(total_lit2)])
  hmap1 = ggplot(data.frame(total_lit_long2)) + 
    geom_tile(aes(y=data_name, x=algorithm, fill=total_lit))+
    scale_fill_gradient(na.value="gray", low="red", high="green"
  ) + labs(fill="min. num. literals / num. literals \n[higher is better]", y="", x="") + 
    guides(color = guide_legend(direction = "horizontal", title.position = "top", label.position = "bottom")) +
    theme(legend.box = "horizontal", legend.position = "bottom")
    # theme(legend.title = element_text(angle = 270, hjust = 0.5, vjust = 0.5))
  # hmap2 = ggplot(data.frame(total_lit_long2[total_lit_long2$algorithm!= "c45" & 
  #                                             total_lit_long2$algorithm != "cart"])) + 
  #   geom_tile(aes(y=data_name, x=algorithm, fill=total_lit))+
  #   scale_fill_gradient(na.value="gray", low="green", high="red"
  #   )
  # return(list(hmap1=hmap1, hmap2=hmap2))  
  return(hmap1)
}

d = read_all_data(folders)
dt = data.table(d)

total_lit = get_total_lit()
total_lit_comparable_auc_only = remove_those_with_bad_auc(total_lit)

ratio_mat = calculate_ratio_against_best(total_lit)
colnames(ratio_mat) = colnames(total_lit)

ratio_mat_comparable_auc_only = calculate_ratio_against_best(total_lit_comparable_auc_only)
colnames(ratio_mat_comparable_auc_only) = colnames(total_lit)

hmap1 = get_heat_map(ratio_mat_comparable_auc_only)

require(xtable)
check_best_mat = get_best_num_lit(total_lit_comparable_auc_only)
get_xtable(total_lit, check_best_mat)

ggsave("~/projects/TURS/turs2/paper_jmlr/heatmap_model_complexity.png", hmap1,
       device = "png", width=20, height=15, units="cm", dpi=300)

rule_length_res = get_res_wide("rule_length")
check_best_mat_rl = get_best_num_lit(rule_length_res)

get_xtable(rule_length_res, check_best_mat_rl, exclude_bad_auc = T)

avg_lit_per_data = get_res_wide("avg_lit")
  
  
rule_length_res_excl_bad_auc = remove_those_with_bad_auc(rule_length_res)
rule_length_res_excl_bad_auc = data.table(rule_length_res_excl_bad_auc)
rule_length_res_long = 
  melt(rule_length_res_excl_bad_auc,
       id.vars = colnames(rule_length_res)[1],
       value.name = "rule_length", variable.name = "algorithm",
       measure.vars = colnames(rule_length_res)[2:ncol(rule_length_res)])
hmap2 = ggplot(data.frame(rule_length_res_long)) + 
  geom_tile(aes(y=data_name, x=algorithm, fill=rule_length))+
  scale_fill_gradient(na.value="gray", low="green", high="red",
                      trans="log10"
  ) + labs(fill="rule_lengths", y="", x="") + 
  guides(color = guide_legend(direction = "horizontal", title.position = "top", label.position = "bottom")) +
  theme(legend.box = "horizontal", legend.position = "bottom")
hmap2

ggplot(data.frame(rule_length_res_long)) + 
  geom_boxplot(aes(x=algorithm, y=rule_length))
