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
  res = dt[order(data_name),.(runtime=mean(runtime,na.rm=T),
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


dd = d[d$alg %in% c("drs",  "turs"),c("data_name","nrow", "alg",
                         "ncol", "runtime", 
                         "num_rules")]
dd$ne = dd$nrow * dd$ncol

ddt = data.table(dd)
ddt = ddt[,.(nrow=mean(nrow), ncol=mean(ncol), num_rules=mean(num_rules),
             ne = mean(nrow * ncol), alg=alg, runtime=mean(runtime)), by=.(data_name)]



runtime_res = dcast(dt[,.(nrow=mean(nrow), ncol=mean(ncol), num_rules=mean(num_rules),
       ne = mean(nrow * ncol), runtime=mean(runtime)), by=.(data_name, alg=alg)], 
 data_name ~ alg, value.var="runtime")
runtime_res=runtime_res[,c("data_name","cn2", "drs", "ids", "brs", "turs")]

for(i in 1:nrow(runtime_res)){
  if(runtime_res$data_name[i] == "14_glass.npz"){
    runtime_res$data_name[i] = "14_glass-2.npz"
  }
  
  if(runtime_res$data_name[i] == "41_Waveform.npz"){
    runtime_res$data_name[i] = "41_waveform-2.npz"
  }
  
  if(runtime_res$data_name[i] == "28_pendigits.npz"){
    runtime_res$data_name[i] = "28_pendigits-2.npz"
  }
  
  split_name = strsplit(runtime_res$data_name[i], split="_")[[1]]
  if(length(split_name) > 1){
    runtime_res$data_name[i] = strsplit(split_name[2], split=".npz")[[1]][1]
  }
}
runtime_res$data_name = tolower(runtime_res$data_name)

runtime_res = runtime_res[order(is.na(runtime_res$brs), data_name)]
runtime_res$data_name = factor(runtime_res$data_name, levels = runtime_res$data_nam)

runtime_res = runtime_res[!is.na(runtime_res$turs),]

runtime_res_long = melt(
  runtime_res, 
  id.vars = colnames(runtime_res)[1],
  value.name = "runtime", variable.name = "alg",
  measure.vars = colnames(runtime_res)[2:length(runtime_res)])

ggplot(runtime_res_long) + geom_point(aes(x=data_name, y=runtime + 1, col=alg)) +
  geom_line(aes(x=data_name, y=runtime + 1, col=alg, group=alg)) + 
  scale_y_continuous(trans="log10") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(y = "runtime (seconds), log10 scale", x = "")
ggsave("~/projects/TURS/turs2/paper_jmlr/runtime.png",
       device = "png", width=20, height=10, units="cm", dpi=300)
  




