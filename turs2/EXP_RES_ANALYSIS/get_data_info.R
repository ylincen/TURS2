rm(list=ls())

datasets_without_header_row = c("chess", "iris", "waveform", "backnote", "contracept", "ionosphere",
                                "magic", "car", "tic-tac-toe", "wine")
datasets_with_header_row = c("avila", "anuran", "diabetes")

data_names = c("chess", "iris", "waveform", "backnote", "contracept", "ionosphere",
               "magic", "car", "tic-tac-toe", "wine",
               "avila", "anuran", "diabetes")
datasets_with_header_row =  c(datasets_with_header_row, c("Vehicle", "DryBeans"))
datasets_without_header_row =  c(datasets_without_header_row, c("glass", "pendigits", "HeartCleveland"))
data_names = c(datasets_without_header_row, datasets_with_header_row)

num_cols = rep(0, length(data_names))
num_rows = rep(0, length(data_names))
num_classes = rep(0, length(data_names))
class_imbalance_entropy = rep(0, length(data_names))
max_prob = rep(0, length(data_names))
min_prob = rep(0, length(data_names))
for(i in 1:length(data_names)){
  data_name = data_names[i]
  print(data_name)
  if(data_name %in% datasets_without_header_row){
    header=F
  } else{
    header=T
  }
  data_path = paste("~/projects/TURS/datasets/", data_name, ".csv", sep="")
  d = read.csv(data_path, header = header)
  num_cols[i] = ncol(d)
  num_rows[i] = nrow(d)
  num_classes[i] = length(table(d[,ncol(d)]))
  p = table(d[,ncol(d)]) / nrow(d)
  class_imbalance_entropy[i] = sum(-log(p) * p)
  max_prob[i] = max(p)
  min_prob[i] = min(p)
}

data_info = data.frame(data_name = data_names, 
                       num_rows=num_rows, num_cols=num_cols, 
                       num_classes=num_classes, 
                       max_class_prob=round(max_prob, 4),
                       min_class_prob=round(min_prob, 4))



if (!require(reticulate)) install.packages('reticulate')
if (!require(caret)) install.packages('caret')
library(caret)

if (!require(R.utils)) install.packages('R.utils')

library(R.utils)

library(reticulate)
np <- import('numpy')



require(RWeka)
require(dplyr)

read_npz <- function(file_path) {
  # Read the npz file
  npz_file <- np$load(file_path, allow_pickle = TRUE)
  
  # Convert the npz file to a list
  file_list <- npz_file$files
  
  # Create an empty list to store the arrays
  array_list <- list()
  
  # Loop over the list to extract each array
  for (i in file_list) {
    array_list[[i]] <- npz_file[[i]]
  }
  
  # Return the list of arrays
  return(array_list)
}

# selected_data = list()
select_files = function(files){
  counter = 1
  for(f in files){
    print(f)
    path = paste("../../dataset/adbench/", f, sep="")
    data = read_npz(path)
    x=data$x
    y=data$y
    
    if(length(y) > 1e5) next()
    
    if(min(c(mean(y == 0), mean(y==1))) > 0.05) next()
    
    selected_data[[counter]] = f
    counter = counter + 1
  }
  unlist(selected_data)
}

# files = list.files("../../dataset/adbench/")
# selected_data = select_files(files)

data_names <- c("1_ALOI.npz", "14_glass.npz", "21_Lymphography.npz", "23_mammography.npz",
                "25_musk.npz", "26_optdigits.npz", "28_pendigits.npz", "3_backdoor.npz",
                "31_satimage-2.npz", "34_smtp.npz", "36_speech.npz", "38_thyroid.npz",
                "40_vowels.npz","41_Waveform.npz", "42_WBC.npz", "43_WDBC.npz")

num_cols = rep(0, length(data_names))
num_rows = rep(0, length(data_names))
num_classes = rep(0, length(data_names))
class_imbalance_entropy = rep(0, length(data_names))
max_prob = rep(0, length(data_names))
min_prob = rep(0, length(data_names))
for(i in 1:length(data_names)){
  data_name = data_names[i]
  print(data_name)
  
  # data_path = paste("~/projects/turs_supllementary_scripts/dataset/adbench/", data_name, sep="")
  data_path = paste("/Users/yanglincen/projects/turs_supllementary_scripts/dataset/adbench/",
                    data_name, sep="")
  data = read_npz(data_path)
  d = data.frame(data$X)
  colnames(d) = paste("V", seq(1, ncol(d)), sep="")
  
  d$"Y" = data$y
  d[,ncol(d)] = as.factor(d[,ncol(d)])
  num_cols[i] = ncol(d)
  num_rows[i] = nrow(d)
  num_classes[i] = length(table(d[,ncol(d)]))
  p = table(d[,ncol(d)]) / nrow(d)
  class_imbalance_entropy[i] = sum(-log(p) * p)
  max_prob[i] = max(p)
  min_prob[i] = min(p)
}
data_info2 = data.frame(data_name=data_names, num_rows=num_rows, num_cols=num_cols, 
                       num_classes=num_classes, 
                       max_class_prob=round(max_prob, 4),
                       min_class_prob=round(min_prob, 4))
data_info = rbind(data_info, data_info2)
data_info
for(i in 1:nrow(data_info)){
  if(data_info$data_name[i] == "14_glass.npz"){
    data_info$data_name[i] = "14_glassanomaly.npz"
  }
  
  if(data_info$data_name[i] == "41_Waveform.npz"){
    data_info$data_name[i] = "41_waveform2.npz"
  }
  split_name = strsplit(data_info$data_name[i], split="_")[[1]]
  if(length(split_name) > 1){
    data_info$data_name[i] = strsplit(split_name[2], split=".npz")[[1]][1]
  }
}

data_info$data_name = tolower(data_info$data_name)
data_info_dt = data.table(data_info)
data_info_dt = data_info_dt[order(data_info$num_classes!=2, data_name)]
data_info = data.frame(data_info_dt)

require(xtable)
# print(xtable(data_info, align = c("l", "c", "r"), digits = 4, 
#              caption = "Datasets for binary and multi-class classification."))
data_info$num_rows = as.integer(data_info$num_rows)
data_info$num_cols = as.integer(data_info$num_cols)
data_info$num_classes = as.integer(data_info$num_classes)


colnames(data_info) = c("data", "# rows", "# columns", "# classes", "max class prob.", "min class prob.")
print(xtable(data_info, align = rep("c", ncol(data_info) + 1), digits=3,
             caption = "Datasets for binary and multi-class classification."),
      include.rownames = F)



