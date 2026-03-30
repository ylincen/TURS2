rm(list=ls())
require(RWeka)
library(pROC)
require(PRROC)
library(caret)

# data_names = list.files("~/projects/TURS/ADbench_datasets_Classical/")

data_names = c('26_optdigits.npz', '42_WBC.npz', '21_Lymphography.npz', '34_smtp.npz', '28_pendigits.npz', '43_WDBC.npz', '36_speech.npz', '31_satimage-2.npz', '3_backdoor.npz', '38_thyroid.npz', '41_Waveform.npz', '23_mammography.npz', '40_vowels.npz', '25_musk.npz', '1_ALOI.npz', '14_glass.npz')
roc_auc_all_data = c()
f1_all_data = c()
runtime_all_data = c()
data_names_all_data = c()
pr_auc_all_data = c()

for(ii in 1:length(data_names)){
  cat("running on data ", data_names[ii], "\n")
  for(fold in 0:4){
    cat("fold: ", fold, "\n")
    train_name = paste(
      "~/projects/TURS/turs2/ad_bench_competitors/noise20percent_train_test_split_data/", 
      data_names[ii], "_train_fold_", fold, sep="")
    d_train = read.csv(train_name)
    colnames(d_train)[ncol(d_train)] = "Ytrain"
    d_train$Ytrain = as.factor(d_train$Ytrain)
    
    test_name = paste(
      "~/projects/TURS/turs2/ad_bench_competitors/noise20percent_train_test_split_data/", 
      data_names[ii], "_test_fold_", fold, sep="")
    d_test = read.csv(test_name)
    colnames(d_test)[ncol(d_test)] = "Ytest"
    
    start_time = Sys.time()
    model = PART(Ytrain~., data = d_train)
    y_hat = predict(model, d_test[,1:(ncol(d_test)-1)], type="class")
    runtime = as.numeric(Sys.time() - start_time)
    
    runtime_all_data = c(runtime_all_data, runtime)
    data_names_all_data = c(data_names_all_data, data_names[ii])
    
    roc_obj <- roc(d_test$Y, as.numeric(y_hat))
    roc_auc_all_data = c(roc_auc_all_data, auc(roc_obj)[1])
    
    pr_auc = pr.curve(scores.class0 = as.numeric(y_hat), weights.class0 = as.numeric(d_test$Ytest))[[2]]
    pr_auc_all_data = c(pr_auc_all_data, pr_auc)
    
    confu_mat = confusionMatrix(y_hat, as.factor(d_test$Ytest), 
                                mode = "everything", positive="1")
    f1 = confu_mat$byClass["F1"]
    f1_all_data = c(f1_all_data, f1)
  }
  res_df = data.frame(data_names_all_data, runtime_all_data, 
                      roc_auc_all_data, 
                      pr_auc_all_data, f1_all_data)
  write.csv(res_df, "~/projects/TURS/turs2/ad_bench_competitors/RIPPER/res_PART_noise20percent.csv")
}
