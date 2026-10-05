| run | changes | params | epochs | best_epoch | val_f1_best | val_f1_last5 | val_acc_last5 | val_r2_best | train_minus_val_acc | minutes |
|---|---|---|---|---|---|---|---|---|---|---|
| f_no_hflip | model.channels=[32,64,128,256,256] train.epochs=40 train.scheduler=cosine train.early_stopping_patience=0 train.save=last augmentation.hflip=0.0 | 982024 | 40 | 37 | 0.8842 | 0.8778 | 0.8903 | 0.8153 | 0.0852 | 12.7 |
| f_wd1e3 | model.channels=[32,64,128,256,256] train.epochs=40 train.scheduler=cosine train.early_stopping_patience=0 train.save=last train.weight_decay=0.001 | 982024 | 40 | 31 | 0.8754 | 0.8678 | 0.8869 | 0.7851 | 0.0859 | 9.4 |
| f_wd0 | model.channels=[32,64,128,256,256] train.epochs=40 train.scheduler=cosine train.early_stopping_patience=0 train.save=last train.weight_decay=0.0 | 982024 | 40 | 26 | 0.8865 | 0.8669 | 0.8857 | 0.7728 | 0.0868 | 9.7 |
| f_no_aug | model.channels=[32,64,128,256,256] train.epochs=40 train.scheduler=cosine train.early_stopping_patience=0 train.save=last augmentation.enabled=false | 982024 | 40 | 36 | 0.8601 | 0.8543 | 0.8766 | 0.7923 | 0.1234 | 9.4 |
| a_dropout03 | train.epochs=60 model.channels=[32,64,128,256,256] model.dropout=0.3 | 982024 | 38 | 32 | 0.8661 | 0.8478 | 0.8737 | 0.7889 | 0.0512 | 9.7 |
| a_base_seed43 | train.epochs=60 model.channels=[32,64,128,256,256] seed=43 | 982024 | 39 | 33 | 0.8895 | 0.8472 | 0.8651 | 0.8101 | 0.0914 | 9.9 |
| f_avgpool | model.channels=[32,64,128,256,256] train.epochs=40 train.scheduler=cosine train.early_stopping_patience=0 train.save=last model.pooling=avg | 982024 | 40 | 37 | 0.843 | 0.8323 | 0.8554 | 0.7669 | 0.0759 | 9.8 |
| f_constant_lr | model.channels=[32,64,128,256,256] train.epochs=40 train.scheduler=cosine train.early_stopping_patience=0 train.save=last train.scheduler=none | 982024 | 40 | 40 | 0.8584 | 0.8115 | 0.8309 | 0.7874 | 0.114 | 9.6 |
| a_base | train.epochs=60 model.channels=[32,64,128,256,256] | 982024 | 23 | 17 | 0.8441 | 0.8016 | 0.836 | 0.7539 | 0.0682 | 6.1 |
| a_base_seed44 | train.epochs=60 model.channels=[32,64,128,256,256] seed=44 | 982024 | 23 | 17 | 0.814 | 0.7763 | 0.8011 | 0.7074 | 0.0944 | 5.8 |
