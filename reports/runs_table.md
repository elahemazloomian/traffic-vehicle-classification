| run | changes | params | epochs | best_epoch | val_f1_best | val_f1_last5 | val_acc_last5 | val_r2_best | train_minus_val_acc | minutes |
|---|---|---|---|---|---|---|---|---|---|---|
| a_dropout03 | train.epochs=60 model.channels=[32,64,128,256,256] model.dropout=0.3 | 982024 | 53 | 47 | 0.8944 | 0.8888 | 0.8875 | 0.8145 | 0.0774 | 13.4 |
| a_no_hflip | train.epochs=60 model.channels=[32,64,128,256,256] augmentation.hflip=0.0 | 982024 | 40 | 34 | 0.894 | 0.8826 | 0.8812 | 0.7947 | 0.0938 | 10.1 |
| a_wd1e3 | train.epochs=60 model.channels=[32,64,128,256,256] train.weight_decay=0.001 | 982024 | 43 | 37 | 0.896 | 0.8809 | 0.8797 | 0.7982 | 0.0901 | 10.9 |
| a_wd0 | train.epochs=60 model.channels=[32,64,128,256,256] train.weight_decay=0.0 | 982024 | 43 | 37 | 0.8939 | 0.8787 | 0.8771 | 0.8051 | 0.0948 | 10.8 |
| a_base_seed43 | train.epochs=60 model.channels=[32,64,128,256,256] seed=43 | 982024 | 42 | 36 | 0.8799 | 0.8755 | 0.8734 | 0.7829 | 0.0799 | 10.7 |
| a_base_seed44 | train.epochs=60 model.channels=[32,64,128,256,256] seed=44 | 982024 | 49 | 43 | 0.8808 | 0.8713 | 0.8693 | 0.7802 | 0.1024 | 12.4 |
| a_avgpool | train.epochs=60 model.channels=[32,64,128,256,256] model.pooling=avg | 982024 | 49 | 43 | 0.8803 | 0.8661 | 0.8641 | 0.7726 | 0.0864 | 12.3 |
| a_base | train.epochs=60 model.channels=[32,64,128,256,256] | 982024 | 33 | 27 | 0.8792 | 0.8615 | 0.8604 | 0.7841 | 0.0912 | 8.6 |
| d5_e60 | train.epochs=60 model.channels=[32,64,128,256,256] | 982024 | 33 | 27 | 0.8792 | 0.8615 | 0.8604 | 0.7841 | 0.0912 | 8.2 |
| a_dropout05 | train.epochs=60 model.channels=[32,64,128,256,256] model.dropout=0.5 | 982024 | 42 | 36 | 0.8607 | 0.856 | 0.8542 | 0.7556 | 0.0503 | 10.6 |
| a_no_aug | train.epochs=60 model.channels=[32,64,128,256,256] augmentation.enabled=false | 982024 | 24 | 18 | 0.8713 | 0.8541 | 0.8516 | 0.7592 | 0.1471 | 5.1 |
| d4_e60 | train.epochs=60 model.channels=[32,64,128,256] | 391432 | 47 | 41 | 0.8329 | 0.8255 | 0.8234 | 0.6905 | 0.045 | 11.5 |
| a_no_scheduler | train.epochs=60 model.channels=[32,64,128,256,256] train.scheduler=none | 982024 | 23 | 17 | 0.7913 | 0.7039 | 0.712 | 0.6389 | 0.1772 | 5.8 |
| e60 | train.epochs=60 | 94728 | 47 | 41 | 0.6786 | 0.6638 | 0.6703 | 0.494 | 0.0367 | 10.8 |
| baseline | - | 94728 | 30 | 30 | 0.6579 | 0.6216 | 0.6318 | 0.4361 | 0.0083 | 7.1 |
