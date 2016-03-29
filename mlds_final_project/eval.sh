model_dir=./tmp/
model=${model_dir}glove_sum_v2_idim_4096_ldim_1800_dropout_0.4_unit_1024_1024_1024.json
qfile=./data/question_wordvector/glove_sum_v2_300_train.pkl.gz
cfile=./data/choice_wordvector/glove_sum_v2_1500_train.pkl.gz
ifile=./data/image_feature/caffenet_4096_train.pkl.gz
idim=4096
ldim=1800
w1=${model_dir}glove_sum_v2_idim_4096_ldim_1800_dropout_0.4_unit_1024_1024_1024_valid_01_epoch_100_loss_3.023_error_0.237.hdf5
w2=${model_dir}glove_sum_v2_idim_4096_ldim_1800_dropout_0.4_unit_1024_1024_1024_valid_02_epoch_100_loss_3.014_error_0.233.hdf5
w3=${model_dir}glove_sum_v2_idim_4096_ldim_1800_dropout_0.4_unit_1024_1024_1024_valid_03_epoch_100_loss_3.016_error_0.237.hdf5
w4=${model_dir}glove_sum_v2_idim_4096_ldim_1800_dropout_0.4_unit_1024_1024_1024_valid_04_epoch_100_loss_3.028_error_0.240.hdf5
w5=${model_dir}glove_sum_v2_idim_4096_ldim_1800_dropout_0.4_unit_1024_1024_1024_valid_05_epoch_100_loss_3.016_error_0.239.hdf5
w6=${model_dir}glove_sum_v2_idim_4096_ldim_1800_dropout_0.4_unit_1024_1024_1024_valid_06_epoch_100_loss_3.005_error_0.246.hdf5
w7=${model_dir}glove_sum_v2_idim_4096_ldim_1800_dropout_0.4_unit_1024_1024_1024_valid_07_epoch_100_loss_3.025_error_0.235.hdf5
w8=${model_dir}glove_sum_v2_idim_4096_ldim_1800_dropout_0.4_unit_1024_1024_1024_valid_08_epoch_100_loss_3.018_error_0.240.hdf5
w9=${model_dir}glove_sum_v2_idim_4096_ldim_1800_dropout_0.4_unit_1024_1024_1024_valid_09_epoch_100_loss_3.019_error_0.233.hdf5
w10=${model_dir}glove_sum_v2_idim_4096_ldim_1800_dropout_0.4_unit_1024_1024_1024_valid_10_epoch_100_loss_3.030_error_0.235.hdf5

#python evaluateLSTMandMLP.py -model ${model} -idim ${idim} -ldim ${ldim} -w ${w1} -qf ${qfile} -cf ${cfile} -if ${ifile} -predict_type train
#for w in $w1 $w2 $w3 $w4 $w5 $w6 $w7 $w8 $w9 $w10
#do
#    python evaluateLSTMandMLP.py -model ${model} -idim ${idim} -ldim ${ldim} -w ${w} -qf ${qfile} -cf ${cfile} -if ${ifile} -predict_type test
#done
model=./data/models/glove_weightedsum_idim_4096_ldim_1800_dropout_0.4_unit_1024_1024_1024.json
qfile=./data/question_wordvector/glove_weightedsum_300_test.pkl.gz
cfile=./data/choice_wordvector/glove_sum_v2_1500_test.pkl.gz
ifile=./data/image_feature/caffenet_4096_test.pkl.gz
w=./data/models/glove_weightedsum_idim_4096_ldim_1800_dropout_0.4_unit_1024_1024_1024_valid_07_epoch_100_loss_3.142_error_0.259.hdf5
python evaluateLSTMandMLP.py -model ${model} -idim ${idim} -ldim ${ldim} -w ${w} -qf ${qfile} -cf ${cfile} -if ${ifile} -predict_type test
