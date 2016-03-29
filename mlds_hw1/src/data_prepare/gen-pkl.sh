mkdir -p ../../map ../../pkl ../../fbank_valid ../../model ../../log ../../result \
         ../../result/smoothed_test_result ../../result/smoothed_valid_result  \
         ../../result/test_result  ../../result/valid_result ../../result/final_result ../../prob \
         ../../prob/train ../../prob/test ../../prob/valid

dataPath=/home/roylu/datashare/MLDS_data/         

echo '... generate map'
python ./0_gen-map.py $dataPath

echo '... generate int label'
python ./1_gen-intlab.py $dataPath

echo '... pick  validation set'
python ./2_pick_valid.py $dataPath

echo '... make pkl file'
python ./3_make_pkl.py 69 $dataPath

echo 'done'
