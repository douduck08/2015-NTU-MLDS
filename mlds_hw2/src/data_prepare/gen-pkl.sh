mkdir -p ../../pkl ../../model ../../log ../../result \
         ../../result/smoothed_test_result ../../result/smoothed_valid_result  \
         ../../result/test_result  ../../result/valid_result ../../result/final_result


# Genally, you should only change the variable "DNNResult".
# t means train and v means valid.
# The integers follow t and v are the FER.
DNNResult=t26v31

# "dim" is the dimension of the probability.
# "tmp" is for the data which is without preprocessing.
# It will be remove after the shell sript finish.

dim=48
probDataPath=/home/roylu/datashare/DNNResult/$DNNResult/
outputFilename=../../pkl/$DNNResult.pkl

echo '... coverting data to pkl'
python ./make_pkl.py $dim $probDataPath $outputFilename

echo 'done'
