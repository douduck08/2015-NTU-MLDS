if [ "$#" -ne 3 ]; then
   echo "Train MLP model"
   echo "Usage: $0 <image_feature> <question_feature> <choice_feature>"
   echo "eg. $0 ./data/image_feature/caffenet_4096_train.pkl.gz ./data/question_wordvector/glove_sum_v2_300_train.pkl.gz ./data/choice_wordvector/glove_sum_1500_train.pkl.gz"
   echo ""
   exit 1;
fi

dropout=0.5
cross_valid=1
language_only=True
image_only=False
language_dim=300
image_dim=0
image_input_dim=0
image_feature=$1
question_feature=$2
choice_feature=$3
cross_valid=8
epochs=100
activation=maxout

python trainLSTMandMLP.py -u 1024 1024 1024 \
                          -dropout ${dropout} \
                          -a ${activation} \
                          -lonly ${language_only} \
                          -ionly ${image_only} \
                          -lfdim ${language_dim} \
                          -ifdim ${image_dim} \
                          -iidim ${image_input_dim} \
                          -cross_valid ${cross_valid} \
                          -qf ${question_feature} \
                          -cf ${choice_feature} \
                          -if ${image_feature} \
                          -cross_valid ${cross_valid} \
                          -epochs ${epochs}
