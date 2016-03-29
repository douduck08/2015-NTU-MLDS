trainLSTMandMLP.py:
    parser.add_argument('-idim', '--image_feature_dim', type=int, default=4096)
    parser.add_argument('-ldim', '--language_feature_dim', type=int, default=300)
    parser.add_argument('-qf', '--question_feature', type=str, required=True)
    parser.add_argument('-cf', '--choice_feature', type=str, required=True)
    parser.add_argument('-lstm', type=bool, default=False)
    parser.add_argument('-lstm_units', type=int, default=512)
    parser.add_argument('-lstm_layers', type=int, default=1)
    parser.add_argument('-u', '--mlp_units', nargs='+', type=int, required=True)
    parser.add_argument('-a', '--mlp_activation', type=str, default='softplus')
    parser.add_argument('-odim', '--mlp_output_dim', type=int, default=300)
    parser.add_argument('-dropout', type=float, default=1.0)
    parser.add_argument('-maxout', type=bool, default=False)
    parser.add_argument('-memory_limit', type=float, default=6.0)
    parser.add_argument('-cross_valid', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-epochs', type=int, default=150)

ex1: cross-valid = 10, units = 512 512
python trainLSTMandMLP.py -ldim 1800 -cross_valid 10 -qf data/question_wordvector/glove_sum_300_train.csv -cf data/choice_wordvector/glove_sum_1500_train.csv -u 512 512
output:
    model/...
    model/glove_sum_idim_4096_ldim_1800_dropout_1.0_unit_512_512.json
    model/glove_sum_idim_4096_ldim_1800_dropout_1.0_unit_512_512_valid_01_epoch_100_loss_0.070_error_0.000.hdf5

ex2: cross-valid = 1, units = 512 512
python trainLSTMandMLP.py -ldim 1800 -qf data/question_wordvector/glove_sum_300_train.csv -cf data/choice_wordvector/glove_sum_1500_train.csv -u 512 512
output:
    model/...
    model/glove_sum_idim_4096_ldim_1800_dropout_1.0_unit_512_512.json
    model/glove_sum_idim_4096_ldim_1800_dropout_1.0_unit_3_epoch_100_loss_0.297.hdf5

===
evaluateLSTMandMLP.py:
    parser.add_argument('-predict_type', type=str, default='test')
    parser.add_argument('-language_feature_dim', type=int, default=300)
    parser.add_argument('-model', type=str, required=True)
    parser.add_argument('-weights', type=str, required=True)
    parser.add_argument('-question_feature', type=str, required=True)
    parser.add_argument('-choice_feature', type=str, required=True)

python evaluateLSTMandMLP.py -predict_type test -model model/2016010703_glove_sum_300.json -weights model/2016010703_glove_sum_300_epock_090_loss_0.259.hdf5 -question_feature data/question_wordvector/glove_sum_300_test.csv -choice_feature data/choice_wordvector/glove_sum_1500_test.csv