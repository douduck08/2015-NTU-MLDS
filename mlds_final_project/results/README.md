
evaluateLSTMandMLP.py

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-predict_type', type=str, default='test')
    parser.add_argument('-language_feature_dim', type=int, default=300)
    parser.add_argument('-model', type=str, required=True)
    parser.add_argument('-weights', type=str, required=True)
    parser.add_argument('-question_feature', type=str, required=True)
    parser.add_argument('-choice_feature', type=str, required=True)
    return parser.parse_args()