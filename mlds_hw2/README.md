The data should be placed in a directory name "data" which is of the same level with directory "src"

To run our program, type in the command line:

  cd ./src
  ./gen-pkl.sh
  python run.py setting.txt

You can get the final result in the directory name "result/final_result"

There are

Notice:
The test FER(%) is close to 100%, because we don't have the label of test data.  Therefore, we initialize the label as sil.