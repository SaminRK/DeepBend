# DeepBend


## How to get started
1. Clone our GitHub repository into your local machine and then enter into the cloned directory.
```bash
git clone https://github.com/SameeLab-BCM/DeepBend.git
cd DeepBend
```
2. Make sure you have virtualenv installed.
3. Create virtual environment.
```bash
virtualenv deepbend-env
source deepbend-env/bin/activate
```
4. Install all dependencies.
```bash
pip install -r requirements.txt
```
5. Train the DeepBend model.
```bash
 python src/train.py --model model35 --train-dataset data/dataset_9_train.txt \
  --validation-dataset data/dataset_8_test.txt --hyperparameters hyperparameters/hyperparameter.txt
 ```
6. Test DeepBend model
```bash
 python src/test.py --model model35 --test-dataset data/dataset_9_test.txt \
  --hyperparameters hyperparameters/hyperparameter.txt --model-weights model_weights/model_weights_file_name
 ```
7. K-fold cross validation
```bash
 python src/cross_validate.py --model model35 --dataset data/dataset_6.txt --k 10 \
  --hyperparameters hyperparameters/hyperparameter.txt
 ```
