taskset -c 0 python run.py --dataset deep-image-96-angular  --definitions algos-definations/qps/algos-deep.yaml --local --runs 1 --count 10
taskset -c 0 python run.py --dataset sift-128-euclidean     --definitions algos-definations/qps/algos-sift.yaml --local --runs 1 --count 10
taskset -c 0 python run.py --dataset glove-100-angular      --definitions algos-definations/qps/algos-glove-100.yaml --local --runs 1 --count 10
taskset -c 0 python run.py --dataset glove-25-angular       --definitions algos-definations/qps/algos-glove-25.yaml --local --runs 1 --count 10

python run.py --dataset deep-image-96-angular   --definitions algos-definations/pr/algos-deep.yaml --local --runs 1 --count 100
python run.py --dataset sift-128-euclidean      --definitions algos-definations/pr/algos-sift.yaml --local --runs 1 --count 100
python run.py --dataset glove-100-angular       --definitions algos-definations/pr/algos-glove-100.yaml --local --runs 1 --count 100
python run.py --dataset glove-25-angular        --definitions algos-definations/pr/algos-glove-25.yaml --local --runs 1 --count 100

python run.py --dataset deep-image-96-angular   --definitions algos-definations/del/algos-deep.yaml --local --runs 1 --count 1 --delete


# python plot.py --dataset deep-image-96-angular
# python plot.py --dataset sift-128-euclidean
# python plot.py --dataset glove-100-angular
# python plot.py --dataset glove-25-angular
