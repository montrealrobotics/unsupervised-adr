mpirun -np 8 python -u train.py --sp-percent=0.0 --sp-polyak=0.95 --seed 128  --approach "udr" 2>&1 | tee reach.log
mpirun -np 8 python -u train.py --sp-percent=0.0 --sp-polyak=0.95 --seed 256  --approach "udr" 2>&1 | tee reach.log
mpirun -np 8 python -u train.py --sp-percent=0.0 --sp-polyak=0.95 --seed 512  --approach "udr" 2>&1 | tee reach.log
