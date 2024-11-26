lambda_con=(0.01 0.05 0.1)
lambda_fea=(0.01 0.05 0.1)


for A in "${lambda_con[@]}"
do
    for B in "${lambda_fea[@]}"
    do
            # 执行 Python 命令
        python main.py --lambda_con $A --lambda_fea $B
    done   
done     

