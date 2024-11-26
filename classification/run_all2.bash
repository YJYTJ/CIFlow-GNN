dataset=('IMDB-BINARY' 'IMDB-MULTI')

for A in "${dataset[@]}"
do
 
    CUDA_VISIBLE_DEVICES=1 python main.py --dataset $A
        
done

