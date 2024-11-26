dataset=('MUTAG' 'BZR' 'BZR_MD' 'DHFR' 'COX2' 'PROTEINS' 'NCI1' 'DD')

for A in "${dataset[@]}"
do
 
    CUDA_VISIBLE_DEVICES=1 python main.py --dataset $A --use_node_labels
        
done

