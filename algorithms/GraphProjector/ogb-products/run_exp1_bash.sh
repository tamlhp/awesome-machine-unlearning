


# SEED=0
# python exp3_graph_projector_v2.py --use_cross_entropy \
#                                       --recycle_steps 500 \
#                                       --epochs 8000 --hop_neighbors 25 \
#                                       --seed $SEED --continue_unlearn_step


    
# for SEED in 0 1 123 231 321; do
#     python exp3_graph_projector_v2.py --use_cross_entropy \
#                                       --recycle_steps 500 \
#                                       --epochs 8000 --hop_neighbors 25 \
#                                       --seed $SEED
                                      
#     python exp3_graph_projector_v2.py --use_cross_entropy \
#                                       --recycle_steps 500 \
#                                       --epochs 8000 --hop_neighbors 25 \
#                                       --seed $SEED \
#                                       --use_adapt_gcs_x --use_adapt_gcs_y

# done

