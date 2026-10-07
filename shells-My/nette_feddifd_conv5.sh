nc=1 # number of clients
dda=0.1 # dirichlet dataset alpha
md=Conv5 # Conv5 or ResNet18
cuda=3 # cuda id
cmr=train_results/Baseline_Ensemble_dir${dda}_nc${nc}_ENSEMBLE_${md}_Imagenette_s42_my # re-use the same client model root

python feddifd_main.py \
    -c configs/imagenette/feddifd.yaml \
    -dda $dda \
    -md $md \
    -is 42 \
    -sn FedDIFD_dir${dda}_nc${nc} \
    -g $cuda \
    -cmr $cmr \
    -nc $nc \
    -cis coreset+_dif_dist_syn \
    --feddifd_ipc 5 \
    --feddifd_inputs_init vae+fourier
