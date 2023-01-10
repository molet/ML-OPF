## Run from GoCompetition.jl directory.
### bash ./src/python_scripts/launch.sh
### Cases:
### pglib_opf_case24_ieee_rts
### pglib_opf_case30_ieee
### pglib_opf_case39_epri
### pglib_opf_case57_ieee
### pglib_opf_case73_ieee_rts
### pglib_opf_case118_ieee
### pglib_opf_case162_ieee_dtc
### pglib_opf_case300_ieee
### pglib_opf_case588_sdet
### pglib_opf_case1354_pegase
### pglib_opf_case2853_sdet

### CUDA Device
export CUDA_VISIBLE_DEVICES=0

### Inputs
export EXPERIMENT="./src/python_scripts/experiment.py"
export DIRECTORY="./src/python_scripts/"
export DATA="/home/ubuntu/processed_local/"
export RESULTS="/home/ubuntu/results_local/"
export PRIMAL="True" # True; False
export LOCAL="True" # True; False - only impacts primal GNNs.
export ARCHITECTURES="fcnn1 fcnn cnn gcn chnn snn gat gc" # fcnn; cnn; gcn; chnn; snn.
export KERNEL_SIZE=5
export PATIENCE=20
export MIN_EPOCHS=1
export MAX_EPOCHS=2000
export BATCH_SIZE=100
export TRAIN_SIZE=0.8
export TEST_SIZE=0.1
export KERNEL_WT=0 # Impedance gauss kernel weight.
export SEEDS="0" #"0 1 2 3 4 5 6 7 8 9"
export LEARNING_RATE=0.001
export GRID="pglib_opf_case118_ieee"

/home/ubuntu/anaconda3/envs/pytorch_latest_p37/bin/python3 ${EXPERIMENT} \
--DIRECTORY ${DIRECTORY} \
--DATA ${DATA} \
--RESULTS ${RESULTS} \
--GRID ${GRID} \
--PRIMAL ${PRIMAL} \
--LOCAL ${LOCAL} \
--ARCHITECTURES "$ARCHITECTURES" \
--KERNEL_SIZE ${KERNEL_SIZE} \
--PATIENCE ${PATIENCE} \
--MIN_EPOCHS ${MIN_EPOCHS} \
--MAX_EPOCHS ${MAX_EPOCHS} \
--BATCH_SIZE ${BATCH_SIZE} \
--TRAIN_SIZE ${TRAIN_SIZE} \
--TEST_SIZE ${TEST_SIZE} \
--KERNEL_WT ${KERNEL_WT} \
--SEEDS "$SEEDS" \
--LEARNING_RATE ${LEARNING_RATE} 
