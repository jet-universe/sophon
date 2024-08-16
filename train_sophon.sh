#!/bin/bash

echo "args: $@"

# set the dataset dir via `DATADIR`
[[ -z $DATADIR ]] && DATADIR='./datasets/JetClassII'

# set a comment via `COMMENT`
suffix=${COMMENT}

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}

[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

# run mode
MODE=$1
[[ -z $MODE ]] && echo "Usage: $0 [make_weight|train|convert]" && exit 1

# default configurations
epochs=80
samples_per_epoch=$((10000 * 1024 / $NGPUS))
samples_per_epoch_val=$((2500 * 1024))

dataconfig="data/JetClassII/JetClassII_full.yaml"
modelopts="networks/example_ParticleTransformer_sophon.py --use-amp -o num_classes 188 -o fc_params [(512,0.1)]"  # enlarge FC layers
dataopts="--num-workers 5 --fetch-step 1.0 --data-split-num 200"
batchopts="--batch-size 512 --start-lr 5e-4"  # remember to scale LR if using the DDP mode

# set the train/eval dataset
# for training: {Res2P: 0000-0199, Res34P: 0000-0859, QCD: 0000-0279}; for eval: {Res2P: 0200-0249, Res34P: 0860-1074, QCD: 0280-0349}
trainset_res2p=$(for i in $(seq -w 0000 0199); do echo -n "Res2P:${DATADIR}/Pythia/Res2P_$i.parquet "; done)
trainset_res34p=$(for i in $(seq -w 0000 0859); do echo -n "Res34P:${DATADIR}/Pythia/Res34P_$i.parquet "; done)
trainset_qcd=$(for i in $(seq -w 0000 0279); do echo -n "QCD:${DATADIR}/Pythia/QCD_$i.parquet "; done)

valset_res2p=$(for i in $(seq -w 0200 0249); do echo -n "${DATADIR}/Pythia/Res2P_$i.parquet "; done)
valset_res34p=$(for i in $(seq -w 0860 1074); do echo -n "${DATADIR}/Pythia/Res34P_$i.parquet "; done)
valset_qcd=$(for i in $(seq -w 0280 0349); do echo -n "${DATADIR}/Pythia/QCD_$i.parquet "; done)

allset_res2p=$(for i in $(seq -w 0000 0249); do echo -n "${DATADIR}/Pythia/Res2P_$i.parquet "; done)
allset_res34p=$(for i in $(seq -w 0000 1074); do echo -n "${DATADIR}/Pythia/Res34P_$i.parquet "; done)
allset_qcd=$(for i in $(seq -w 0000 0349); do echo -n "${DATADIR}/Pythia/QCD_$i.parquet "; done)

# run command
if [[ $MODE == "make_weight" ]]; then
    # before training, using all samples to precalculate data sampling weights based on the `weights` section in the data config
    $CMD --print \
        --data-train $allset_res2p $allset_res34p $allset_qcd \
        --data-config $dataconfig --network-config $modelopts \
        --model-prefix training/JetClassII_Sophon${suffix}/net \
        $dataopts $batchopts \
        --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} --num-epochs $epochs --optimizer ranger --gpus 0 \
        --log-file logs/JetClassII_Sophon${suffix}/make_weight.log \
        "${@:2}"

elif [[ $MODE == "train" ]]; then
    $CMD --no-remake-weights \
        --data-train $trainset_res2p $trainset_res34p $trainset_qcd \
        --data-val $valset_res2p $valset_res34p $valset_qcd \
        --data-config $dataconfig --network-config $modelopts \
        --model-prefix training/JetClassII_Sophon${suffix}/net \
        $dataopts $batchopts \
        --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} --num-epochs $epochs --optimizer ranger --gpus 0 \
        --log-file logs/JetClassII_Sophon${suffix}/train.log --tensorboard JetClassII_Sophon${suffix} \
        "${@:2}"

elif [[ $MODE == "convert" ]]; then
    # convert the trained model (best epoch) to ONNX format. Should set "-o export_embed True" to the model
    $CMD --no-remake-weights \
        --data-config $dataconfig --network-config $modelopts -o export_embed True \
        --model-prefix training/JetClassII_Sophon${suffix}/net_best_epoch_state.pt \
        --export-onnx model.onnx \
        "${@:2}"

else
    echo "Usage: $0 [make_weight|train|convert]"
    exit 1
fi
