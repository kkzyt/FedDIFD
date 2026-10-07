export CUDA_VISIBLE_DEVICES="0"
export MODLE="/opt/data/private/code/FedDIFD/InstantSwap/Flower_Concept"
export SOURCE_MASK="./example-my/bbox.jpg"
export SOURCE_IMAGE="./example-my/ILSVRC2012_val_00009379.JPEG"
export OUTPUT_DIR="./example-my/tench"
python InstantSwap.py \
    --model_id $MODLE \
    --source_mask $SOURCE_MASK \
    --source_image $SOURCE_IMAGE \
    --source_prompt "An image of a tench." \
    --target_prompt "An image of a" \
    --diff_prompt "tench" \
    --diff_prompt_source "shell" \
    --guidance_scale 7.5 \
    --output $OUTPUT_DIR \
    --interval 5 \
    --iters 550 
