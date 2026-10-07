#export CUDA_VISIBLE_DEVICES="5"
export MODLE="stabilityai/stable-diffusion-2-1-base"
export SOURCE_IMAGE="./example-my/ILSVRC2012_val_00009379.JPEG"
export OUTPUT_DIR="./example-my"
python get_bbox.py \
    --model_id $MODLE \
    --source_image $SOURCE_IMAGE \
    --source_prompt "A fish is held in the hand of someone" \
    --guidance_scale 3 \
    --word_idx 5 \
    --output $OUTPUT_DIR \
    --iters 3
