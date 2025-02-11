python sample_fab.py \
    --checkpoint_dir outputs/030-MMDiT-B-2/ \
    --checkpoint_pattern "*.pt" \
    --prompts "a young boy skateboarder jumping on a platform on a skateboard" "A white dog is running down a rocky beach" "A boy smiles for the camera at a beach" "a Man" "a girl" "" \
    --output_dir outputs/sample \
    --image_size 256 \
    --batch_size 4