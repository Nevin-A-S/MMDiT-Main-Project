python SampleMMditControlnet.py \
    --model MMDiT-B/2 \
    --pretrained_path outputs/009-MMDiT-B-2/checkpoint_1100000.pt \
    --image_size 256 \
    --batch_size 64 \
    --output_dir outputs/sample_control_net \
    --ckpt outputs/028-MMDiT-B-2/checkpoint_0070000.pt \
    --cfg 0 