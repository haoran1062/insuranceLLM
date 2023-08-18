CUDA_VISIBLE_DEVICES=0,1 python api_server.py \
--model "/data/weights/trained/sft_merged" \
--port 8001 \
--tensor-parallel-size 2
