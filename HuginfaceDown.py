from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="unsloth/gpt-oss-20b-GGUF",
    filename="gpt-oss-20b-Q6_K.gguf",
    local_dir="/home/roshan/models/gpt-oss",
    local_dir_use_symlinks=False
)
