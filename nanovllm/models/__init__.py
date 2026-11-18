def get_model_class(hf_config):
    """Pick the nanovllm model class from the HF `architectures` field."""
    arch = (getattr(hf_config, "architectures", None) or ["Qwen3ForCausalLM"])[0]
    if arch == "Qwen3ForCausalLM":
        from nanovllm.models.qwen3 import Qwen3ForCausalLM
        return Qwen3ForCausalLM
    if arch == "LlamaForCausalLM":
        from nanovllm.models.llama import LlamaForCausalLM
        return LlamaForCausalLM
    if arch == "OlmoeForCausalLM":
        from nanovllm.models.olmoe import OlmoeForCausalLM
        return OlmoeForCausalLM
    raise ValueError(f"unsupported architecture {arch}")
