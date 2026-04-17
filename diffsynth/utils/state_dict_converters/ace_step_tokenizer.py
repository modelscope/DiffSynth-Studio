"""
State dict converter for ACE-Step Tokenizer model.

The original checkpoint stores tokenizer and detokenizer weights at the top level:
- tokenizer.* (AceStepAudioTokenizer: audio_acoustic_proj, attention_pooler, quantizer)
- detokenizer.* (AudioTokenDetokenizer: embed_tokens, layers, proj_out)

These map directly to the AceStepTokenizer class which wraps both as
self.tokenizer and self.detokenizer submodules.
"""


def ace_step_tokenizer_converter(state_dict):
    """
    Convert ACE-Step Tokenizer checkpoint keys to DiffSynth format.

    The checkpoint keys `tokenizer.*` and `detokenizer.*` already match
    the DiffSynth AceStepTokenizer module structure (self.tokenizer, self.detokenizer).
    No key remapping needed — just extract the relevant keys.
    """
    new_state_dict = {}

    for key in state_dict:
        if key.startswith("tokenizer.") or key.startswith("detokenizer."):
            new_state_dict[key] = state_dict[key]

    return new_state_dict
