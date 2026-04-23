def AceStepTokenizerStateDictConverter(state_dict):
    new_state_dict = {}

    for key in state_dict:
        if key.startswith("tokenizer.") or key.startswith("detokenizer."):
            new_state_dict[key] = state_dict[key]

    return new_state_dict
