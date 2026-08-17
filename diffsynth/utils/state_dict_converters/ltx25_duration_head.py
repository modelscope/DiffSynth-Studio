def LTX25DurationHeadStateDictConverter(state_dict):
    return {
        name.removeprefix("duration_head."): state_dict[name]
        for name in state_dict
        if name.startswith("duration_head.")
    }
