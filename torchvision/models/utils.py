# Don't remove this file and don't change the imports of load_state_dict_from_url
# from other files. We need this so that we can swap load_state_dict_from_url with
# a custom internal version in fbcode.
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
