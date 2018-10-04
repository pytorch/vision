# /**************************For Repo Owner************************************
# entrypoints is a tuple of (func_name, module_name, checkpoint_dir).
#   func_name: required, this function should return a model
#   module_name: required, specifies where to load the module in the package
#   checkpoint_url: optional, specifies where to download the checkpoint
# dependencies is a list of dependent package names.
# help_msp is a string, explaining what model each entrypoint points to.
# ****************************************************************************/

entrypoints = [
    ('wrapper1',
     'hub_example.example',
     'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'),
    ('alexnet',
     'torchvision.models.alexnet',
     'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    ]
dependencies = ['torch', 'math']
help_msg = ("/****** Hub Help Section ******/\n"
            "hello world\n"
            "/******  End of Message  ******/")
