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
     'torchvision_hub.models.hub_example',
     'https://download.pytorch.org/models/resnet18-5c106cde.pth'),
    ('wrapper2',
     'torchvision_hub.transforms.hub_example',
     'https://download.pytorch.org/models/resnet18-5c106cde.pth')
    ]
dependencies = ['torch', 'math']
help_msg = ("/****** Hub Help Section ******/\n"
            "hello world\n"
            "/******  End of Message  ******/")
