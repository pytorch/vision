# /**************************For Repo Owner************************************
# entrypoints is a list of (func_name, module_name, checkpoint_dir).
#   func_name: required, this function should return a model
#   module_name: required, specifies where to load the module in the package
#   checkpoint_url: optional, specifies where to download the checkpoint
# dependencies is a list of dependent package names.
# help_msp is a string, explaining what model each entrypoint points to.
# Example repo structure:
# repo/
#   hubconf.py
#   a/
#     __init__.py
#     b/
#       __init__.py
#       c.py
# For callable `x` defined in c.py, make sure you can do `import a.b.c.x` when
# starting an interactive python shell in repo/.
# The corresponding entrypoint is ('x', 'a.b.c', <checkpoint_url>).
# ****************************************************************************/

entrypoints = [
    ('wrapper1',
     'hub_examples.example',
     'https://download.pytorch.org/models/resnet18-5c106cde.pth'),
    ('resnet18',
     'torchvision.models.restnet18',
     'https://download.pytorch.org/models/resnet18-5c106cde.pth')
    ]
dependencies = ['torch', 'math']
help_msg = ("/****** Hub Help Section ******/\n"
            "hello world\n"
            "/******  End of Message  ******/")
