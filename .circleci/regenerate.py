#!/usr/bin/env python

import jinja2
import os.path

d = os.path.dirname(__file__)
env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(d),
    lstrip_blocks=True,
    autoescape=False,
)
with open(os.path.join(d, 'config.yml'), 'w') as f:
    f.write(env.get_template('config.yml.in').render())
