.. role:: hidden
    :class: hidden-section
.. currentmodule:: {{ module }}


{{ name | underline}}

.. autoclass:: {{ name }}
    :members:
        __getitem__,
        {% if "category_name" in methods %} category_name {% endif %}
    :special-members:
