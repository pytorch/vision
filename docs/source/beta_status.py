from docutils import nodes
from docutils.parsers.rst import Directive


class BetaStatus(Directive):
    has_content = True
    text = "The {api_name} is in Beta stage, and backward compatibility is not guaranteed."
    node = nodes.warning

    def run(self):
        text = self.text.format(api_name=" ".join(self.content))
        return [self.node("", nodes.paragraph("", "", nodes.Text(text)))]


def setup(app):
    app.add_directive("betastatus", BetaStatus)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
