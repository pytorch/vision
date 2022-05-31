from docutils import nodes
from docutils.parsers.rst import Directive


class BetaStatus(Directive):
    has_content = True

    def run(self):
        api_name = " ".join(self.content)
        text = f"The {api_name} is in Beta stage, and backward compatibility is not guaranteed."
        return [nodes.warning("", nodes.paragraph("", "", nodes.Text(text)))]


def setup(app):
    app.add_directive("betastatus", BetaStatus)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
