from docutils import nodes
from docutils.parsers.rst import Directive


class BetaStatus(Directive):
    has_content = True
    text = "The {api_name} is in Beta stage, and backward compatibility is not guaranteed."

    def run(self):
        text = self.text.format(api_name=" ".join(self.content))
        return [nodes.warning("", nodes.paragraph("", "", nodes.Text(text)))]


class V2BetaStatus(BetaStatus):
    text = (
        "The {api_name} is in Beta stage, and while we do not expect major breaking changes, "
        "some APIs may still change according to user feedback. Please submit any feedback you may have "
        "in this issue: https://github.com/pytorch/vision/issues/6753, and you can also check "
        "out https://github.com/pytorch/vision/issues/7319 to learn "
        "more about the APIs that we suspect might involve future changes."
    )


def setup(app):
    app.add_directive("betastatus", BetaStatus)
    app.add_directive("v2betastatus", V2BetaStatus)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
