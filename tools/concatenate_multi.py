from comfy_api.latest import io


class SuperConcatenateMulti(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        autogrow_template = io.Autogrow.TemplatePrefix(
            input=io.String.Input("text", force_input=True),
            prefix="text_",
            min=2,
        )

        return io.Schema(
            node_id="SuperConcatenateMulti",
            display_name="🐧 Concatenate Multi",
            category="SuperNodes/Tools",
            inputs=[
                io.Autogrow.Input("texts", template=autogrow_template),
                io.String.Input("delimiter", default=", "),
            ],
            outputs=[
                io.String.Output(display_name="text"),
            ],
        )

    @classmethod
    def execute(
        cls, texts: io.Autogrow.Type, delimiter: str, **kwargs
    ) -> io.NodeOutput:
        # texts is a dict mapping input names ('text_0', 'text_1') to their values
        text_values = list(texts.values())

        # Filter out empty strings
        filtered = [str(t) for t in text_values if str(t)]

        return io.NodeOutput(delimiter.join(filtered))


V3_NODES = [SuperConcatenateMulti]
