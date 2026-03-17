from comfy_api.latest import io


class SuperConcatenateMulti(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        autogrow_template = io.Autogrow.TemplatePrefix(
            input=io.String.Input("text", force_input=True),
            prefix="text_",
            min=1,
            max=io.Autogrow._MaxNames,
        )

        return io.Schema(
            node_id="SuperConcatenateMulti",
            display_name="🐧 Concatenate Multi",
            category="SuperNodes/Tools",
            inputs=[
                io.String.Input("delimiter", force_input=True),
                io.Autogrow.Input("texts", template=autogrow_template),
            ],
            outputs=[
                io.String.Output(display_name="text"),
            ],
        )

    @classmethod
    def execute(
        cls, delimiter: str, texts: io.Autogrow.Type, **kwargs
    ) -> io.NodeOutput:
        # Replace literal backslash-n with actual newlines
        delimiter = delimiter.replace("\\n", "\n")

        # texts is a dict mapping input names ('text_0', 'text_1') to their values
        text_values = list(texts.values())

        # Filter out empty strings
        filtered = [str(t) for t in text_values if str(t)]

        return io.NodeOutput(delimiter.join(filtered))


NODE = [SuperConcatenateMulti]
