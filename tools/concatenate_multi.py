from comfy_api.latest import io


class SuperConcatenateMulti(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        names = [f"text_{chr(i)}" for i in range(ord("a"), ord("z") + 1)]
        autogrow_template = io.Autogrow.TemplateNames(
            input=io.String.Input("text", force_input=True),
            names=names,
            min=1,
        )

        return io.Schema(
            node_id="SuperConcatenateMulti",
            display_name="🐧 Concatenate Multi",
            category="SuperNodes/Tools",
            inputs=[
                io.String.Input(
                    "delimiter", force_input=True
                ),  # required to be forced input due to scramble bug
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

        text_values = list(texts.values())

        # Filter out empty strings
        filtered = [str(t) for t in text_values if str(t)]

        return io.NodeOutput(delimiter.join(filtered))


NODE = [SuperConcatenateMulti]
