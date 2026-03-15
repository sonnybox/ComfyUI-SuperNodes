class SuperStopExecution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "message": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "Message.",
                    },
                ),
                "trigger": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If True, execution halts. If False, nothing happens.",
                        "forceInput": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "halt_execution"
    OUTPUT_NODE = True
    CATEGORY = "SuperNodes/Tools"

    def halt_execution(self, message, trigger):
        if trigger:
            alert = str(message)
            raise Exception(f"{alert}")

        return ()


NODE_CLASS_MAPPINGS = {"User Error": SuperStopExecution}

NODE_DISPLAY_NAME_MAPPINGS = {"User Error": "🐧 Show Error Message"}
