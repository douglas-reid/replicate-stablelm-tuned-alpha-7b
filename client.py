from steamship import Steamship

from plugin_handle import PLUGIN_HANDLE


def main(prompt: str):
    with Steamship.temporary_workspace() as client:
        print(f'Running in workspace: {client.config.workspace_handle}')
        llm = client.use_plugin(PLUGIN_HANDLE)
        task = llm.generate(text=prompt, options={"max_tokens": 500, "temperature": 0.8})
        task.wait()

        output_blocks = task.output.blocks
        for block in output_blocks:
            print(block.text)


if __name__ == "__main__":
    main(input("Prompt: "))
