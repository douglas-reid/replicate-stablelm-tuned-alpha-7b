# Example StableLM Generator Plugin for Steamship

This is **exclusively** meant to be a demonstration of how one might add a custom LLM into the Steamship
ecosystem and use it for text-generation. It SHOULD NOT be used for anything beyond experimentation with
the platform and as inspiration for future integrations.

## Model info

Uses an integration with [Replicate](replicate.com) for model hosting.

Model page: https://replicate.com/stability-ai/stablelm-tuned-alpha-7b

### Licenses

- Base model checkpoints (StableLM-Base-Alpha) are licensed under the Creative Commons license (CC BY-SA-4.0). Under the license, you must give credit to Stability AI, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the Stability AI endorses you or your use.

- Fine-tuned checkpoints (StableLM-Tuned-Alpha) are licensed under the Non-Commercial Creative Commons license ([CC BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)), in-line with the original non-commercial license specified by Stanford Alpaca.

## Disclaimers

This is only an example to show how alternate LLMs could be integrated into [Steamship](steamship.com).
Absolutely no warranty is provided. Use at your risk.

This is NOT contributed or endorsed by Steamship in any way. It reflects entirely personal use and development, with
the goal of inspiring others in the larger community to pursue custom LLM integration.

**REMINDER**: Use of any plugin on Steamship is subject to Steamship's
[Terms of Service](https://www.steamship.com/policies/terms-of-service). 

## Deployment

From this directory, run `ship deploy`. This will provide a guided configuration of the Plugin for you, including
the local creation of a Steamship manifest that you can reuse for updating the Plugin, etc.

Make sure you select `plugin` and `generator` in the prompts provided.

Once the Plugin has been successfully deployed to Steamship, you can create **instances** of the plugin for
use in Steamship Packages and Agents.

## Use

### Simple

`client.py` provides a simple usage example (creating a plugin instance, and calling it directly). You will need to
supply your own `PLUGIN_HANDLE` to match your plugin.

### In an agent

To use in an agent, the code in `client.py` must be adapted slightly to create an `steamship.agents.schema.LLM`.

Rough sketch:

```python
from typing import List, Optional

from steamship import Block, Steamship, PluginInstance
from steamship.agents.schema import LLM

class StableLMTunedAlphaLLM(LLM):
    generator: PluginInstance
    client: Steamship
    _max_tokens: int
    _temperature: float
    

    def __init__(
        self, client, max_tokens: int = 500, temperature: float = 0.75, *args, **kwargs
    ):
        generator = client.use_plugin(PLUGIN_HANDLE)
        self._max_tokens = max_tokens
        self._temperature = temperature
        super().__init__(client=client, generator=generator, *args, **kwargs)

    def complete(self, prompt: str, stop: Optional[str] = None) -> List[Block]:
        options = {
            'max_tokens': self._max_tokens,
            'temperature': self._temperature,
        }
        action_task = self.generator.generate(text=prompt, options=options)
        action_task.wait()
        return action_task.output.blocks
```

and 

```python
from steamship.agents.react import ReACTAgent
from steamship.agents.service.agent_service import AgentService

class ExampleAgentService(AgentService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._agent = ReACTAgent(
            tools=[...],
            llm=StableLMTunedAlphaLLM(client=self.client),
        )
    
    ...
```