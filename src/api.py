"""Generator plugin for Stable Diffusion running on replicate.com."""
import json
import logging
import time
from enum import Enum
from typing import Any, Dict, Type, Union

import replicate
from pydantic import Field
from steamship import Block, MimeTypes, Steamship, SteamshipError, Task, TaskState
from steamship.invocable import Config, InvocableResponse, InvocationContext
from steamship.plugin.generator import Generator
from steamship.plugin.inputs.raw_block_and_tag_plugin_input import RawBlockAndTagPluginInput
from steamship.plugin.outputs.raw_block_and_tag_plugin_output import RawBlockAndTagPluginOutput
from steamship.plugin.request import PluginRequest

REPLICATE_MODEL_NAME = "stability-ai/stablelm-tuned-alpha-7b"


class ModelVersionEnum(str, Enum):
    """Models supported by the plugin."""
    default = "c49dae362cbaecd2ceabb5bd34fdb68413c4ff775111fea065d259d577757beb"

    @classmethod
    def list(cls):
        """List all supported model versions sizes."""
        return list(map(lambda c: c.value, cls))


def task_status_response(state: TaskState, message, prediction_id: str) -> InvocableResponse:
    """Build a response object with a TaskState and message for a given transcription_id."""
    return InvocableResponse(
        status=Task(
            state=state,
            remote_status_message=message,
            remote_status_input={"prediction_id": prediction_id},
        )
    )


class StableLMTunedAlpha7BPlugin(Generator):
    """**Example** plugin for generating text from prompts via StableLM-Tuned-Alpha-7B running on replicate.

    StableLM-Tuned-Alpha-7B is a 7B parameter decoder-only language model built on top of the StableLM-Base-Alpha models
    and further fine-tuned on various chat and instruction-following datasets.

    The plugin accepts the following **runtime** params:
    - prompt: the input prompt for the model
    - temperature: adjusts randomness of outputs, greater than 1 is random and 0 is deterministic.
    - max_tokens: maximum number of tokens to generate. A word is generally 2-3 tokens.
    """

    class StableLMTunedAlpha7BPluginConfig(Config):
        """Configuration for the Stable Diffusion Plugin."""

        replicate_api_key: str = Field(
            "",
            description="API key to use for replicate.",
        )
        model_version: ModelVersionEnum = Field(
            ModelVersionEnum.default,
            description="Model version to use for generation. Must be one of:"
            f"{ModelVersionEnum.list()}",
        )

    @classmethod
    def config_cls(cls) -> Type[Config]:
        """Return configuration template for the generator."""
        return cls.StableLMTunedAlpha7BPluginConfig

    config: StableLMTunedAlpha7BPluginConfig

    def __init__(
        self,
        client: Steamship = None,
        config: Dict[str, Any] = None,
        context: InvocationContext = None,
    ):
        super().__init__(client, config, context)
        self._replicate_client = replicate.Client(api_token=self.config.replicate_api_key)
        self._model = self._replicate_client.models.get(REPLICATE_MODEL_NAME)
        self._version = self._model.versions.get(self.config.model_version)

    def run(
        self, request: PluginRequest[RawBlockAndTagPluginInput]
    ) -> InvocableResponse[RawBlockAndTagPluginOutput]:
        """Run the image generator against all the text, combined."""
        if request.is_status_check:
            return self._check_status(request)
        else:
            return self._start_work(request)

    def _check_status(
        self, request: PluginRequest[RawBlockAndTagPluginInput]
    ) -> Union[InvocableResponse, InvocableResponse[RawBlockAndTagPluginOutput]]:
        if (
            request.status.remote_status_input is None
            or "prediction_id" not in request.status.remote_status_input
        ):
            raise SteamshipError(
                message="Status check requests must provide a valid 'prediction_id'."
            )

        prediction_id = request.status.remote_status_input.get("prediction_id")
        return self._check_prediction_status(prediction_id)

    def _start_work(
        self, request: PluginRequest[RawBlockAndTagPluginInput]
    ) -> Union[InvocableResponse, InvocableResponse[RawBlockAndTagPluginOutput]]:
        logging.debug("starting prediction...")

        options = request.data.options
        prompt = " ".join([block.text for block in request.data.blocks if block.text is not None])
        inputs = {
            "prompt": prompt,
        }

        if options:
            if options.get("max_tokens"):
                inputs["max_tokens"] = options.get("max_tokens")
            if options.get("temperature"):
                inputs["temperature"] = options.get("temperature")

        # ideally, we would validate inputs here (ensure temp is between 0 and 1, etc.)
        # logging.info(f"prediction inputs: {inputs}")

        try:
            prediction = self._replicate_client.predictions.create(
                version=self._version, input=inputs
            )
            prediction_id = prediction.id
            logging.info("started prediction", extra={'prediction_id': json.dumps(prediction_id)})
        except Exception as e:
            raise SteamshipError(f"could not schedule work: {json.dumps(e)}")

        # poll for up to one minute on initial request in attempt to reduce task churn and overall latency
        # NOTE: this may be overkill for text-generation. it may make more sense to use the sync API instead
        # of just polling.
        max_time = time.time() + 60
        running = True
        while (time.time() < max_time) and running:
            response = self._check_prediction_status(prediction_id)
            if response.status and response.status.state not in [
                TaskState.waiting,
                TaskState.running,
            ]:
                running = False
            time.sleep(0.1)

        # _Believe_ the endpoint is idempotent, so this _should_ be ok?
        return self._check_prediction_status(prediction_id)

    def _check_prediction_status(
        self, prediction_id: str
    ) -> Union[InvocableResponse, InvocableResponse[RawBlockAndTagPluginOutput]]:
        logging.info("checking prediction", extra={'prediction_id': json.dumps(prediction_id)})
        try:
            prediction = self._replicate_client.predictions.get(id=prediction_id)
        except Exception as e:
            logging.warning(
                f"could not get status of prediction: error={json.dumps(e)}",
                extra={'prediction_id': json.dumps(prediction_id)}
            )
            return task_status_response(TaskState.running, "Generation job ongoing.", prediction_id)

        if prediction.status not in ["succeeded", "failed", "canceled"]:
            logging.info("prediction in-progress", extra={'prediction_id': json.dumps(prediction_id)})
            return task_status_response(TaskState.running, "Prediction job ongoing.", prediction_id)

        if prediction.status in ["failed", "canceled"]:
            raise SteamshipError(f"prediction task ({prediction_id}) failed (or was cancelled).")

        logging.info("prediction complete", extra={'prediction_id': json.dumps(prediction_id)})
        output_text = "".join(prediction.output)
        # logging.info(f"prediction output is: {output_text}")

        blocks = [
            Block(text=output_text, mime_type=MimeTypes.TXT)
        ]
        return InvocableResponse(data=RawBlockAndTagPluginOutput(blocks=blocks))
