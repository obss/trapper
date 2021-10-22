# pylint: disable=protected-access
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    PreTrainedModel,
)
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.utils import logging

from trapper.common import Registrable

logger = logging.get_logger(__name__)


class ModelWrapper(Registrable):
    """
    The `ModelWrapper` is the registrable base factory that is responsible
    for creating transformer models from the `transformers` library using
    the `auto` classes . It does so by keeping a mapping from task name to the
    `AutoModelFor...`  classes defined in `transformers` such as
    `AutoModelForSequenceClassification`.
    To make this work, we register the wrapped `auto` classes with their task name
    to the registry as a subclass of `ModelWrapper`.

    This class does some extra things depending on the `model_type` of the created
    task-specific class. These include:
        - inspecting the `forward` parameters of the model,
        - checking the invalid parameters i.e. the parameters that are required
        by most of the models for that task but not the model with that
        `model_type`,
        - the extra parameters required by the models with that `model_type`
        but are not required by other models in general.

    Then, the legitimate parameters are stored as `forward_params` attribute of
    the model wrapper object. Thanks to that, `trapper.training.TransformerTrainer`
    can access that attribute to use as argument for the `model_forward_params`
    parameter while creating the dataset collator.

    Below are the explanations of the important class variables.
        _TASK_SPECIFIC_AUTO_CLASS (Type) : a registered auto class from the
        `Transformers` library

        _TASK_SPECIFIC_FORWARD_PARAMS (Tuple[str]) : The parameters that are
        required by the `forward` method of the created class. While registering
        task-specific subclasses, we manually inspect the models provided by the
        `Transformers` and try to find a common interface for the `forward`
        method, by trying to keep only parameters that are strictly required for
        the correct behavior.

        _MODEL_TYPE_TO_INVALID_PARAMS (Tuple[str]) : Here, we list the invalid
        parameters for the supported models. Invalid parameter means the model
        does not have it in its forward method signature. Unlisted models are
        either do not have invalid parameters since their APIs are very similar
        to the common transformer architectures or we have not listed them yet.

        _MODEL_TYPE_TO_EXTRA_PARAMS (Dict[str, Tuple[str]]) : Some models require
        extra parameters that are peculiar to them. E.g. `longformer` utilizes
        "global_attention_mask".

    Args:
        pretrained_model (): The pretrained model to be wrapped
    """

    default_implementation = "from_pretrained"
    _TASK_SPECIFIC_AUTO_CLASS: _BaseAutoModelClass = None
    _TASK_SPECIFIC_FORWARD_PARAMS: Tuple[str, ...] = None

    _MODEL_TYPE_TO_INVALID_PARAMS: Dict[str, Tuple[str]] = {
        "roberta": ("token_type_ids",),
        "longformer": ("token_type_ids",),
        "distilbert": ("token_type_ids",),
        "bart": ("token_type_ids",),
    }

    _MODEL_TYPE_TO_EXTRA_PARAMS: Dict[str, Tuple[str]] = {
        "longformer": ("global_attention_mask",),
    }

    def __init__(self, pretrained_model: Optional[PreTrainedModel] = None):
        #  We need to make `pretrained_model` optional with default of None,
        #  since otherwise allennlp tries to invoke __init__ although we
        #  register a classmethod as a default constructor and demand it via the
        #  "type" parameter inside the from_params method or a config file.
        if pretrained_model is None:
            raise ValueError("`pretrained_model` can not be None!")
        self._pretrained_model = pretrained_model
        self._forward_params = self._get_forward_params(pretrained_model)

    @property
    def forward_params(self) -> Tuple[str, ...]:
        return self._forward_params

    @property
    def model(self) -> PreTrainedModel:
        return self._pretrained_model

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, Path], *model_args, **kwargs
    ) -> ModelWrapper:
        """
        Creates and returns a transformer model wrapper from a task-specific wrapper
        factory. Additionally, handles the architectural changes (e.g. when further
        training an already fine-tuned model on a new dataset with different labels)
        for the task-specific models that require such changes.

        Args:
            pretrained_model_name_or_path ():
            *model_args ():
            **kwargs ():

        Returns:

        """
        if cls is ModelWrapper:
            raise EnvironmentError(
                "ModelWrapper is designed to be a factory that can "
                "instantiate concrete models using `ModelWrapper.from_params` "
                "method. If you want to instantiate a ModelWrapper object using "
                "`from_pretrained` method, you need to use a subclass of "
                "ModelWrapper instead."
            )
        if cls._TASK_SPECIFIC_AUTO_CLASS is AutoModelForTokenClassification:
            pretrained_model = cls._create_token_classification_model(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        else:
            pretrained_model = cls._TASK_SPECIFIC_AUTO_CLASS.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )

        return cls(pretrained_model=pretrained_model)

    @classmethod
    def _create_token_classification_model(
        cls, pretrained_model_name_or_path: Union[str, Path], *model_args, **kwargs
    ) -> PreTrainedModel:
        provided_num_labels = kwargs.get("num_labels")
        if provided_num_labels is not None:
            pretrained_model_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path
            )
            new_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
            pretrained_num_labels = getattr(
                pretrained_model_config, "num_labels", None
            )
            if pretrained_num_labels is None:
                raise ValueError(
                    f"Unexpected argument `num_labels` with "
                    f"the value of {provided_num_labels}"
                )
            if pretrained_num_labels != provided_num_labels:
                logger.warning(
                    "Provided `num_labels` value (%(provided)d) is different from "
                    "the one found in the archived config (%(pretrained)d). "
                    "The classifier head will have %(provided)d labels and be"
                    " initialized randomly!",
                    {
                        "provided": provided_num_labels,
                        "pretrained": pretrained_num_labels,
                    },
                )
                kwargs.pop("num_labels")
                pretrained_weights = cls._TASK_SPECIFIC_AUTO_CLASS.from_pretrained(
                    pretrained_model_name_or_path, *model_args, **kwargs
                )
                in_features = pretrained_weights.classifier.in_features
                has_bias = pretrained_weights.classifier.bias is not None
                pretrained_weights.classifier = nn.Linear(
                    in_features, provided_num_labels, bias=has_bias
                )
                pretrained_weights._init_weights(pretrained_weights.classifier)
                return cls._TASK_SPECIFIC_AUTO_CLASS.from_pretrained(
                    pretrained_model_name_or_path,
                    *model_args,
                    config=new_config,
                    state_dict=pretrained_weights.state_dict(),
                    **kwargs,
                )

        return cls._TASK_SPECIFIC_AUTO_CLASS.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    def _get_forward_params(self, model: PreTrainedModel) -> Tuple[str, ...]:
        forward_method_signature = inspect.signature(model.forward)
        model_specific_invalid_params = self._MODEL_TYPE_TO_INVALID_PARAMS.get(
            model.config.model_type, ()
        )
        model_specific_extra_params = self._MODEL_TYPE_TO_EXTRA_PARAMS.get(
            model.config.model_type, ()
        )
        task_and_model_params = {
            *self._TASK_SPECIFIC_FORWARD_PARAMS,
            *model_specific_extra_params,
        }
        return tuple(
            key
            for key in task_and_model_params
            if key in forward_method_signature.parameters
            and key not in model_specific_invalid_params
        )


ModelWrapper.register("from_pretrained", constructor="from_pretrained")(
    ModelWrapper
)
