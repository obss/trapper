import inspect
from pathlib import Path
from typing import Dict, Tuple, Union

from torch import nn
from transformers import AutoConfig, PreTrainedModel
from transformers.utils import logging

from trapper.common import Registrable
from trapper.common.utils import add_property

logger = logging.get_logger(__name__)


class TransformerModel(PreTrainedModel, Registrable):
    """
    The `TransformerModel` is the base registrable model factory that is responsible
    for creating transformer models from the `Transformers` library. It does so by
    keeping a mapping from task name to the `auto_class_factory` (i.e. `auto)
    classes defined in the `Transformers` such as
    `AutoModelForSequenceClassification`. To make this work, we register the `auto`
    classes with their task name to the registry as a subclass of
    `TransformerModel`.

    This class does extra things depending on the `model_type` of the created
    task-specific class. These include:
        - inspecting the `forward` parameters of the model,
        - checking the invalid parameters i.e. the parameters that are required
        by most of the models for that task but not the model with that
        `model_type`,
        - the extra parameters required by the models with that `model_type`
        but are not required by other models in general.

    Then, the legitimate parameters are stored as `forward_params` attribute of
    the returned model. Thanks to that, `trapper.training.TransformerTrainer`
    can access that attribute to use as argument for the `model_input_keys`
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

    Below are the constructor parameters, which are left generic.
        *inputs ():
        **kwargs ():
    """
    default_implementation = "from_pretrained"
    _TASK_SPECIFIC_AUTO_CLASS = None
    _TASK_SPECIFIC_FORWARD_PARAMS: Tuple[str] = None

    _MODEL_TYPE_TO_INVALID_PARAMS: Dict[str, Tuple[str]] = {
        "roberta": ("token_type_ids",),
        "longformer": ("token_type_ids",),
        "distilbert": ("token_type_ids",),
        "bart": ("token_type_ids",),
    }

    _MODEL_TYPE_TO_EXTRA_PARAMS: Dict[str, Tuple[str]] = {
        "longformer": ("global_attention_mask",),
    }

    def __init__(self, *inputs, **kwargs):
        if self.__class__ == TransformerModel:
            raise EnvironmentError(
                "TransformerModel is designed to be a factory that can "
                "instantiate concrete models using `TransformerModel.from_params` "
                "method."
            )
        else:
            raise EnvironmentError(
                "Task-specific `TransformerModel` subclasses are designed to be "
                "instantiated using the `from_pretrained` method."
            )

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, Path],
            *model_args,
            **kwargs
    ) -> PreTrainedModel:
        # Handles architectural changes (e.g. for further training an already
        # fine-tuned downstream model on a new dataset) for the required
        # task-specific `auto` models.
        if cls == TransformerModel:
            raise EnvironmentError(
                "TransformerModel is designed to be a factory that can "
                "instantiate concrete models using `TransformerModel.from_params` "
                "method."
            )
        if (cls._TASK_SPECIFIC_AUTO_CLASS.__name__ ==
                "AutoModelForTokenClassification"):
            model = cls._create_token_classification_model(
                pretrained_model_name_or_path, *model_args, **kwargs)
        else:
            model = cls._TASK_SPECIFIC_AUTO_CLASS.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )

        cls._post_init(model)
        return model

    @classmethod
    def _create_token_classification_model(
            cls,
            pretrained_model_name_or_path: Union[str, Path],
            *model_args,
            **kwargs) -> PreTrainedModel:
        provided_num_labels = kwargs.get("num_labels")
        if provided_num_labels is not None:
            pretrained_model_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path)
            new_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, **kwargs)
            pretrained_num_labels = getattr(pretrained_model_config,
                                            "num_labels", None)
            if pretrained_num_labels is None:
                raise ValueError(f"Unexpected argument `num_labels` with "
                                 f"the value of {provided_num_labels}")
            if pretrained_num_labels != provided_num_labels:
                logger.warning(
                    f"Provided `num_labels` value ({provided_num_labels}) "
                    f"is different from the one found in the archived config "
                    f"({pretrained_num_labels}). The classifier head will have "
                    f"{provided_num_labels} labels and be initialized randomly!"
                )
                kwargs.pop("num_labels")
                pretrained_weights = cls._TASK_SPECIFIC_AUTO_CLASS.from_pretrained(
                    pretrained_model_name_or_path, *model_args, **kwargs
                )
                in_features = pretrained_weights.classifier.in_features
                has_bias = pretrained_weights.classifier.bias is not None
                pretrained_weights.classifier = nn.Linear(
                    in_features,
                    provided_num_labels,
                    bias=has_bias)
                pretrained_weights._init_weights(pretrained_weights.classifier)
                return cls._TASK_SPECIFIC_AUTO_CLASS.from_pretrained(
                    pretrained_model_name_or_path,
                    *model_args,
                    config=new_config,
                    state_dict=pretrained_weights.state_dict(),
                    **kwargs
                )

        return cls._TASK_SPECIFIC_AUTO_CLASS.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    @classmethod
    def _post_init(cls, model: PreTrainedModel):
        model.__forward_params = cls._get_forward_params(model)
        add_property(
            model,
            {
                "forward_params": lambda self: self.__forward_params,
            },
        )

    @classmethod
    def _get_forward_params(cls, model: PreTrainedModel) -> Tuple[str, ...]:
        forward_method_signature = inspect.signature(model.forward)
        model_specific_invalid_params = cls._MODEL_TYPE_TO_INVALID_PARAMS.get(
            model.config.model_type, ()
        )
        model_specific_extra_params = cls._MODEL_TYPE_TO_EXTRA_PARAMS.get(
            model.config.model_type, ()
        )
        task_and_model_params = {*cls._TASK_SPECIFIC_FORWARD_PARAMS,
                                 *model_specific_extra_params}
        return tuple(
            key
            for key in task_and_model_params
            if key in forward_method_signature.parameters
            and key not in model_specific_invalid_params
        )


TransformerModel.register("from_pretrained", constructor="from_pretrained")(
    TransformerModel
)
