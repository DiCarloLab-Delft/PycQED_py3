# -------------------------------------------
# Customized exceptions for better maintainability
# -------------------------------------------
import numpy as np


class InterfaceMethodException(Exception):
    """
    Raised when the interface method is not implemented.
    """


class WeakRefException(Exception):
    """
    Raised when weak visa-instance reference is being retrieved which is not available.
    """


class ModelParameterException(Exception):
    """
    Raised when model-parameter class is being constructed using an inconsistent amount of parameters.
    """


class ModelParameterSubClassException(Exception):
    """
    Raised when model-parameter does not sub class the expected model-parameter class.
    """


class KeyboardFinish(KeyboardInterrupt):
    """
    Indicates that the user safely aborts/interrupts terminal process.
    """


class IdentifierException(Exception):
    """
    Raised when (qubit) identifier is not correctly handled.
    """


class InvalidProminenceThresholdException(Exception):
    """
    Raised when dynamic prominence threshold for peak detection is inconclusive.
    """


class EnumNotDefinedException(Exception):
    """
    Raised when undefined enum is detected.
    """


class EvaluationException(Exception):
    """
    Raised when optimizer parameters have not yet been evaluated.
    """


class OverloadSignatureNotDefinedException(Exception):
    """
    Raised when overload signature for specific function is not defined or recognized.
    Search-keys: overload, dispatch, multipledispatch, type casting.
    """


class ArrayShapeInconsistencyException(Exception):
    """
    Raised when the shape of arrays are inconsistent or incompatible with each other.
    """

    # region Static Class Methods
    @staticmethod
    def format_arrays(x: np.ndarray, y: np.ndarray) -> 'ArrayShapeInconsistencyException':
        return ArrayShapeInconsistencyException(f'Provided x-y arrays are do not have the same shape: {x.shape} != {y.shape}')
    # endregion


class ArrayNotComplexException(Exception):
    """
    Raised when not all array elements are complex.
    """


class StateEvaluationException(Exception):
    """
    Raised when state vector evaluation (expression to real float) fails.
    """


class StateConditionEvaluationException(Exception):
    """
    Raised when state vector condition evaluation fails.
    """


class WrapperException(Exception):
    """
    Raised any form of exception is needed within wrapper implementation.
    """


class InvalidPointerException(Exception):
    """
    Raised when file-pointer is invalid (path-to-file does not exist).
    """


class SerializationException(Exception):
    """
    Raised when there is a problem serializing an object.
    """


class HDF5ItemTypeException(Exception):
    """
    Raised when type from an item inside hdf5-file group is not recognized.
    """


class DataGenerationCompleteException(Exception):
    """
    Raised when upper bound of data generation has been reached.
    """


class DataInconclusiveException(Exception):
    """
    Raised when data is incomplete or inconclusive.
    """


class LinspaceBoundaryException(Exception):
    """
    Raised when the boundary values of a linear space sampler are identical.
    """


class TransmonFrequencyRangeException(Exception):
    """
    Raised when frequency falls outside the range of Transmon frequency.
    """

    # region Static Class Methods
    @staticmethod
    def format_arrays(qubit_max_frequency: float, target_frequency: float) -> 'TransmonFrequencyRangeException':
        return TransmonFrequencyRangeException(f'Target frequency value {target_frequency*1e-9:2.f} [GHz] not within qubit frequency range: 0-{qubit_max_frequency*1e-9:2.f} [GHz].')
    # endregion


class DimensionalityException(Exception):
    """
    Raised when dataset dimensionality is unknown or does not match expected.
    """


class FactoryRequirementNotSatisfiedException(Exception):
    """
    Raised when factory deployment requirement is not satisfied.
    """


class NoSamplesToEvaluateException(Exception):
    """
    Raised when functionality depending on non-zero number of samples fails.
    """

    # region Static Class Methods
    @staticmethod
    def format_for_model_driven_agent() -> 'NoSamplesToEvaluateException':
        return NoSamplesToEvaluateException(f"Agent can not perform sample evaluation with 0 samples. Ensure to execute 'self.next(state: CoordinateResponsePair)' with at least a single state before requesting model evaluation.")
    # endregion


class HardwareModuleChannelException(Exception):
    """
    Raised when module channel index is out of range.
    """


class OperationTypeException(Exception):
    """
    Raised when operation type does not correspond to expected type.
    """


class RegexGroupException(Exception):
    """
    Raised when regex match does not find intended group.
    """


class IsolatedGroupException(Exception):
    """
    Raised when a list of grouped elements are not isolated. Members from one group are shared in another group.
    """


class PeakDetectionException(Exception):
    """
    Raised when the number of detected peaks is not sufficient.
    """


class FactoryManagerKeyException(Exception):
    """
    Raised when the key is not present in the factory-manager components.
    """

    # region Static Class Methods
    @staticmethod
    def format_log(key, dictionary) -> 'FactoryManagerKeyException':
        return FactoryManagerKeyException(f'Provided key: {key} is not present in {dictionary}.')
    # endregion


class RequestNotSupportedException(FactoryManagerKeyException):
    """
    Raised when (measurement) execution request is not support or can not be handled.
    """


class IncompleteParameterizationException(Exception):
    """
    Raised when operation is not completely parameterized.
    """


class ElementNotIncludedException(Exception):
    """
    Raised when element (such as IQubitID, IEdgeID or IFeedlineID) is not included in the connectivity layer.
    """


class GenericTypeException(Exception):
    """
    Raised when generic type is not found or supported.
    """

    # region Static Class Methods
    @staticmethod
    def format_log(generic_type: type) -> 'GenericTypeException':
        return GenericTypeException(f'Generic type : {generic_type} is not supported.')
    # endregion
