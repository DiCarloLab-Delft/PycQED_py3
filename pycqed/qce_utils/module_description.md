Purpose QCE-Utils
===
This sub-module is a direct port from the standalone QCoExtended repository.
Only well established functionality from the standalone repository is transferred to PycQED.

- Custom exceptions. Good practice to have a library of custom exceptions, these help identify which exceptions are raised in what situations. The most used one is 'InterfaceMethodException' which is raised if an (ABC) interface abstractmethod is not implemented. 

Control Interfaces
===
Contains:
- Channel identifier interfaces. These are identifiers for individual qubits, edges and feedlines.
- Connectivity interfaces. These describe building blocks like nodes and edges, but also larger structures like connectivity layers and stacks (multiple layers). Together they combine in the Device layer interface, exposing get methods for relationships between nodes and edges.
- Surface-code specific connectivity interfaces. These extend the connectivity interfaces by exposing surface-code specific terminology like parity groups and (qubit) frequency groups.
- Surface-code connectivity. This implements the above-mentioned interfaces to create a so called 'Surface-17' connectivity layer. This can be used throughout to obtain qubit-to-qubit relations by simply referring to their corresponding identifiers. An example of its use is during multi-qubit experiments which use inter-dependent flux trajectories (like 'flux-dance cycles').  