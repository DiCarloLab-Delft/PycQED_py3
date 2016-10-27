mov r0, 20000   # r0 stores the cycle time (heartbeat), 100 us
mov r1, 0       # sets the inter pulse wait to 0
mov r14, 0      # r14 stores number of repetitions

# Experiment: repeat the rounds for infinite times
Exp_Start:
    WaitReg r0
    Trigger 1111111, 1            # Marker for the initial pulse
    mov r1, 10
    WaitReg r1
    Trigger 1111111, 1            # Marker for the RO (using same for test)
    Measure                        # Start measurement integration

    WaitReg r0
    Trigger 1111111, 1            # Marker for the initial pulse
    mov r1, 20
    WaitReg r1
    Trigger 1111111, 1            # Marker for the RO (using same for test)
    Measure                        # Start measurement integration

    WaitReg r0
    Trigger 1111111, 1            # Marker for the initial pulse
    mov r1, 30
    WaitReg r1
    Trigger 1111111, 1            # Marker for the RO (using same for test)
    Measure                        # Start measurement integration

    WaitReg r0
    Trigger 1111111, 1            # Marker for the initial pulse
    mov r1, 40
    WaitReg r1
    Trigger 1111111, 1            # Marker for the RO (using same for test)
    Measure                        # Start measurement integration

    WaitReg r0
    Trigger 1111111, 1            # Marker for the initial pulse
    mov r1, 50
    WaitReg r1
    Trigger 1111111, 1            # Marker for the RO (using same for test)
    Measure                        # Start measurement integration

    beq r14, r14, Exp_Start       # Infinite loop
