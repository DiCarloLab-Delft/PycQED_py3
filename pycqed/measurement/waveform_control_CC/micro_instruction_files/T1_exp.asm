mov r15, 20000   # r15 stores the cycle time (heartbeat), 100 us
mov r7, 400      # sweep step, 1 us
mov r13, 16000   # r13 stores the max $T_{inteval}$, 80 us
mov r14, 0
ProgramEntry:   WaitReg r15

# Experiment: repeat the rounds for infinite times
T1Exp_Start:    addi r2, r7, 0                 # r2 = Inteval $T_{inteval}$ between X180 Pulse and measurement

# One round: vary the value of the interval
Round_Start:    Sub r5, r15, r2                # r5 = RelaxationTime (20 us) – $T_{inteval}$
                WaitReg r5
                Pulse 1000, 0000, 0000         # Pulse 0 of AWG0 corresponds to X180 pulse
                Trigger 0111111, 10
                Trigger 1000000, 10
                WaitReg r2                     # Wait for $T_{inteval}$  (1 us)
                add r2, r2, r7                 # Increase $T_{inteval}$ by one step (1 us)
                Pulse 0000, 1111, 0000         # pulse 7 of AWG1 corresponds to Readout pulse
                Trigger 0111111, 100
                Measure                        # Start measurement integration
                BNE r2, r13, Round_Start       # sweep the interval $T_{inteval}$ from 1 us to max $T_{inteval}$ (5 us)
                Trigger 1000000, 1000
                addi r5, r5, 1
                beq r14, r14, T1Exp_Start       # Infinite loop
