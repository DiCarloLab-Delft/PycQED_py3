init_all
X180 q0
.c

init_all
x q0
RO

# here we test if a full line comment gets removed
init_all
I q0
RO q1

Y90 q0  # Here we test if an inline comment gets removed
RO q0
