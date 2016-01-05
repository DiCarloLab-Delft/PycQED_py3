# Import LabVIEW Client Script
from . import LabVIEW

# Import time module for time.sleep() function
import time
print("\nConnecting to LabVIEW...\n")
time.sleep(0.5)

# Connect to LabVIEW
LabVIEW.connect()

print("Connected\n")

time.sleep(0.5)
print("\nCalling Foo.Echo...\n")
time.sleep(0.5)

# Call Foo.Echo(0) four times
for x in range(1,5):
    print(LabVIEW.Foo.Echo(repr(x)))

time.sleep(0.5)
print("\nCalling Foo.Bar...\n")
time.sleep(0.5)

# Call Foo.Bar() four times
for x in range(1,5):
    print(LabVIEW.Foo.Bar(x))

time.sleep(0.5)
print("\nDisconnecting...\n")

# Disconnect from LabVIEW
LabVIEW.disconnect()

time.sleep(0.5)
print("Disconnected\n")
time.sleep(0.5)
print("Good Bye!")
time.sleep(2)