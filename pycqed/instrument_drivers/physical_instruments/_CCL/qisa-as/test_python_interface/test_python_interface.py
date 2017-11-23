from pyQisaAs import QISA_Driver

inputFilename = '../qisa_test_assembly/test_assembly.qisa'
outputFilename = 'test_assembly.out'
disassemblyOutputFilename = 'test_disassembly.out'

print ("QISA_AS Version: ", QISA_Driver.getVersion())

print ("Retrieving QISA Opcode specification...")
print (QISA_Driver.dumpOpcodeSpecification())

driver = QISA_Driver()

driver.enableScannerTracing(False)
driver.enableParserTracing(False)
driver.setVerbose(True)

print ("parsing file ", inputFilename)
success = driver.parse(inputFilename)

if success:
    print ("Generated instructions:")
    instHex = driver.getInstructionsAsHexStrings(False)
    for inst in instHex:
        print ("  " + inst)
    print()

    print ("Generated instructions, including binary:")
    instHexBin = driver.getInstructionsAsHexStrings(True)
    for inst in instHexBin:
        print ("  " + inst)
    print()

    print ("Saving instructions to file: ", outputFilename)
    success = driver.save(outputFilename)
    if success:
        print ("Disassembling saved instructions from file: ", outputFilename)

        success = driver.disassemble(outputFilename)
        if success:
            print(driver.getDisassemblyOutput())

            print ("Saving disassembly to file: ", disassemblyOutputFilename)
            success = driver.save(disassemblyOutputFilename)
            if not success:
                print ("Saving disassembly terminated with errors:")
                print (driver.getLastErrorMessage())
        else:
            print ("Disassembly terminated with errors:")
            print (driver.getLastErrorMessage())
    else:
        print ("Saving assembly terminated with errors:")
        print (driver.getLastErrorMessage())
else:
    print ("Assembly terminated with errors:")
    print (driver.getLastErrorMessage())
