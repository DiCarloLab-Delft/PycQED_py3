# Using Cythonbased files
@author: Adriaan

I have sped up some parts of the code by writing them in Cython a language
that is a mixture of C and python.

In order to compile these a setup.py file must be executed with the following command.

'''
python setup.py build_ext --inplace
'''

On windows computers a c-compiler is needed to be able to compile cython .pyx
files.

Look at "https://github.com/cython/cython/wiki/CythonExtensionsOnWindows" for instructions on installing a c compiler.

Scroll down to the section called
'''
Using Windows SDK C/C++ compiler (works for all Python versions)
'''
And look for the following section:


>to install or build a package, you do have to start a SDK Command Prompt or CMD Shell and set some environment variables.

>By default the shell starts in at the installation path of the Windows SDK (C:\Program Files\Microsoft SDKs\Windows\v7.0. There, we have to to two things:
>
>    Tell distutils / setuptools to use the Microsoft SDK compiler
>    Tell the compiler to compile a 32 or 64 bit release and whether it should be a debug or a release build

>Thus, we have to enter two commands

>set DISTUTILS_USE_SDK=1
>setenv /x64 /release
'''
