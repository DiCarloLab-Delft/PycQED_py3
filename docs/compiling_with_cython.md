# Using Cythonbased files
@author: Adriaan

In order to compile cython files a setup.py file must be executed with the following command.
'''
python setup.py build_ext --inplace
'''

Most likely you will get a vcvarshall not found error when compiling on windows the first time. This is because cython needs MS visual studio. See also: https://github.com/cython/cython/wiki/CythonExtensionsOnWindows

The following gives a clear explanation on how to fix versions redirection
of the visual studio for compiling c++
https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_Cython_On_Anaconda_On_Windows?lang=en. 
It tells you 

1. Install MS visual studio with c++ compiler 
2. Edit distutils file hidden deep in the python folder 

MS visual studio 2015 has a free academic version that can be downloaded [here](
https://www.visualstudio.com/en-us/products/vs-2015-product-editions.aspx)
make sure you check c++ drivers when installing. If you forget you can start a c++ project and it will prompt you to install. 
