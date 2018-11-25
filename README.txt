# 
# Copyright (c) 2018 Nathan Lay (enslay@gmail.com)
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#
# Nathan Lay
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# National Institutes of Health
# March 2017
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#######################################################################
# Introduction                                                        #
#######################################################################
ComputeBValue is a tool that can calculate b-value images from one or 
more given b-value images using one of four different models. These
include: mono exponential, intravoxel incoherent motion (IVIM), 
diffusion  kurtosis (DK), and a combined DK and IVIM model. It can read 
vendor-specific formatted diffusion b-values from DICOM and can write 
b-value, Apparent Diffusion Coefficient (ADC), kurtosis, and perfusion 
fraction images to medical image formats like MetaIO as well as DICOM. 
As diffusion MRI sequences are often interleaves with b-values, this 
tool supports extracting individual b-value image volumes from such 
sequences!

NOTE: This tool is especially designed for prostate mpMRI imaging. 
Model assumptions follow those in the following paper:

Grant, Kinzya B., et al. "Comparison of calculated and acquired high 
b value diffusion-weighted imaging in prostate cancer." 
Abdominal imaging 40.3 (2015): 578-586.

The source code is partly based on SortDicomFiles and 
StandardizeBValues from which this README is also partly based.
https://github.com/nslay/SortDicomFiles
https://github.com/nslay/StandardizeBValue

#######################################################################
# Installing                                                          #
#######################################################################
If a precompiled version is available for your operating system, either
extract the archive where it best suits you, or copy the executable to
the desired location.

Once installed, the path to ComputeBValue should be added to PATH.

Windows: Right-click on "Computer", select "Properties", then select
"Advanced system settings." In the "System Properties" window, look
toward the bottom and click the "Environment Variables" button. Under
the "System variables" list, look for "Path" and select "Edit." Append
the ";C:\Path\To\Folder" where "C:\Path\To\Folder\ComputeBValue.exe"
is the path to the executable. Click "OK" and you are done.

Linux: Use a text editor to open the file ~/.profile or ~/.bashrc
Add the line export PATH="${PATH}:/path/to/folder" where
/path/to/folder/ComputeBValue is the path to the executable. Save
the file and you are done.

ComputeBValue can also be compiled from source. Instructions are
given in the "Building from Source" section.

#######################################################################
# Usage                                                               #
#######################################################################
Once installed, ComputeBValue must be run from the command line. In
Windows this is accomplished using Command Prompt or PowerShell.
Unix-like environments include terminals where commands may be issued.

WINDOWS TIP: Command Prompt can be launched conveniently in a folder
holding shift and right clicking in the folder's window and selecting
"Open command window here."

ComputeBValue accepts one or more given DICOM folders or DICOM files.
DICOM files are used to deduce the folder and series UID of the DICOM
series.

As a quick start example

ComputeBValue -b 1500 mono C:\Path\To\6-ep2ddifftraDYNDIST-03788

will compute a b-1500 image on ProstateX-0026's diffusion series using
the mono exponential model. The output of this command might look like

Info: Loaded b = 50
Info: Loaded b = 400
Info: Loaded b = 800
Info: Calculating b-1500
Info: Saving b-value image to 'output.mha' ...

You may change the output path using the -o flag. Providing an output
file path without a file extension will result in a DICOM folder. For
example

ComputeBValue -b 1500 -o B1500Image mono 6-ep2ddifftraDYNDIST-03788

will produce a DICOM folder with the following file hierarchy

B1500Image/
+-- 1.dcm
+-- 2.dcm
+-- 3.dcm
+-- 4.dcm
...
+-- 19.dcm

There are additional flags for saving the calculated ADC (-a),
perfusion fraction (-p), and kurtosis image (-k). You may additionally
compress the output images (-c) as well as change the output DICOM 
series number for the calculated b-value image (-n). By default, the 
series number is 13701 for the calculated b-value image. Other images 
share a similar series number offset by values of 1-3 (e.g. 13702-4).

Lastly, ComputeBValue provides the below usage message when
provided with the -h flag or no arguments. It's useful if you
forget.

Usage: ComputeBValue [-achkp] [-o outputPath] [-n seriesNumber] 
-b targetBValue mono|ivim|dk|dkivim diffusionFolder1|diffusionFile1 
[diffusionFolder2|diffusionFile2 ...]

Options:
-a -- Save calculated ADC. The output path will have _ADC appended 
(folder --> folder_ADC or file.ext --> file_ADC.ext).
-b -- Target b-value to calculate.
-c -- Compress output.
-h -- This help message.
-k -- Save calculated kurtosis image. The output path will have 
_Kurtosis appended.
-n -- Series number for calculated b-value image (default 13701).
-o -- Output path which may be a folder for DICOM output or a medical 
image format file.
-p -- Save calculated perfusion fraction image. The output path will 
have _Perfusion appended.

#######################################################################
# Models                                                              #
#######################################################################
This section briefly describes the how the models are implemented. All
models solve a least-squares minimization problem of the form:

\ell(\theta) = \sum_{b \in B} (f(\theta) - \log(S_b/S_0))^2

where B is the set of b-values, S_b is a pixel intensity from a b-value
image, and f(\theta) is the model predicting exponential terms. In 
addition to \theta, one S_b may be an unknown value to optimize for.

The models are discussed in more detail in the following paper:

Grant, Kinzya B., et al. "Comparison of calculated and acquired high 
b value diffusion-weighted imaging in prostate cancer." 
Abdominal imaging 40.3 (2015): 578-586.


# Mono Exponential
The mono exponential model relates S_b and D through the expression:
S_b = S_0 exp(-bD)

where D is the ADC value.

This gives the parameters \theta = \{ D \} and the model function
f(\theta) = -bD

NOTE: This model can cope with the absence of S_0 using a mathematical
trick. In this model, the ratio of two b-value image intensities is
S_a / S_b = exp(-(a-b)D)

With some rearrangement we have:

S_a = S_b exp(-(a-b)D)

Hence, by simply shifting by the minimum b-value, this model may be
used to calculate any b-value image in the absence of a proper b-0
image (it may even be used to calculate the b-0 image!).

# Intravoxel Incoherent Motion (IVIM)
The IVIM model relates S_b, D, and f through the expression:
S_b = S_0 (1-f) exp(-bD)

where D is the ADC value and f is the perfusion fraction term

This gives the parameters \theta = \{ D, f \} and the model function
f(\theta) = ln(1-f) - bD

NOTE: This model requires a b-0 image. In the absence of a b-0 image
the mono exponential model is used to calculate it.

# Diffusion Kurtosis (DK)
The DK model relates S_b, D and K through the expression:
S_b = S_0 exp(-bD + K(bD)^2/6)

where D is the ADC value and K is the kurtosis term.

This gives the parameters \theta = \{ D, K \} and the model function
f(\theta) = -bD + K(bD)^2/6

NOTE: This model requires a b-0 image. In the absence of a b-0 image
the mono exponential model is used to calculate it.

# Combined Model (DK+IVIM)
The DK+IVIM model relates S_b, D, K and f through the expression:
S_b = S_0 (1-f) exp(-bD + K(bD)^2/6)

where D is the ADC value, K is the kurtosis term and f is the perfusion
fraction term.

This gives the parameters \theta = \{ D, K, f \} and the model function
f(\theta) = ln(1-f) - bD + K(bD)^2/6

NOTE: This model requires a b-0 image. In the absence of a b-0 image
the mono exponential model is used to calculate it.

NOTE: Depending on your images and assumptions, some of these models 
may be inappropriate!


#######################################################################
# Building from Source                                                #
#######################################################################
To build ComputeBValue from source, you will need a recent version of
CMake, a C++11 compiler, and InsightToolkit version 4 or later.

First extract the source code somewhere. Next create a separate
directory elsewhere. This will serve as the build directory. Run CMake
and configure the source and build directories as chosen. More
specifically

On Windows:
- Run cmake-gui (Look in Start Menu) and proceed from there.

On Unix-like systems:
- From a terminal, change directory to the build directory and then
run:

ccmake /path/to/source/directory

In both cases, "Configure." If you encounter an error, set ITK_DIR
and then run "Configure" again. Then select "Generate." On Unix-like
systems, you may additionally want to set CMAKE_BUILD_TYPE to "Release"

NOTE: ITK_DIR should be set to the cmake folder in the ITK lib
folder. For example: /path/to/ITK/lib/cmake/ITK-4.13/

Visual Studio:
- Open the solution in the build directory and build ComputeBValue.
Make sure you select "Release" mode.

Unix-like systems:
- Run the "make" command.

ComputeBValue has been successfully built and tested with:
Microsoft Visual Studio 2017 on Windows 10 Professional
Clang 6.0.1 on FreeBSD 11.2-STABLE

using ITK versions:
ITK 4.9
ITK 4.13

#######################################################################
# Caveats                                                             #
#######################################################################
Using absolute paths to Windows shares (i.e. \\name\folder) could cause
problems since BaseName() and DirName() have not yet implemented
parsing these kinds of paths.

