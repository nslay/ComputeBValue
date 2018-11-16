/*-
 * Copyright (c) 2018 Nathan Lay (enslay@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*-
 * Nathan Lay
 * Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
 * National Institutes of Health
 * March 2017
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdlib>
#include <cctype>
#include <memory>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <utility>
#include "Common.h"
#include "ADVar.h"
#include "strcasestr.h"
#include "bsdgetopt.h"

// ITK stuff
#include "itkGDCMImageIO.h"
#include "itkMetaDataDictionary.h"
#include "itkMetaDataObject.h"
#include "itkRGBPixel.h"
#include "itkRGBAPixel.h"

// VNL stuff
#include "vnl/vnl_cost_function.h"
#include "vnl/vnl_vector.h"
#include "vnl/vnl_cross.h"
#include "vnl/algo/vnl_lbfgsb.h"

void Usage(const char *p_cArg0) {
  std::cerr << "Usage: " << p_cArg0 << " [-ahkp] [-o outputPath] -b targetBValue mono|ivim|dk diffusionFolder1|diffusionFile1 [diffusionFolder2|diffusionFile2 ...]" << std::endl;
  std::cerr << "\nOptions:" << std::endl;
  std::cerr << "-a -- Save calculated ADC. The output path will have _ADC appended (folder --> folder_ADC or file.ext --> file_ADC.ext)." << std::endl;
  std::cerr << "-b -- Target b-value to calculate." << std::endl;
  std::cerr << "-h -- This help message." << std::endl;
  std::cerr << "-k -- Save calculated kurtosis image. The output path will have _Kurtosis appended." << std::endl;
  std::cerr << "-o -- Output path which may be a folder for DICOM output or a medical image format file." << std::endl;
  std::cerr << "-p -- Save calculated perfusion fraction image." << std::endl;
  exit(1);
}

// Needed for sorting the B value dictionaries
template<typename RealType>
bool GetOrientationMatrix(const itk::MetaDataDictionary &clDicomTags, vnl_matrix_fixed<RealType, 3, 3> &clOrientation);

template<typename RealType>
bool GetOrigin(const itk::MetaDataDictionary &clDicomTags, vnl_vector_fixed<RealType, 3> &clOrigin);

// Change name from ComputeDiffusionBValue* to GetDiffusionBValue* (since this program actually calculates B value images!)
double GetDiffusionBValue(const itk::MetaDataDictionary &clDicomTags);
double GetDiffusionBValueSiemens(const itk::MetaDataDictionary &clDicomTags);
double GetDiffusionBValueGE(const itk::MetaDataDictionary &clDicomTags);
double GetDiffusionBValueProstateX(const itk::MetaDataDictionary &clDicomTags); // Same as Skyra and Verio (and many others I can't remember)
double GetDiffusionBValuePhilips(const itk::MetaDataDictionary &clDicomTags);

std::map<double, std::vector<std::string>> ComputeBValueFileNames(const std::string &strPath, const std::string &strSeriesUID = std::string());

template<typename PixelType>
std::map<double, typename itk::Image<PixelType, 3>::Pointer> LoadBValueImages(const std::string &strPath, const std::string &strSeriesUID = std::string());

class BValueModel : public vnl_cost_function {
public:
  typedef itk::Image<short, 3> ImageType;
  typedef std::map<double, ImageType::Pointer> ImageMapType;
  typedef vnl_lbfgsb SolverType;

  virtual ~BValueModel() = default;

  virtual bool Good() const { 
    if (GetOutputPath().empty() || GetTargetBValue() < 0.0 || m_mImagesByBValue.size() < 2)
      return false;

    auto itr = GetImages().begin();
    auto prevItr = itr++;

    if (prevItr->first < 0.0 || !prevItr->second)
      return false;

    while (itr != GetImages().end()) {
      if (!itr->second || itr->second->GetBufferedRegion() != prevItr->second->GetBufferedRegion())
        return false;

      prevItr = itr++;
    }

    return true;
  }

  // Returns true if the model supports calculating this image
  virtual bool SaveADC() { return false; }
  virtual bool SavePerfusion() { return false; }
  virtual bool SaveKurtosis() { return false; }

  void SetOutputPath(const std::string &strOutputPath) {
    m_strOutputPath = strOutputPath;
  }

  const std::string & GetOutputPath() const { return m_strOutputPath; }

  std::string GetOutputPathWithPrefix(const std::string &strPrefix) const {
    const std::string strExtension = GetExtension(GetOutputPath());

    if (strExtension.empty())
      return StripTrailingDelimiters(GetOutputPath()) + strPrefix;

    return StripExtension(GetOutputPath()) + strPrefix + strExtension;
  }

  virtual bool SetImages(const ImageMapType &mImagesByBValue) {
    m_mImagesByBValue = mImagesByBValue;
    return m_mImagesByBValue.size() > 0;
  }

  void SetTargetBValue(double dTargetBValue) { m_dTargetBValue = dTargetBValue; }
  double GetTargetBValue() const { return m_dTargetBValue; }

  virtual bool Run();

  const ImageMapType & GetImages() const { return m_mImagesByBValue; }

protected:
  BValueModel(int iNumberOfUnknowns)
  : vnl_cost_function(iNumberOfUnknowns), m_clSolver(*this) { }

  template<typename PixelType>
  typename itk::Image<PixelType, 3>::Pointer NewImage() const;

  virtual double Solve(const itk::Index<3> &clIndex) = 0; // Return b-value or negative value for failure

  virtual bool SetLogIntensities(const itk::Index<3> &clIndex);

  double MinBValue() const {
    if (GetImages().empty())
      return -1.0;

    return GetImages().begin()->first;
  }

  const std::vector<std::pair<double, double>> & GetLogIntensities() const { return m_vBValueAndLogIntensity; }

  SolverType & GetSolver() { return m_clSolver; }

private:
  SolverType m_clSolver;
  ImageMapType m_mImagesByBValue;
  std::string m_strOutputPath;
  double m_dTargetBValue = -1.0;
  std::vector<std::pair<double, double>> m_vBValueAndLogIntensity;
};

class MonoExponentialModel : public BValueModel {
public:
  typedef BValueModel SuperType;
  typedef ADVar<double, 2> ADVarType;

  MonoExponentialModel()
  : SuperType(ADVarType::GetNumIndependents()) { }

  virtual ~MonoExponentialModel() = default;

  virtual bool SaveADC() override { return m_bSaveADC = true; }

  std::string GetADCOutputPath() const { return GetOutputPathWithPrefix("_ADC"); }

  virtual bool Run() override;

  // Since we use automatic differentiation, let's do both operations simultaneously...
  virtual void compute(const vnl_vector<double> &clX, double *p_dF, vnl_vector<double> *p_clG) override;

protected:
  virtual double Solve(const itk::Index<3> &clIndex) override;

private:
  bool m_bSaveADC = false;
  ImageType::Pointer m_p_clADCImage;
};

class IVIMModel : public BValueModel {
public:
  typedef BValueModel SuperType;
  typedef ADVar<double, 3> ADVarType;
  typedef itk::Image<float, 3> FloatImageType;

  IVIMModel()
  : SuperType(ADVarType::GetNumIndependents()) { }

  virtual ~IVIMModel() = default;

  virtual bool SaveADC() override { return m_bSaveADC = true; }
  virtual bool SavePerfusion() override { return m_bSavePerfusion = true; }

  std::string GetADCOutputPath() const { return GetOutputPathWithPrefix("_ADC"); }
  std::string GetPerfusionOutputPath() const { return GetOutputPathWithPrefix("_Perfusion"); }

  virtual bool Run() override;

  // Since we use automatic differentiation, let's do both operations simultaneously...
  virtual void compute(const vnl_vector<double> &clX, double *p_dF, vnl_vector<double> *p_clG) override;

protected:
  virtual double Solve(const itk::Index<3> &clIndex) override;

private:
  bool m_bSaveADC = false;
  bool m_bSavePerfusion = false;
  ImageType::Pointer m_p_clADCImage;
  FloatImageType::Pointer m_p_clPerfusionImage;
};

int main(int argc, char **argv) {
  const char * const p_cArg0 = argv[0];

  bool bSaveADC = false;
  bool bSavePerfusion = false;
  bool bSaveKurtosis = false;

  double dBValue = -1.0;

  std::string strOutputPath = "output.mha";

  int c = 0;
  while ((c = getopt(argc, argv, "ab:hko:p")) != -1) {
    switch (c) {
    case 'a':
      bSaveADC = true;
      break;
    case 'b':
      dBValue = FromString<double>(optarg, -1.0);
      if (dBValue < 0.0)
        Usage(p_cArg0);
      break;
    case 'h':
      Usage(p_cArg0);
      break;
    case 'k':
      bSaveKurtosis = true;
      break;
    case 'o':
      strOutputPath = optarg;
      break;
    case 'p':
      bSavePerfusion = true;
      break;
    case '?':
    default:
      Usage(p_cArg0);
    }
  }

  argc -= optind;
  argv += optind;

  if (argc < 2 || dBValue < 0.0)
    Usage(p_cArg0);

  std::string strModel = argv[0];
  std::unique_ptr<BValueModel> p_clModel;

  if (strModel == "mono") {
    p_clModel = std::make_unique<MonoExponentialModel>();
  }
  else if (strModel == "ivim") {
    p_clModel = std::make_unique<IVIMModel>();
  }
  else {
    std::cerr << "Error: Unknown model type '" << strModel << "'.\n" << std::endl;
    Usage(p_cArg0);
  }

  typedef itk::Image<short, 3> ImageType;

  std::map<double, ImageType::Pointer> mImagesByBValue;

  for (int i = 1; i < argc; ++i) {
    std::map<double, ImageType::Pointer> mTmp = LoadBValueImages<ImageType::PixelType>(argv[i]);

    for (const auto &stPair : mTmp) {
      std::cout << "Info: Loaded b = " << stPair.first << std::endl;
      SaveImg<short, 3>(stPair.second, "b" + std::to_string(stPair.first) + ".mha");

      if (!mImagesByBValue.emplace(stPair).second) {
        std::cerr << "Error: Duplicate b-value " << stPair.first << " from series " << argv[i] << std::endl;
        return -1;
      }
    }
  }

  p_clModel->SetTargetBValue(dBValue);
  p_clModel->SetImages(mImagesByBValue);
  p_clModel->SetOutputPath(strOutputPath);

  if (bSaveADC && !p_clModel->SaveADC())
    std::cerr << "Warning: '" << strModel << "' model does not support saving ADC images." << std::endl;

  if (bSavePerfusion && !p_clModel->SavePerfusion())
    std::cerr << "Warning: '" << strModel << "' model does not support saving perfusion fraction images." << std::endl;

  if (bSaveKurtosis && !p_clModel->SaveKurtosis())
    std::cerr << "Warning: '" << strModel << "' model does not support saving kurtosis images." << std::endl;

  std::cout << "Info: Calculating b-" << dBValue << std::endl;

  if (!p_clModel->Run()) {
    std::cerr << "Error: Failed to compute b-value image." << std::endl;
    return -1;
  }

  return 0;
}

template<typename RealType>
bool GetOrientationMatrix(const itk::MetaDataDictionary &clDicomTags, vnl_matrix_fixed<RealType, 3, 3> &clOrientation) {
  std::vector<RealType> vImageOrientationPatient; // 0020|0037

  if (!ExposeStringMetaData(clDicomTags, "0020|0037", vImageOrientationPatient) || vImageOrientationPatient.size() != 6)
    return false;

  vnl_vector_fixed<RealType, 3> clX, clY, clZ;

  clX.copy_in(vImageOrientationPatient.data());
  clY.copy_in(vImageOrientationPatient.data() + 3);
  clZ = vnl_cross_3d(clX, clY);

  clOrientation[0][0] = clX[0];
  clOrientation[1][0] = clX[1];
  clOrientation[2][0] = clX[2];

  clOrientation[0][1] = clY[0];
  clOrientation[1][1] = clY[1];
  clOrientation[2][1] = clY[2];

  clOrientation[0][2] = clZ[0];
  clOrientation[1][2] = clZ[1];
  clOrientation[2][2] = clZ[2];

  return true;
}

template<typename RealType>
bool GetOrigin(const itk::MetaDataDictionary &clDicomTags, vnl_vector_fixed<RealType, 3> &clOrigin) {
  std::vector<RealType> vImagePositionPatient; // 0020|0032

  if (!ExposeStringMetaData(clDicomTags, "0020|0032", vImagePositionPatient) || vImagePositionPatient.size() != clOrigin.size())
    return false;

  clOrigin.copy_in(vImagePositionPatient.data());

  return true;
}

double GetDiffusionBValue(const itk::MetaDataDictionary &clDicomTags) {
  double dBValue = -1.0;

  if (ExposeStringMetaData(clDicomTags, "0018|9087", dBValue))
    return dBValue;

  std::string strPatientName;
  std::string strPatientId;
  std::string strManufacturer;

  ExposeStringMetaData(clDicomTags, "0010|0010", strPatientName);
  ExposeStringMetaData(clDicomTags, "0010|0020", strPatientId);

  if (strcasestr(strPatientName.c_str(), "prostatex") != nullptr || strcasestr(strPatientId.c_str(), "prostatex") != nullptr)
    return GetDiffusionBValueProstateX(clDicomTags);

  if (!ExposeStringMetaData(clDicomTags, "0008|0070", strManufacturer)) {
    std::cerr << "Error: Could not determine manufacturer." << std::endl;
    return -1.0;
  }

  if (strcasestr(strManufacturer.c_str(), "siemens") != nullptr)
    return GetDiffusionBValueSiemens(clDicomTags);
  else if (strcasestr(strManufacturer.c_str(), "ge") != nullptr)
    return GetDiffusionBValueGE(clDicomTags);
  else if (strcasestr(strManufacturer.c_str(), "philips") != nullptr)
    return GetDiffusionBValuePhilips(clDicomTags);

  return -1.0;
}

double GetDiffusionBValueSiemens(const itk::MetaDataDictionary &clDicomTags) {
  std::string strModel;

  if (itk::ExposeMetaData(clDicomTags, "0008|1090", strModel)) {
    if ((strcasestr(strModel.c_str(), "skyra") != nullptr || strcasestr(strModel.c_str(), "verio") != nullptr)) {
      double dBValue = GetDiffusionBValueProstateX(clDicomTags);

      if (dBValue >= 0.0)
        return dBValue;
    }
  }

  gdcm::CSAHeader clCSAHeader;

  if (!ExposeStringMetaData(clDicomTags, "0029|1010", clCSAHeader)) // Nothing to do
    return -1.0;

  std::string strTmp;

  if (ExposeCSAMetaData(clCSAHeader, "B_value", strTmp))
    return FromString(strTmp, -1.0);

  return -1.0;
}

double GetDiffusionBValueGE(const itk::MetaDataDictionary &clDicomTags) {
  std::string strValue;
  if (!ExposeStringMetaData(clDicomTags, "0043|1039", strValue))
    return -1.0; // Nothing to do

  size_t p = strValue.find('\\');
  if (p == std::string::npos)
    return -1.0; // Not sure what to do

  strValue.erase(p);

  std::stringstream valueStream;
  valueStream.str(strValue);

  double dValue = 0.0;

  if (!(valueStream >> dValue) || dValue < 0.0) // Bogus value
    return -1.0;

  // Something is screwed up here ... let's try to remove the largest significant digit
  if (dValue > 4000.0) {
    p = strValue.find_first_not_of(" \t0");

    strValue.erase(strValue.begin(), strValue.begin()+p+1);

    valueStream.clear();
    valueStream.str(strValue);

    if (!(valueStream >> dValue) || dValue < 0.0 || dValue > 4000.0)
      return -1.0;
  }

  return std::floor(dValue);
}

double GetDiffusionBValueProstateX(const itk::MetaDataDictionary &clDicomTags) {
  std::string strSequenceName;
  if (!ExposeStringMetaData(clDicomTags, "0018|0024", strSequenceName)) {
    std::cerr << "Error: Could not extract sequence name (0018,0024)." << std::endl;
    return -1.0;
  }

  Trim(strSequenceName);

  if (strSequenceName.empty()) {
    std::cerr << "Error: Empty sequence name (0018,0024)." << std::endl;
    return -1.0;
  }

  std::stringstream valueStream;

  unsigned int uiBValue = 0;

  size_t i = 0, j = 0;
  while (i < strSequenceName.size()) {
    i = strSequenceName.find('b', i); 

    if (i == std::string::npos || ++i >= strSequenceName.size())
      break;

    j = strSequenceName.find_first_not_of("0123456789", i); 

    // Should end with a 't' or a '\0'
    if (j == std::string::npos)
      j = strSequenceName.size();
    else if (strSequenceName[j] != 't')
      break;

    if (j > i) {
      std::string strBValue = strSequenceName.substr(i, j-i);
      valueStream.clear();
      valueStream.str(strBValue);

      uiBValue = 0;

      if (valueStream >> uiBValue) {
        if (uiBValue < 4000)
          return (double)uiBValue;
        else
          std::cerr << "Error: B-value of " << uiBValue << " seems bogus. Continuing to parse." << std::endl;
      }   
    }   

    i = j;
  }

  std::cerr << "Error: Could not parse sequence name '" << strSequenceName << "'." << std::endl;

  return -1.0;
}

double GetDiffusionBValuePhilips(const itk::MetaDataDictionary &clDicomTags) {
  double dBValue = -1.0;

  if (!ExposeStringMetaData(clDicomTags, "2001|1003", dBValue))
    return -1.0;

  return dBValue;
}

std::map<double, std::vector<std::string>> ComputeBValueFileNames(const std::string &strPath, const std::string &strSeriesUID) {
  typedef std::map<double, std::vector<std::string>> MapType;
  typedef itk::GDCMImageIO ImageIOType;
  typedef itk::GDCMSeriesFileNames FileNameGeneratorType;

  ImageIOType::Pointer p_clImageIO = ImageIOType::New();

  p_clImageIO->LoadPrivateTagsOn();
  p_clImageIO->KeepOriginalUIDOn();

  if (!IsFolder(strPath.c_str())) {
    // Get the series UID of the file
    p_clImageIO->SetFileName(strPath);

    try {
      p_clImageIO->ReadImageInformation();
    }
    catch (itk::ExceptionObject &e) {
      std::cerr << "Error: " << e << std::endl;
      return MapType();
    }

    std::string strTmpSeriesUID;
    if (!itk::ExposeMetaData(p_clImageIO->GetMetaDataDictionary(), "0020|000e", strTmpSeriesUID)) {
      std::cerr << "Error: Could not load series UID from DICOM." << std::endl;
      return MapType();
    }

    Trim(strTmpSeriesUID);

    if (GetDiffusionBValue(p_clImageIO->GetMetaDataDictionary()) < 0.0) {
      std::cerr << "Error: Image does not appear to be a diffusion image." << std::endl;
      return MapType();
    }

    return ComputeBValueFileNames(DirName(strPath), strTmpSeriesUID);
  }

  FileNameGeneratorType::Pointer p_clFileNameGenerator = FileNameGeneratorType::New();

  // Use the ACTUAL series UID ... not some custom ITK concatenations of lots of junk.
  p_clFileNameGenerator->SetUseSeriesDetails(false); 
  p_clFileNameGenerator->SetDirectory(strPath);

  if (strSeriesUID.empty()) {
    // Passed a folder but no series UID ... pick the first series UID
    const FileNameGeneratorType::SeriesUIDContainerType &vSeriesUIDs = p_clFileNameGenerator->GetSeriesUIDs();

    if (vSeriesUIDs.empty())
      return MapType();

    return ComputeBValueFileNames(strPath, vSeriesUIDs[0]);
  }

  // These should be ordered by Z coordinate (they're not)
  const FileNameGeneratorType::FileNamesContainerType &vDicomFiles = p_clFileNameGenerator->GetFileNames(strSeriesUID);

  if (vDicomFiles.empty())
    return MapType();

  MapType mFilesByBValue;

  typedef std::pair<std::string, itk::MetaDataDictionary> FileAndDictionaryPair;
  std::unordered_map<double, std::vector<FileAndDictionaryPair>> mFilesAndDictionariesByBValue;

  for (const std::string &strFileName : vDicomFiles) {
    p_clImageIO->SetFileName(strFileName);

    try {
      p_clImageIO->ReadImageInformation();
    }
    catch (itk::ExceptionObject &e) {
      std::cerr << "Error: " << e << std::endl;
      return MapType();
    }

    const double dBValue = GetDiffusionBValue(p_clImageIO->GetMetaDataDictionary());
    if (dBValue < 0.0) {
      std::cerr << "Error: Could not extract diffusion b-value for '" << strFileName << "'." << std::endl;
      return MapType();
    }

    mFilesAndDictionariesByBValue[dBValue].emplace_back(strFileName, p_clImageIO->GetMetaDataDictionary());
  }

  vnl_matrix_fixed<double, 3, 3> clR; // Assume this is the same for all slices

  if (!GetOrientationMatrix(p_clImageIO->GetMetaDataDictionary(), clR)) {
    std::cerr << "Error: Could not get orientation matrix." << std::endl;
    return MapType();
  }

  auto fnCompareByPosition = [&clR](const FileAndDictionaryPair &a, const FileAndDictionaryPair &b) -> bool {
    vnl_vector_fixed<double, 3> clTa, clTb;

    // XXX: Could fail, but shouldn't!!!!
    GetOrigin(a.second, clTa);
    GetOrigin(b.second, clTb);

    return (clR.transpose()*(clTa - clTb))[2] < 0.0;
  };

  // Sort by position
  for (auto &stBValueAndFilesPair : mFilesAndDictionariesByBValue) {
    const double &dBValue = stBValueAndFilesPair.first;
    std::vector<FileAndDictionaryPair> &vFilesAndDictionaries = stBValueAndFilesPair.second;

    std::sort(vFilesAndDictionaries.begin(), vFilesAndDictionaries.end(), fnCompareByPosition);

    std::vector<std::string> &vFileNames = mFilesByBValue[dBValue];

    vFileNames.resize(vFilesAndDictionaries.size());
    std::transform(vFilesAndDictionaries.begin(), vFilesAndDictionaries.end(), vFileNames.begin(),
      [](const FileAndDictionaryPair &stPair) -> const std::string & {
        return stPair.first;
      });
  }

  return mFilesByBValue;
}

template<typename PixelType>
std::map<double, typename itk::Image<PixelType, 3>::Pointer> LoadBValueImages(const std::string &strPath, const std::string &strSeriesUID) {
  typedef itk::GDCMImageIO ImageIOType;
  typedef itk::Image<PixelType, 3> ImageType;
  typedef std::map<double, typename ImageType::Pointer> MapType;
  typedef itk::ImageSeriesReader<ImageType> ReaderType;

  auto mFilesByBalue = ComputeBValueFileNames(strPath, strSeriesUID);

  if (mFilesByBalue.empty())
    return MapType();

  ImageIOType::Pointer p_clImageIO = ImageIOType::New();

  p_clImageIO->LoadPrivateTagsOn();
  p_clImageIO->KeepOriginalUIDOn();

  MapType mImagesByBValue;

  for (const auto &stBValueFilesPair : mFilesByBalue) {
    typename ReaderType::Pointer p_clReader = ReaderType::New();

    p_clReader->SetImageIO(p_clImageIO);
    p_clReader->SetFileNames(stBValueFilesPair.second);

    try {
      p_clReader->Update();
    }
    catch (itk::ExceptionObject &e) {
      std::cerr << "Error: " << e << std::endl;
      return MapType();
    }

    mImagesByBValue[stBValueFilesPair.first] = p_clReader->GetOutput();
  }

  return mImagesByBValue;
}

///////////////////////////////////////////////////////////////////////
// BValueModel funcrions
///////////////////////////////////////////////////////////////////////

template<typename PixelType>
typename itk::Image<PixelType, 3>::Pointer BValueModel::NewImage() const {
  typedef itk::Image<PixelType, 3> OtherImageType;

  if (m_mImagesByBValue.empty() || !m_mImagesByBValue.begin()->second)
    return OtherImageType::Pointer();

  ImageType::Pointer p_clRefImage = m_mImagesByBValue.begin()->second;

  OtherImageType::Pointer p_clImage = OtherImageType::New();
  p_clImage->SetRegions(p_clRefImage->GetBufferedRegion());
  p_clImage->SetSpacing(p_clRefImage->GetSpacing());
  p_clImage->SetOrigin(p_clRefImage->GetOrigin());
  p_clImage->SetDirection(p_clRefImage->GetDirection());

  p_clImage->Allocate();
  p_clImage->FillBuffer(OtherImageType::PixelType());    

  return p_clImage;
}

bool BValueModel::SetLogIntensities(const itk::Index<3> &clIndex) {
  auto itr = GetImages().begin();

  const ImageType::PixelType &b0 = itr->second->GetPixel(clIndex); // OK, not B0 necessarily...

  if (b0 < 0) // Should NEVER happen
    return false;

  ++itr;

  m_vBValueAndLogIntensity.clear();
  m_vBValueAndLogIntensity.reserve(GetImages().size());

  while (itr != m_mImagesByBValue.end()) {

    const ImageType::PixelType &bN = itr->second->GetPixel(clIndex);

    if (bN < 0) // Should NEVER happen
      return false;

    double dLogValue = 0.0;

    if (b0 == 0) {
      dLogValue = bN > 0 ? 1e6 : 0.0;
    }
    else if (bN == 0) {
      dLogValue = -1e6;
    }
    else
      dLogValue = std::log((double)bN / (double)b0);

    m_vBValueAndLogIntensity.emplace_back(itr->first - MinBValue(), dLogValue);

    ++itr;
  }

  return true;
}

bool BValueModel::Run() {
  if (!Good())
    return false;

  //clSolver.set_verbose(true);

  const itk::Size<3> clSize = m_mImagesByBValue.begin()->second->GetBufferedRegion().GetSize();

  m_vBValueAndLogIntensity.reserve(m_mImagesByBValue.size());

  ImageType::Pointer p_clBNImage = NewImage<ImageType::PixelType>();

  for (itk::IndexValueType z = 0; z < clSize[2]; ++z) {
    for (itk::IndexValueType y = 0; y < clSize[1]; ++y) {
      for (itk::IndexValueType x = 0; x < clSize[0]; ++x) {
        const itk::Index<3> clIndex = {{ x, y, z }};

        if (!SetLogIntensities(clIndex))
          continue;

        const double dBN = std::min(4095.0, std::max(0.0, Solve(clIndex)));

        p_clBNImage->SetPixel(clIndex, ImageType::PixelType(dBN + 0.5));
      }
    }
  }

  std::cout << "Info: Saving b-value image to '" << GetOutputPath() << "' ..." << std::endl;
  if (!SaveImg<ImageType::PixelType, 3>(p_clBNImage, GetOutputPath())) {
    std::cerr << "Error: Failed to save b-value image." << std::endl;
    return false;
  }

  return true;
}

///////////////////////////////////////////////////////////////////////
// MonoExponentialModel functions
///////////////////////////////////////////////////////////////////////

bool MonoExponentialModel::Run() {
  if (!Good())
    return false;

  SolverType &clSolver = GetSolver();

  vnl_vector<long> clBoundSelection(ADVarType::GetNumIndependents(), 0); // By default, not constrained
  clBoundSelection[0] = 1; // ADC cannot be negative

  vnl_vector<double> clLowerBound(clBoundSelection.size(), 0.0); // By default, lower bound is 0.0
 
  clSolver.set_bound_selection(clBoundSelection);
  clSolver.set_lower_bound(clLowerBound);

  if (m_bSaveADC)
    m_p_clADCImage = NewImage<ImageType::PixelType>();

  if (!SuperType::Run())
    return false;

  if (m_p_clADCImage.IsNotNull()) {
    std::cout << "Info: Saving ADC image to '" << GetADCOutputPath() << "' ..." << std::endl;

    if (!SaveImg<ImageType::PixelType, 3>(m_p_clADCImage, GetADCOutputPath())) {
      std::cerr << "Error: Failed to save ADC image." << std::endl;
      return false;
    }
  }

  return true;
}

void MonoExponentialModel::compute(const vnl_vector<double> &clX, double *p_dF, vnl_vector<double> *p_clG) {
  ADVarType clD(clX[0], 0), clLogS(clX[1], 1), clLoss(0.0);

  for (auto &stPair : GetLogIntensities()) {      
    clLoss += pow(-stPair.first * clD - stPair.second, 2);
  }

  clLoss += pow(-(GetTargetBValue() - MinBValue()) * clD - clLogS, 2);

  if (p_dF != nullptr)
    *p_dF = clLoss.Value();

  if (p_clG != nullptr) {
    p_clG->set_size(clX.size());
    p_clG->copy_in(clLoss.Gradient().data());
  }
}

double MonoExponentialModel::Solve(const itk::Index<3> &clIndex) {
  vnl_vector<double> clX(ADVarType::GetNumIndependents(), 1.0);

  if (!GetSolver().minimize(clX))
    std::cerr << "Warning: Solver failed at pixel: " << clIndex << std::endl;

  if (m_p_clADCImage.IsNotNull())
    m_p_clADCImage->SetPixel(clIndex, ImageType::PixelType(1e6*clX[0] + 0.5));

  const ImageType::PixelType &b0 = GetImages().begin()->second->GetPixel(clIndex);

  return (double)b0 * std::exp(clX[1]);
}

///////////////////////////////////////////////////////////////////////
// IVIMModel functions
///////////////////////////////////////////////////////////////////////

bool IVIMModel::Run() {
  if (!Good())
    return false;

  if (MinBValue() != 0.0) {
    std::cerr << "Error: IVIM model needs B0 image." << std::endl;
    return false;
  }

  SolverType &clSolver = GetSolver();

  vnl_vector<long> clBoundSelection(ADVarType::GetNumIndependents(), 0); // By default, not constrained
  clBoundSelection[0] = 1; // ADC cannot be negative
  clBoundSelection[2] = 3; // Log perfusion fraction cannot be positive

  vnl_vector<double> clLowerBound(clBoundSelection.size(), 0.0); // By default, lower bound is 0.0 (if applicable)
  vnl_vector<double> clUpperBound(clBoundSelection.size(), 0.0); // By default, upper bound is 0.0 (if applicable)
 
  clSolver.set_bound_selection(clBoundSelection);
  clSolver.set_lower_bound(clLowerBound);
  clSolver.set_upper_bound(clUpperBound);

  if (m_bSaveADC)
    m_p_clADCImage = NewImage<ImageType::PixelType>();

  if (m_bSavePerfusion)
    m_p_clPerfusionImage = NewImage<FloatImageType::PixelType>();

  if (!SuperType::Run())
    return false;

  if (m_p_clADCImage.IsNotNull()) {
    std::cout << "Info: Saving ADC image to '" << GetADCOutputPath() << "' ..." << std::endl;

    if (!SaveImg<ImageType::PixelType, 3>(m_p_clADCImage, GetADCOutputPath())) {
      std::cerr << "Error: Failed to save ADC value image." << std::endl;
      return false;
    }
  }

  if (m_p_clPerfusionImage.IsNotNull()) {
    std::cout << "Info: Saving perfusion fraction image to '" << GetPerfusionOutputPath() << "' ..." << std::endl;

    if (!SaveImg<FloatImageType::PixelType, 3>(m_p_clPerfusionImage, GetPerfusionOutputPath())) {
      std::cerr << "Error: Failed to save perfusion fraction image." << std::endl;
      return false;
    }
  }

  return true;
}

void IVIMModel::compute(const vnl_vector<double> &clX, double *p_dF, vnl_vector<double> *p_clG) {
  ADVarType clD(clX[0], 0), clLogS(clX[1], 1), clLogF(clX[2], 2), clLoss(0.0);

  for (auto &stPair : GetLogIntensities()) {      
    clLoss += pow(clLogF - stPair.first * clD - stPair.second, 2);
  }

  clLoss += pow(clLogF - GetTargetBValue() * clD - clLogS, 2);

  if (p_dF != nullptr)
    *p_dF = clLoss.Value();

  if (p_clG != nullptr) {
    p_clG->set_size(clX.size());
    p_clG->copy_in(clLoss.Gradient().data());
  }
}

double IVIMModel::Solve(const itk::Index<3> &clIndex) {
  vnl_vector<double> clX(ADVarType::GetNumIndependents(), 1.0);

  clX[2] = -1.0; // Log perfusion fraction

  if (!GetSolver().minimize(clX))
    std::cerr << "Warning: Solver failed at pixel: " << clIndex << std::endl;

  if (m_p_clADCImage.IsNotNull())
    m_p_clADCImage->SetPixel(clIndex, ImageType::PixelType(1e6*clX[0] + 0.5));

  if (m_p_clPerfusionImage.IsNotNull())
    m_p_clPerfusionImage->SetPixel(clIndex, FloatImageType::PixelType(1.0 - std::exp(clX[2])));

  const ImageType::PixelType &b0 = GetImages().begin()->second->GetPixel(clIndex);

  return (double)b0 * std::exp(clX[1]);
}
