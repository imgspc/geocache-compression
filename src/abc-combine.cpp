//-*****************************************************************************
//
// Copyright (c) 2013,
//  Sony Pictures Imageworks, Inc. and
//  Industrial Light & Magic, a division of Lucasfilm Entertainment Company Ltd.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Sony Pictures Imageworks, nor
// Industrial Light & Magic nor the names of their contributors may be used
// to endorse or promote products derived from this software without specific
// prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//-*****************************************************************************

// Adapted from AbcConvert.cpp version 1.8.8
//
// This does basically the same job of copying all the data from source to
// destination, but it replaces a channel with raw data read from a .bin
// source that we assume is formatted properly. And we always write Ogawa.

#include <Alembic/Abc/All.h>
#include <Alembic/AbcCoreFactory/All.h>
#include <Alembic/AbcCoreOgawa/All.h>

typedef Alembic::AbcCoreFactory::IFactory IFactoryNS;

enum ArgMode
{
    kOptions,
    kInFile,
    kOutFile,
    kPropertyName,
    kPropertyFile,
};

class ConversionOptions
{
public:

    ConversionOptions()
    {
        toType = IFactoryNS::kUnknown;
        force = false;
    }

    // path to the input file, or blank for stdin.
    std::string inFile;

    // path to the output file, or blank for stdout.
    std::string outFile;

    // map of component path -> binary filename path
    std::map<std::string, std::string> inProperties;

    // Return the file path to read as a binary file for the channel,
    // or an empty string if we should just copy the source file.
    std::string getPropertyFile(const std::string& path) const
    {
        auto it = inProperties.find(path);
        if (it == inProperties.end()) {
            return "";
        } else {
            return *it;
        }
    }
};

void copyProps(Alembic::Abc::ICompoundProperty & iRead,
    Alembic::Abc::OCompoundProperty & iWrite, 
    const ConversionOptions& options,
    const std::string& path)
{
    std::size_t numChildren = iRead.getNumProperties();
    for (std::size_t i = 0; i < numChildren; ++i)
    {
        Alembic::AbcCoreAbstract::PropertyHeader header =
            iRead.getPropertyHeader(i);
        std::string childPath = path + '/' + header.getName();

        auto dataType = header.getDataType();
        size_t bytesPerDatum = dataType.getNumBytes();

        if (header.isArray())
        {
            Alembic::Abc::IArrayProperty inProp(iRead, header.getName());
            Alembic::Abc::OArrayProperty outProp(iWrite, header.getName(),
                dataType, header.getMetaData(),
                header.getTimeSampling());

            std::size_t numSamples = inProp.getNumSamples();

            std::string binaryFilePath = options.getPropertyFile(childPath);
            if (binaryFilePath != "") {
                if (options.verbose) {
                    std::cerr << "Replacing " << childPath << " with " << binaryFilePath << std::endl;
                }
                std::ifstream binaryFile(binaryFilePath, std::ios::in | std::ios::binary);

                auto dimensions = header.getDimensions();
                char *buffer = new char[bytesPerDatum];
                for (std::size_t j = 0; j < numSamples; ++j)
                {
                    Alembic::AbcCoreAbstract::ArraySample samp(buffer, dataType, dimensions);
                    binaryFile.read(buffer, bytesPerDatum);
                    outProp.set(samp)
                }
                delete [] buffer;
            } else {
                if (options.verbose) {
                    std::cerr << "Copying " << childPath << std::endl;
                }
                for (std::size_t j = 0; j < numSamples; ++j)
                {
                    Alembic::AbcCoreAbstract::ArraySamplePtr samp;
                    Alembic::Abc::ISampleSelector sel(
                            (Alembic::Abc::index_t) j);
                    inProp.get(samp, sel);
                    outProp.set(*samp);
                }
            }
        }
        else if (header.isScalar())
        {
            Alembic::Abc::IScalarProperty inProp(iRead, header.getName());
            Alembic::Abc::OScalarProperty outProp(iWrite, header.getName(),
                header.getDataType(), header.getMetaData(),
                header.getTimeSampling());

            std::size_t numSamples = inProp.getNumSamples();
            std::vector<std::string> sampStrVec;
            std::vector<std::wstring> sampWStrVec;
            if (header.getDataType().getPod() ==
                Alembic::AbcCoreAbstract::kStringPOD)
            {
                sampStrVec.resize(header.getDataType().getExtent());
            }
            else if (header.getDataType().getPod() ==
                     Alembic::AbcCoreAbstract::kWstringPOD)
            {
                sampWStrVec.resize(header.getDataType().getExtent());
            }

            std::string binaryFilePath = options.getPropertyFile(childPath);
            if (binaryFilePath != "")
            {
                if (options.verbose) {
                    std::cerr << "Replacing " << childPath << " with " << binaryFilePath << std::endl;
                }
                std::ifstream binaryFile(binaryFilePath, std::ios::in | std::ios::binary);

                if (header.getDataType().getPod() == Alembic::AbcCoreAbstract::kStringPOD
                        || header.getDataType().getPod() == Alembic::AbcCoreAbstract::kWstringPOD) {
                    std::cerr << "ERROR: property " << childPath << " is a string property and can't be replaced\n";
                    exit(1);
                }

                Alembic::AbcCoreAbstract::ScalarSample samp(dataType);
                for (std::size_t j = 0; j < numSamples; ++j)
                {
                    // read straight into the sample buffer, skip the copy
                    binaryFile.read(const_cast<void*>(samp.getData()), bytesPerDatum);
                    outProp.set(samp)
                }
            }
            else 
            {
                if (options.verbose) {
                    std::cerr << "Copying " << childPath << std::endl;
                }
                for (std::size_t j = 0; j < numSamples; ++j)
                {
                    Alembic::Abc::ISampleSelector sel(
                            (Alembic::Abc::index_t) j);

                    if (header.getDataType().getPod() ==
                            Alembic::AbcCoreAbstract::kStringPOD)
                    {
                        inProp.get(&sampStrVec.front(), sel);
                        outProp.set(&sampStrVec.front());
                    }
                    else if (header.getDataType().getPod() ==
                            Alembic::AbcCoreAbstract::kWstringPOD)
                    {
                        inProp.get(&sampWStrVec.front(), sel);
                        outProp.set(&sampWStrVec.front());
                    }
                    else
                    {
                        char samp[4096]; // max extent * max POD size is 255 * 8
                        inProp.get(samp, sel);
                        outProp.set(samp);
                    }
                }
            }
        }
        else if (header.isCompound())
        {
            Alembic::Abc::OCompoundProperty outProp(iWrite,
                header.getName(), header.getMetaData());
            Alembic::Abc::ICompoundProperty inProp(iRead, header.getName());
            copyProps(inProp, outProp, options, childPath);
        }
    }
}

void copyObject(Alembic::Abc::IObject & iIn,
    Alembic::Abc::OObject & iOut,
    const ConversionOptions& options,
    const std::string& path)
{
    std::size_t numChildren = iIn.getNumChildren();

    Alembic::Abc::ICompoundProperty inProps = iIn.getProperties();
    Alembic::Abc::OCompoundProperty outProps = iOut.getProperties();
    copyProps(inProps, outProps);

    // We aren't using a leading '/' at the root of the object tree.
    std::string pathPrefix = (path == "") ? "" : (path + "/");

    for (std::size_t i = 0; i < numChildren; ++i)
    {
        Alembic::Abc::IObject childIn(iIn.getChild(i));
        Alembic::Abc::OObject childOut(iOut, childIn.getName(),
                                       childIn.getMetaData());
        copyObject(childIn, childOut, options, pathPrefix + childIn.getName());
    }
}

void displayHelp()
{
    std::cerr << "abc-combine [inFile.abc] [-o outFile.abc] [-p path/to//component file.bin] [-v] [-h]\n"
        << std::endl
        << "Combine an alembic file with binary files containing components. This is the opposite of abc-separate.\n"
        << std::endl
        << "OPTIONS:"
        << "  inFile.abc        the Alembic file to read; defaults to stdin\n"
        << "  --output or -o outFile.abc    the Alembic file to write; defaults to stdout. Must not be the same as inFile.abc\n"
        << "  --property or -p path/to//component file.bin    the binary file and component to replace. The double-/ separates the object hierarchy from the component hierarchy\n"
        << "  --verbose or -v   print verbose output to stderr\n";
        << "  --help or -h      print this help message to stderr\n";
}

bool parseArgs( int iArgc, char *iArgv[], ConversionOptions &oOptions, bool &oDoConversion )
{
    oDoConversion = true;
    ArgMode argMode = kOptions;

    std::string propertyName;

    for( int i = 1; i < iArgc; i++ )
    {
        bool argHandled = true;
        std::string arg = iArgv[i];

        switch( argMode )
        {
            case kOptions:
            {
                if( (arg == "-h") || (arg == "--help") )
                {
                    displayHelp();
                    oDoConversion = false;
                    return true;
                }
                else if ( (arg == "-p") || (arg == "--property") )
                {
                    argMode = kPropertyName;
                }
                else if ( (arg == "-o") || (arg == "--output") )
                {
                    argMode = kOutFile;
                }
                else if ( (arg == "-v") || (arg == "--verbose") )
                {
                    oOptions.verbose = true;
                }
                else
                {
                    argMode = kInFile;
                    i--;
                }
            }
            break;

            case kPropertyName:
            {
                propertyName = arg;
                argMode = kPropertyFile;
            }
            break;

            case kPropertyFile:
            {
                oOptions.inProperties[propertyName] = arg;
                propertyName = "";
                argMode = kOptions;
            }
            break;

            case kInFile:
            {
                oOptions.inFile = arg;
                argMode = kOptions;
            }
            break;

            case kOutFile:
            {
                oOptions.outFile = arg;
                argMode = kOptions;
            }
            break;
        }

        if( !argHandled )
        {
            displayHelp();
            oDoConversion = false;
            return false;
        }
    }

    return true;
}

int main(int argc, char *argv[])
{
    ConversionOptions options;
    bool doConversion = false;

    if (!parseArgs(argc, argv, options, doConversion))
        return 1;

    if (!doConversion)
        return 0;

    if (options.inFile == options.outFile && options.inFile != "")
    {
        std::cerr << "ERROR: input and output files can't be the same (" << options.inFile << std::endl;
        return 1;
    }

    Alembic::AbcCoreFactory::IFactory factory;
    Alembic::Abc::IArchive archive;
    if (options.inFile != "")
    {
        archive = factory.getArchive(options.inFile);
    }
    else
    {
        // The API demands we have a vector of streams that all point to the
        // same Ogawa-format data. Ask no questions.
        std::vector<std::istream*> instreams;
        instreams.push_back(&std::cin);
        CoreType coreType;
        archive = factory.getArchive(instreams, coreType);
    }

    Alembic::Abc::IObject inTop = archive.getTop();

    Alembic::Abc::ArchiveWriterPtr archiveWriter;
    if (options.outFile != "")
    {
        archiveWriter = Alembic::AbcCoreOgawa::WriteArchive(
                options.outFile,
                inTop.getMetaData());
    }
    else
    {
        archiveWriter = Alembic::AbcCoreOgawa::WriteArchive(
                &std::cout,
                inTop.getMetaData());
    }
    Alembic::Abc::OArchive outArchive(archiveWriter);

    // start at 1, we don't need to worry about intrinsic default case
    for (Alembic::Util::uint32_t i = 1; i < archive.getNumTimeSamplings(); ++i)
    {
        outArchive.addTimeSampling(*archive.getTimeSampling(i));
    }

    Alembic::Abc::OObject outTop = outArchive.getTop();
    copyObject(inTop, outTop, options, "");
    return 0;
}
