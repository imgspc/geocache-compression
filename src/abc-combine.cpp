// Imaginary Spaces (C) 2025

// TODO: this is just a copy of some earlier code, taking up the space of a file.


#include <Alembic/AbcCoreFactory/IFactory.h>
#include <Alembic/Abc/IObject.h>

#include <iostream>
#include <fstream>
#include <set>
#include <string>

using Alembic::Abc::kStringPOD;
using Alembic::Abc::kWstringPOD;
using Alembic::Abc::DataType;
using Alembic::Abc::Dimensions;
using Alembic::Abc::IArchive;
using Alembic::Abc::IObject;
using Alembic::Abc::kArrayProperty;
using Alembic::Abc::kCompoundProperty;
using Alembic::Abc::kScalarProperty;
using Alembic::Abc::ObjectReaderPtr;
using Alembic::AbcCoreAbstract::ArrayPropertyReaderPtr;
using Alembic::AbcCoreAbstract::BasePropertyReaderPtr;
using Alembic::AbcCoreAbstract::CompoundPropertyReaderPtr;
using Alembic::AbcCoreAbstract::ScalarPropertyReaderPtr;
using Alembic::AbcCoreFactory::IFactory;

static BasePropertyReaderPtr find_property(CompoundPropertyReaderPtr prop, const std::string& name);
static BasePropertyReaderPtr find_property(BasePropertyReaderPtr prop, const std::string& name);

static std::pair<std::string, std::string> split(const std::string& name)
{
    size_t index = name.find_first_of('/');
    if (index == std::string::npos) {
        return std::make_pair(name, "");
    } else {
        return std::make_pair(
                name.substr(0, index),
                name.substr(index+1)
                );
    }
}


static BasePropertyReaderPtr find_property(CompoundPropertyReaderPtr prop, const std::string& name)
{
    auto parts = split(name);

    BasePropertyReaderPtr child = prop->getProperty(parts.first);
    if (!child) { 
        std::cout << name << " not found in property " << prop->getName() << " of object " << prop->getObject()->getName() << std::endl;
        return nullptr;
    }

    // Find the remaining path and search for it.
    return find_property(child, parts.second);
}


static BasePropertyReaderPtr find_property(BasePropertyReaderPtr prop, const std::string& name)
{
    switch (prop->getPropertyType())
    {
        case kCompoundProperty:
            return find_property(prop->asCompoundPtr(), name);
        case kScalarProperty:
        case kArrayProperty:
            if (name != "") {
                // We don't have child properties so if there's a path to keep looking
                // for, it doesn't exist.
                return nullptr;
            }
            return prop;
    }
}


static BasePropertyReaderPtr find_property(ObjectReaderPtr obj, const std::string& name)
{
    auto parts = split(name);
    BasePropertyReaderPtr found_property;

    // The full path will look like obj1/obj2//prop1/prop2
    // We switch into the properties at the double-dot, which shows up as a
    // blank parts.first.
    if (parts.first == "") {
        BasePropertyReaderPtr prop = obj->getProperties();
        if (!prop) {
            std::cout << name << " not found in object " << obj->getName() << std::endl;
            return nullptr;
        }
        return find_property(prop, parts.second);
    } else {
        ObjectReaderPtr child = obj->getChild(parts.first);
        if (!child) {
            std::cout << name << " not found in object " << obj->getName() << std::endl;
            return nullptr;
        }
        return find_property(child, parts.second);
    }
}


static bool write_property(ScalarPropertyReaderPtr prop, std::ofstream& out)
{
    DataType dataType = prop->getDataType();
    if (dataType.getPod() == kStringPOD || dataType.getPod() == kWstringPOD) {
        std::cerr << "unable to handle string data\n";
        return false;
    }

    size_t bytesPerDatum = dataType.getNumBytes();

    std::cout << "writing " << prop->getNumSamples() << " samples, "
        << (size_t)dataType.getExtent() << " x " << PODName(dataType.getPod())
        << " (" << bytesPerDatum << " bytes) per sample\n";

    char *buffer = new char[bytesPerDatum];
    for(size_t i = 0, n = prop->getNumSamples(); i < n; ++i) {
        prop->getSample(i, buffer);
        out.write(buffer, bytesPerDatum);
    }
    delete [] buffer;

    return true;
}


static bool write_property(ArrayPropertyReaderPtr prop, std::ofstream& out)
{
    DataType dataType = prop->getDataType();
    auto pod = dataType.getPod();
    if (pod == kStringPOD || pod == kWstringPOD) {
        std::cerr << "unable to handle string data\n";
        return false;
    }

    size_t bytesPerDatum = dataType.getNumBytes();
    size_t n = prop->getNumSamples();
    if (prop->isConstant()) {
        n = 1;
    }
    std::cout << "writing " << n << " array samples, " 
        << (size_t)dataType.getExtent() << " x " << PODName(pod)
        << " (" << bytesPerDatum << " bytes) per item\n";

    // Verify consistent rank/dimension. Otherwise we're cooked.
    std::set<uint64_t> ranks;
    std::set<uint64_t> dimensions;
    for (size_t i = 0; i < n; ++i) {
        Dimensions dim;
        prop->getDimensions(i, dim);
        ranks.insert(dim.rank());
        for (size_t j = 0; j < dim.rank(); ++j) {
            dimensions.insert(dim[j]);
        }
    }
    if (ranks.size() == 0 || dimensions.size() == 0) {
        std::cerr << "No data to write\n";
        return false;
    }
    if (ranks.size() != 1) {
        std::cerr << "Rank differs\n";
        return false;
    }
    if (dimensions.size() != 1) {
        std::cerr << "Dimension differs\n";
        return false;
    }

    size_t dimension = *dimensions.begin();
    size_t rank = *ranks.begin();

    size_t bytesPerSample = bytesPerDatum * dimension * rank;

    std::cout << "samples are rank " << rank << " x " 
        << dimension << " (" << bytesPerSample << " bytes)\n";

    char *buffer = new char[bytesPerSample];
    for(size_t i = 0; i < n; ++i) {
        prop->getAs(i, buffer, pod);
        out.write(buffer, bytesPerSample);
    }
    delete [] buffer;

    return true;
}


static bool write_property(BasePropertyReaderPtr prop, const std::string& filename)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }

    switch (prop->getPropertyType())
    {
        case kScalarProperty:
            return write_property(prop->asScalarPtr(), out);
        case kArrayProperty:
            return write_property(prop->asArrayPtr(), out);
        case kCompoundProperty:
        default:
            return false;
    }
}


int main(int argc, const char * const *argv)
{
    bool verbose = false;
    std::vector<std::string> values;

    for(int i = 1; i < argc; ++i)
    {
        std::string str = argv[i];
        if (str == "-v" || str == "--verbose") {
            verbose = true;
        }
        values.push_back(str);
    }

    if (values.size() != 3) {
        std::cerr << "arguments: alembic-to-flat [-v|--verbose] input.abc property.path output.bin\n";
        std::cerr << "\n";
        std::cerr << "Reads the property (specified by path) from input.abc, dumps the samples to the output.bin file\n";
        return 1;
    }
    std::string input_filename = values[0];
    std::string property_path = values[1];
    std::string output_filename = values[2];

    IFactory factory;
    IArchive archive = factory.getArchive(input_filename);
    if (!archive.valid()) {
        std::cerr << "failed to open " << input_filename << std::endl;
        return 1;
    }

    BasePropertyReaderPtr property = find_property(archive.getTop().getPtr(), property_path);
    if (!property) {
        std::cerr << "property " << property_path << " not found in " << input_filename << std::endl;
        return 1;
    }

    bool did_write = write_property(property, output_filename);
    if (!did_write) {
        std::cerr << "failed to write output to " << output_filename;
        return 1;
    }

    return 0;
}
