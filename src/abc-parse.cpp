// Imaginary Spaces (C) 2025

// List out the hierarchy of an .abc file.
// This was really just my attempt at figuring out how to read an alembic file at all.

#include <Alembic/AbcCoreFactory/IFactory.h>
#include <Alembic/Abc/IObject.h>

#include <iostream>
#include <set>

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

// drop-in replacement for std::cout that indents the first line
class Printer {
    static size_t depth;

    public:
    Printer() {
        depth++;
    }
    ~Printer() {
        depth--;
    }
    template <class T> std::ostream& operator << (const T& msg) {
        for(size_t i = 0; i < depth; ++i) {
            std::cout << "  ";
        }
        std::cout << msg;
        return std::cout;
    }
};
size_t Printer::depth = ~(size_t)0;

std::string property_type(BasePropertyReaderPtr reader)
{
    switch(reader->getPropertyType()) {
        case kCompoundProperty: return "compound";
        case kScalarProperty: return "scalar";
        case kArrayProperty: return "array";
        default: return "(unknown)";
    }
}

void read_property(BasePropertyReaderPtr reader)
{
    Printer print;

    switch (reader->getPropertyType())
    {
        case kCompoundProperty:
            {
                CompoundPropertyReaderPtr compound = reader->asCompoundPtr();
                print << "compound property " << reader->getName() << " has "
                      << compound->getNumProperties() << " sub-properties\n";
                for (size_t i = 0; i < compound->getNumProperties(); ++i) {
                    read_property(compound->getProperty(i));
                }
                break;
            }
        case kScalarProperty:
            {
                ScalarPropertyReaderPtr scalar = reader->asScalarPtr();
                print << "scalar property " << reader->getName() 
                      << " has " << scalar->getNumSamples() 
                      << " " << PODName(scalar->getDataType().getPod())
                      << " samples\n";
                if (scalar->isConstant()) {
                    print << " ** value is constant\n";
                }
                break;
            }
        case kArrayProperty:
            {
                ArrayPropertyReaderPtr array = reader->asArrayPtr();
                DataType dataType = reader->getDataType();
                size_t bytesPerDatum = dataType.getNumBytes();
                print << "array property " << reader->getName() 
                      << " has " << array->getNumSamples() 
                      << " " << bytesPerDatum << "-byte " 
                      << PODName(array->getDataType().getPod())
                      << " samples\n";
                if (array->isConstant()) {
                    print << " ** value is constant\n";
                }
                Printer arrayPrint;
                std::set<uint64_t> ranks;
                std::set<uint64_t> dimensions;
                for (size_t i = 0; i < array->getNumSamples(); ++i) {
                    Dimensions dim;
                    array->getDimensions(i, dim);
                    ranks.insert(dim.rank());
                    for (size_t j = 0; j < dim.rank(); ++j) {
                        dimensions.insert(dim[j]);
                    }
                }
                if (ranks.size() == 0) {
                    arrayPrint << "zero rank\n";
                }
                else if (ranks.size() == 1) {
                    arrayPrint << "rank " << *ranks.begin() << std::endl;
                }
                else {
                    arrayPrint << "varying rank\n";
                }
                if (dimensions.size() == 0) {
                    arrayPrint << "zero dimension\n";
                }
                else if (dimensions.size() == 1) {
                    arrayPrint << "dimension " << *dimensions.begin() << std::endl;
                }
                else {
                    arrayPrint << "varying dimensions\n";
                }
                break;
            }
        }
}


// Print out the name of the object and recursively its children and components.
void read_object(ObjectReaderPtr reader)
{
    Printer print;

    print << "object " << reader->getName() << std::endl;

    // Properties are all under a nameless compound property
    CompoundPropertyReaderPtr rootProperty = reader->getProperties();
    for(size_t i = 0; i < rootProperty->getNumProperties(); ++i) {
        read_property(rootProperty->getProperty(i));
    }

    // Then we have child objects
    print << "object " << reader->getName() << " has " << reader->getNumChildren() << " children\n";
    for (size_t i = 0; i < reader->getNumChildren(); ++i) {
        read_object(reader->getChild(i));
    }
}

// Read an alembic file, print its hierarchy in a human-readable format
bool read_alembic(const std::string& filename)
{
   Printer print;

   // Step 1: convert the filename into an archive.
   print << "opening " << filename << std::endl;
   IFactory factory;
   IArchive archive = factory.getArchive(filename);
   if (!archive.valid()) {
        print << "failed to open " << filename << std::endl;
        return false;
   }

   // Step 2: print out all the objects in hierarchy.
   read_object(archive.getTop().getPtr());
   return true;
}

int main(int argc, const char * const *argv)
{
    if (argc != 2) {
        std::cerr << "we require exactly one argument: the path to an .abc file";
        return 1;
    }
    std::string filename = argv[1];
    bool ok = read_alembic(filename);
    return ok ? 0 : 1;
}
