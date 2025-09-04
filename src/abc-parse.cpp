// Imaginary Spaces (C) 2025

// List out the hierarchy of an .abc file.
// Output a json to stdout that includes all the components we can deal with.
// Schema:
// { "abc" : "/path/to/abc",
//   "components" : [
//      { "path" : "object//component",
//        "bin" : "object-component.bin",
//        "type" : "float32",
//        "extent" : 3,
//        "size" : 94081,
//        "samples" : 31 }
//   ...
// ] }
//
// Scalar components don't have a size.
//
// Array components with variable size aren't output.
//
// Constant components aren't output, only those that actually have samples.
//

#include <Alembic/Abc/IObject.h>
#include <Alembic/AbcCoreFactory/IFactory.h>

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

// drop-in replacement for std::cout that indents
// on each print.
class Printer {
  static size_t depth;

  std::string prefix;

public:
  Printer() : prefix(2 * depth, ' ') {
    // Increment depth *after* generating the prefix so that the top-level
    // Printer has no prefix (depth 0).
    depth++;
  }
  ~Printer() { depth--; }
  template <class T> std::ostream &operator<<(const T &msg) {
    return std::cout << prefix << msg;
  }
};
size_t Printer::depth = 0;

// Write a property into the json on stdout.
// Specify a size of zero (default) for scalars.
void write_property(bool &is_first_component, const std::string &path,
                    Alembic::Abc::PlainOldDataType podType, size_t numSamples,
                    size_t extent, size_t size = 0) {
  // Assumption: dashes in object/component names won't match dashes,
  // so the bin filenames don't collide. We don't check this.
  // We also remove any leading /.- character
  std::string bin = path;
  size_t first_non_path = std::string::npos;
  for (size_t i = 0, n = bin.size(); i < n; ++i) {
    switch (bin[i]) {
    case '/':
    case ':':
    case ';':
      bin[i] = '-';
      break;
    case '.':
    case '-':
      break;
    default:
      if (first_non_path == std::string::npos) {
        first_non_path = i;
      }
    }
  }
  bin = bin.substr(first_non_path) + ".bin";

  Printer print;

  if (is_first_component) {
    is_first_component = false;
  } else {
    print << ",\n";
  }
  print << "{\n";
  print << "  \"path\": \"" << path << "\",\n";
  print << "  \"type\": \"" << PODName(podType) << "\",\n";
  print << "  \"samples\": " << numSamples << ",\n";
  print << "  \"extent\": " << extent << ",\n";
  if (size) {
    print << "  \"size\" : " << size << ",\n";
  }
  print << "  \"bin\" : \"" << bin << "\"\n";
  print << "}\n";
}

void read_property(bool &is_first_component, const std::string &path,
                   BasePropertyReaderPtr reader) {
  switch (reader->getPropertyType()) {
  case kCompoundProperty: {
    CompoundPropertyReaderPtr compound = reader->asCompoundPtr();
    for (size_t i = 0; i < compound->getNumProperties(); ++i) {
      auto child = compound->getProperty(i);
      std::string childPath = path + '/' + child->getName();
      read_property(is_first_component, childPath, compound->getProperty(i));
    }
    break;
  }
  case kScalarProperty: {
    ScalarPropertyReaderPtr scalar = reader->asScalarPtr();
    if (!scalar->isConstant()) {
      auto dataType = scalar->getDataType();
      write_property(is_first_component, path, dataType.getPod(),
                     scalar->getNumSamples(), dataType.getExtent());
    }
    break;
  }
  case kArrayProperty: {
    ArrayPropertyReaderPtr array = reader->asArrayPtr();
    if (array->isConstant()) {
      break;
    }
    auto dataType = reader->getDataType();
    std::set<uint64_t> sizes;
    for (size_t i = 0; i < array->getNumSamples(); ++i) {
      Dimensions dim;
      array->getDimensions(i, dim);
      sizes.insert(dim.numPoints());
    }
    if (sizes.size() != 1) {
      break;
    }
    write_property(is_first_component, path, dataType.getPod(),
                   array->getNumSamples(), dataType.getExtent(),
                   *sizes.begin());
    break;
  }
  }
}

// Print out the name of the object and recursively its children and components.
void read_object(bool &is_first_component, const std::string &path,
                 ObjectReaderPtr reader) {
  // Properties are under one nameless root property.
  read_property(is_first_component, path + '/', reader->getProperties());

  // Then we have child objects
  for (size_t i = 0; i < reader->getNumChildren(); ++i) {
    auto child = reader->getChild(i);
    std::string childPath = path + '/' + child->getName();
    read_object(is_first_component, childPath, child);
  }
}

// Read an alembic file, print its hierarchy in a human-readable format
bool read_alembic(const std::string &filename) {
  // First, read. If we fail, just quit now with an error result.
  IFactory factory;
  IArchive archive = factory.getArchive(filename);
  if (!archive.valid()) {
    std::cout << "{ \"error\" : \"" << filename << "\" }";
    return false;
  }

  Printer rootPrint;
  rootPrint << "{\n";
  {
    Printer print;
    print << "\"abc\" : \"" << filename << "\",\n";
    print << "\"components\" : [\n";
    // We need to know if it's the first component, if not, we need to add
    // a leading comma, because json is strict about needing a comma to
    // separate, not terminate.
    bool is_first_component = true;
    read_object(is_first_component, "", archive.getTop().getPtr());
    print << "]\n";
  }
  rootPrint << "}\n";
  return true;
}

int main(int argc, const char *const *argv) {
  if (argc != 2) {
    std::cerr << "we require exactly one argument: the path to an .abc file";
    return 1;
  }
  std::string filename = argv[1];
  bool ok = read_alembic(filename);
  return ok ? 0 : 1;
}
