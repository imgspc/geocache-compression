// Imaginary Spaces (C) 2025

// Compare two .abc files.
// Return 0 if their data matches, 1 if they differ.
// Write a diff-like output to stdout if they differ (unless -q).
// + path/to//property
// - path/to//property
// * path/to//property
// If it's in the right file but not the left, +
// If it's in the right file but not the left, -
// If it's in both files but the size/extent/values differ, *

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
using Alembic::Abc::kStringPOD;
;
using Alembic::Abc::kWstringPOD;
;
using Alembic::Abc::ObjectReaderPtr;
using Alembic::AbcCoreAbstract::ArrayPropertyReaderPtr;
using Alembic::AbcCoreAbstract::BasePropertyReaderPtr;
using Alembic::AbcCoreAbstract::CompoundPropertyReaderPtr;
using Alembic::AbcCoreAbstract::ScalarPropertyReaderPtr;
using Alembic::AbcCoreFactory::IFactory;

// Size of a POD DataType is the extent (uint8_t) times up to 1, 2, 4, or 8
// bytes. This is the max possible size of any such. wstring and string could be
// larger but we don't care.
static const size_t maxPodDataSize =
    size_t(std::numeric_limits<uint8_t>::max()) * 8;

// ostream-like class, helps process the verbose and quiet flags.
class Output {
private:
  bool quiet;

public:
  Output(bool quiet) : quiet(quiet) {}

  void make_quiet() { quiet = true; }
  void make_verbose() { quiet = false; }

  template <class T> Output &operator<<(const T &msg) {
    if (!quiet) {
      std::cout << msg;
    }
    return *this;
  }
};

// normal output is not quiet by default, but verbose output is.
static Output output(false);
static Output verbose(true);

// Return the number of samples we need to check.
// If the property is constant, then return 1.
template <class PropertyReaderPtr>
static size_t getNumSamples(PropertyReaderPtr reader) {
  if (reader->isConstant()) {
    return 1;
  } else {
    return reader->getNumSamples();
  }
}

// Compare the data of two scalar properties, which are guaranteed to be POD
// types (not strings) and have the same number of samples.
static bool compare_leaf_property_data(ScalarPropertyReaderPtr readerA,
                                       ScalarPropertyReaderPtr readerB) {
  char bufferA[maxPodDataSize];
  char bufferB[maxPodDataSize];
  size_t dataSize = readerA->getDataType().getNumBytes();
  for (size_t i = 0, n = getNumSamples(readerA); i < n; ++i) {
    readerA->getSample(i, bufferA);
    readerB->getSample(i, bufferB);
    if (memcmp(bufferA, bufferB, dataSize)) {
      return false;
    }
  }
  return true;
}

// Compare the data of two array properties, which are guaranteed to be POD
// types (not strings) and have the same number of samples.
static bool compare_leaf_property_data(ArrayPropertyReaderPtr readerA,
                                       ArrayPropertyReaderPtr readerB) {
  size_t dataSize = readerA->getDataType().getNumBytes();
  for (size_t i = 0, n = getNumSamples(readerA); i < n; ++i) {
    Dimensions dimA;
    Dimensions dimB;
    readerA->getDimensions(i, dimA);
    readerB->getDimensions(i, dimB);
    if (dimA != dimB) {
      return false;
    }
    size_t numPoints = dimA.numPoints();
    size_t numBytes = numPoints * dataSize;
    verbose << '[' << i << "] comparing " << numBytes << " bytes\n";

    // Performance could be improved here by reusing the allocated buffers.
    Alembic::AbcCoreAbstract::ArraySamplePtr sampleA;
    Alembic::AbcCoreAbstract::ArraySamplePtr sampleB;
    readerA->getSample(i, sampleA);
    readerB->getSample(i, sampleB);
    if (memcmp(sampleA->getData(), sampleB->getData(), numBytes)) {
      return false;
    }
  }
  return true;
}

template <class PropertyReaderPtr>
static bool compare_leaf_property(PropertyReaderPtr readerA,
                                  PropertyReaderPtr readerB) {
  size_t nA = getNumSamples(readerA);
  size_t nB = getNumSamples(readerB);
  if (nA != nB) {
    return false;
  }

  if (readerA->getDataType() != readerB->getDataType()) {
    return false;
  }
  switch (readerA->getDataType().getPod()) {
  case kStringPOD:
  case kWstringPOD:
    // TODO: compare string samples
    return true;
  default:
    return compare_leaf_property_data(readerA, readerB);
  }
}

static bool compare_property(const std::string &path,
                             BasePropertyReaderPtr readerA,
                             BasePropertyReaderPtr readerB);

static bool compare_compound_property(const std::string &path,
                                      CompoundPropertyReaderPtr compoundA,
                                      CompoundPropertyReaderPtr compoundB) {
  std::set<std::string> namesA;
  std::set<std::string> namesB;
  for (size_t i = 0; i < compoundA->getNumProperties(); ++i) {
    auto child = compoundA->getProperty(i);
    namesA.insert(child->getName());
  }
  for (size_t i = 0; i < compoundB->getNumProperties(); ++i) {
    auto child = compoundB->getProperty(i);
    namesB.insert(child->getName());
  }
  // try to output as many differences as possible.
  bool different = false;

  for (auto name : namesA) {
    std::string childPath = path + '/' + name;
    if (!namesB.count(name)) {
      output << "- " << childPath << '\n';
      different = true;
    } else {
      auto propertyA = compoundA->getProperty(name);
      auto propertyB = compoundB->getProperty(name);
      if (!compare_property(childPath, propertyA, propertyB)) {
        different = true;
      }
    }
  }
  for (auto name : namesB) {
    if (!namesA.count(name)) {
      output << "- " << path << '/' << name << '\n';
      different = true;
    }
  }
  return !different;
}

static bool compare_property(const std::string &path,
                             BasePropertyReaderPtr readerA,
                             BasePropertyReaderPtr readerB) {
  verbose << "Comparing properties at " << path << '\n';

  auto propertyType = readerA->getPropertyType();
  if (readerB->getPropertyType() != propertyType) {
    output << "* " << path << '\n';
    return false;
  }

  bool sameLeaf = false;
  switch (propertyType) {
  case kCompoundProperty:
    return compare_compound_property(path, readerA->asCompoundPtr(),
                                     readerB->asCompoundPtr());
  case kScalarProperty:
    sameLeaf =
        compare_leaf_property(readerA->asScalarPtr(), readerB->asScalarPtr());
    break;
  case kArrayProperty:
    sameLeaf =
        compare_leaf_property(readerA->asArrayPtr(), readerB->asArrayPtr());
    break;
  default:
    // Theoretically impossible in Alembic 1.8.8 but maybe in the future...
    sameLeaf = false;
    break;
  }
  if (!sameLeaf) {
    output << "* " << path << '\n';
  }
  return sameLeaf;
}

bool compare_object(const std::string &path, ObjectReaderPtr readerA,
                    ObjectReaderPtr readerB) {
  verbose << "Comparing objects at " << path << '\n';
  bool same = true;

  // Properties are under one nameless root property.
  same = compare_property(path + '/', readerA->getProperties(),
                          readerB->getProperties());

  // Collect up the children.
  std::set<std::string> childrenA;
  std::set<std::string> childrenB;
  for (size_t i = 0; i < readerA->getNumChildren(); ++i) {
    childrenA.insert(readerA->getChild(i)->getName());
  }
  for (size_t i = 0; i < readerB->getNumChildren(); ++i) {
    childrenB.insert(readerB->getChild(i)->getName());
  }
  for (auto childName : childrenA) {
    std::string childPath = path + '/' + childName;
    if (!childrenB.count(childName)) {
      same = false;
      output << "- " << childPath << '\n';
    } else {
      if (!compare_object(childPath, readerA->getChild(childName),
                          readerB->getChild(childName))) {
        same = false;
      }
    }
  }
  for (auto childName : childrenB) {
    if (!childrenA.count(childName)) {
      same = false;
      output << "+ " << path << '/' << childName << '\n';
    }
  }

  return same;
}

// Read an alembic file, print its hierarchy in a human-readable format
bool compare_alembic(const std::string &filenameA,
                     const std::string &filenameB) {
  verbose << "Comparing " << filenameA << " and " << filenameB << '\n';

  // First, read. If we fail, just quit now with an error result.
  IFactory factory;
  IArchive archiveA = factory.getArchive(filenameA);
  IArchive archiveB = factory.getArchive(filenameB);
  if (!archiveA.valid()) {
    std::cerr << filenameA << " is not a valid Alembic file\n";
    return false;
  }
  if (!archiveB.valid()) {
    std::cerr << filenameB << " is not a valid Alembic file\n";
    return false;
  }

  return compare_object("", archiveA.getTop().getPtr(),
                        archiveB.getTop().getPtr());
}

void usage() {
  std::cerr << "USAGE: abc-compare [--verbose|--quiet] file1.abc file2.abc\n";
}

int main(int argc, const char *const *argv) {
  std::vector<std::string> file_args;
  bool process_flags = true;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (process_flags) {
      if (arg == "-v" || arg == "--verbose") {
        verbose.make_verbose();
        continue;
      } else if (arg == "-q" || arg == "--quiet") {
        output.make_quiet();
        continue;
      } else if (arg == "-h" || arg == "--help") {
        usage();
        return 0;
      } else if (arg == "--") {
        process_flags = false;
        continue;
      } else if (arg[0] == '-') {
        usage();
        return 1;
      }
    }
    // not a flag -- must be a file argument
    file_args.push_back(arg);
  }
  if (file_args.size() != 2) {
    usage();
    return 1;
  }
  std::string filenameA = file_args[0];
  std::string filenameB = file_args[1];
  bool same = compare_alembic(filenameA, filenameB);
  return same ? 0 : 1;
}
